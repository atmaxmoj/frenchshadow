"""Dedicated CosyVoice3 service for shadow-reader voice cloning.

This service lives outside the main FastAPI backend because CosyVoice3 needs a
separate Python environment (CosyVoice + Matcha-TTS on the path) and is
expensive to load.  It is started by `start.sh` on port 8769 and exposes a
single `/clone` endpoint that turns a user recording + target text into an audio
file that sounds like the user speaking the target text fluently.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

# CosyVoice3 must import from the user's tts-bench checkout.
COSY_BASE = Path("/Users/wangsijie/Develop/tools/tts-bench")
COSYVOICE = COSY_BASE / "CosyVoice"
MATCHA = COSYVOICE / "third_party" / "Matcha-TTS"
MODEL_DIR = COSY_BASE / "models" / "CosyVoice3-0.5B"

for p in (str(COSYVOICE), str(MATCHA)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cosy3_util import evict_cache, instruct_for

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "audio_cache" / "cosy3_service"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bound the on-disk output accumulation (configurable, default 500 MB).
OUTPUT_MAX_BYTES = int(os.environ.get("SHADOW_READER_COSY_CACHE_MAX_MB", "500")) * 1024 * 1024

app = FastAPI(title="shadow-reader CosyVoice3 service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated during startup lifespan.
_model: "CosyVoice3" | None = None
_sample_rate: int = 24000


class CloneRequest(BaseModel):
    target_text: str
    prompt_text: str
    ref_path: str
    language: str = ""


def _to_cosy_wav(src: str | Path, dst: str | Path) -> Path:
    """Convert any ffmpeg-readable audio to 24 kHz mono WAV for CosyVoice3."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"reference audio not found: {src}")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-ar", "24000", "-ac", "1", str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


@app.on_event("startup")
def startup() -> None:
    global _model, _sample_rate
    if not MODEL_DIR.exists():
        raise RuntimeError(f"CosyVoice3 model not found at {MODEL_DIR}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required by cosy3_service")

    logger.info("Loading CosyVoice3 from %s...", MODEL_DIR)
    from cosyvoice.cli.cosyvoice import CosyVoice3
    _model = CosyVoice3(str(MODEL_DIR), load_trt=False, fp16=False)
    _sample_rate = _model.sample_rate
    logger.info("CosyVoice3 ready on %s; sample_rate=%d", torch.device("cuda" if torch.cuda.is_available() else "cpu"), _sample_rate)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": str(MODEL_DIR),
        "loaded": _model is not None,
        "sample_rate": _sample_rate,
    }


@app.post("/clone")
def clone(req: CloneRequest) -> FileResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="CosyVoice3 model not loaded")
    if not req.target_text or not req.target_text.strip():
        raise HTTPException(status_code=400, detail="empty target_text")
    if not req.prompt_text or not req.prompt_text.strip():
        raise HTTPException(status_code=400, detail="empty prompt_text")
    ref_src = Path(req.ref_path)
    if not ref_src.exists():
        raise HTTPException(status_code=404, detail=f"reference audio not found: {ref_src}")

    start = time.time()
    job_id = uuid.uuid4().hex
    ref_wav = OUTPUT_DIR / f"{job_id}_ref.wav"
    out_wav = OUTPUT_DIR / f"{job_id}_clone.wav"
    # Evict oldest outputs before adding a new one (never touches out_wav).
    evict_cache(OUTPUT_DIR, OUTPUT_MAX_BYTES)
    try:
        _to_cosy_wav(ref_src, ref_wav)
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg conversion failed: %s", exc)
        raise HTTPException(status_code=400, detail="reference audio conversion failed") from exc

    # When the language is known, pin it with an instruct so the model does not
    # guess (French zero-shot clones came out sounding German). instruct2 still
    # clones the voice from the reference audio; it just drops the prompt
    # transcript in favour of the language directive.
    instruct = instruct_for(req.language)
    try:
        if instruct:
            logger.info("clone with language instruct: %s", req.language)
            generator = _model.inference_instruct2(
                req.target_text.strip(),
                instruct,
                str(ref_wav),
                stream=False,
                text_frontend=True,
            )
        else:
            prompt = "You are a helpful assistant.<|endofprompt|>" + req.prompt_text.strip()
            generator = _model.inference_zero_shot(
                req.target_text.strip(),
                prompt,
                str(ref_wav),
                stream=False,
                text_frontend=True,
            )
        for chunk in generator:
            torchaudio.save(str(out_wav), chunk["tts_speech"], _sample_rate)
            break
    except Exception as exc:
        logger.exception("CosyVoice3 inference failed")
        raise HTTPException(status_code=500, detail=f"voice cloning failed: {exc}") from exc
    finally:
        ref_wav.unlink(missing_ok=True)

    if not out_wav.exists():
        raise HTTPException(status_code=500, detail="no audio generated")
    elapsed = time.time() - start
    logger.info("clone generated %s in %.2fs", out_wav, elapsed)
    return FileResponse(out_wav, media_type="audio/wav", filename="clone.wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

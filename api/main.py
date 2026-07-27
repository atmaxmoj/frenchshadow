"""FastAPI backend for shadow-reader pronunciation practice."""

from __future__ import annotations

import logging
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.analyzer import analyze
from src.articulatory import attach_tips
from src.models import TARGET_SR, load_audio, load_model, transcribe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

processor: Any | None = None
model: Any | None = None
model_device: str = "cpu"

STATIC_DIR = Path(__file__).with_suffix("").parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global processor, model, model_device
    processor, model, model_device = load_model()
    logger.info("shadow-reader backend ready on %s", model_device)
    yield


app = FastAPI(title="shadow-reader backend", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "device": model_device,
        "loaded": model is not None,
    }


def _attach_word_times(analysis: dict, duration_s: float, token_times: list[float]) -> dict:
    """Add start_time / end_time to each word for KTV-style playback."""
    n_tokens = len(token_times)
    if n_tokens == 0:
        avg = 0.0
    else:
        avg = duration_s / n_tokens

    for word in analysis.get("words", []):
        start_idx = word.get("learner_start", 0)
        end_idx = word.get("learner_end", start_idx)

        if start_idx < n_tokens:
            word["start_time"] = token_times[start_idx]
        else:
            word["start_time"] = duration_s

        if end_idx > 0 and end_idx - 1 < n_tokens:
            word["end_time"] = min(token_times[end_idx - 1] + avg, duration_s)
        else:
            word["end_time"] = duration_s

        # Ensure monotonic within the word
        if word["end_time"] < word["start_time"]:
            word["end_time"] = word["start_time"]

    return analysis


@app.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),
    target_text: str | None = Form(None),
    language: str = Form("en-us"),
) -> dict:
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty audio")

    try:
        wav = load_audio(raw)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")[:300]
        raise HTTPException(status_code=400, detail=f"ffmpeg decode failed: {stderr}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"audio decode failed: {e}")

    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    duration_s = len(wav) / TARGET_SR
    try:
        result = transcribe(wav, processor=processor, model=model)
    except Exception as e:
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    response = {
        "duration_s": round(duration_s, 2),
        "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "tokens": result["tokens"],
    }

    if target_text:
        try:
            analysis = analyze(target_text, result["tokens"], language=language)
            attach_tips(analysis)
            _attach_word_times(analysis, duration_s, result["token_times"])
            response["analysis"] = analysis
        except Exception as e:
            logger.exception("analysis failed")
            response["analysis_error"] = str(e)

    return response

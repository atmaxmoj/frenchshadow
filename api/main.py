"""FastAPI backend for shadow-reader pronunciation practice."""

from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.analyzer import analyze, reference_ipa_per_word
from src.articulatory import attach_tips
from src.audio import reduce_noise
from src.models import TARGET_SR, load_audio, load_model, transcribe
from src.liaison import detect_liaisons, reference_text_for_word
from src.diagrams import diagram as mouth_diagram, has_diagram
from src.tts import synthesize
from src.youtube import TranscriptError, extract_video_id, fetch_transcript, fetch_video_info

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
    """Add start_time / end_time to each word for KTV-style playback.

    The learner_start / learner_end indices come from the alignment between
    reference and learner phones. We map those indices back to seconds, then
    clamp each word so it does not spill into the next word's time region.
    Words with no matched learner tokens get a zero-duration slice instead of
    spanning the whole recording.
    """
    n_tokens = len(token_times)
    avg = duration_s / max(n_tokens, 1)

    words = analysis.get("words", [])
    for word in words:
        start_idx = word.get("learner_start", 0)
        end_idx = word.get("learner_end", start_idx)

        if start_idx < n_tokens:
            word["start_time"] = token_times[start_idx]
        else:
            word["start_time"] = duration_s

        if end_idx > start_idx and end_idx - 1 < n_tokens:
            word["end_time"] = min(token_times[end_idx - 1] + avg, duration_s)
        else:
            # No learner tokens mapped to this word: do not span the whole audio.
            word["end_time"] = word["start_time"]

    # Clamp word boundaries so each word ends before the next one starts.
    sorted_words = sorted(enumerate(words), key=lambda x: x[1]["start_time"])
    for i, (_, word) in enumerate(sorted_words):
        next_start = sorted_words[i + 1][1]["start_time"] if i + 1 < len(sorted_words) else duration_s
        word["end_time"] = min(word["end_time"], next_start)
        # Ensure monotonic within the word.
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
        wav = reduce_noise(wav)
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
            attach_tips(analysis, language=language)
            _attach_word_times(analysis, duration_s, result["token_times"])
            response["analysis"] = analysis
        except Exception as e:
            logger.exception("analysis failed")
            response["analysis_error"] = str(e)

    return response


@app.get("/liaisons")
def liaisons(sentence: str, language: str = "fr-fr") -> dict:
    """Return liaison word pairs for *sentence*.

    Each pair is (word1, word2) indicating that word1 gains a liaison consonant
    before word2 in fluent speech.
    """
    return {"sentence": sentence, "liaisons": detect_liaisons(sentence, language)}


@app.get("/youtube/info")
def youtube_info(url: str, language: str = "fr-fr") -> dict:
    """Return metadata and available transcript languages for a YouTube URL."""
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="empty URL")
    try:
        return fetch_video_info(url.strip(), preferred_language=language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("youtube info failed")
        raise HTTPException(status_code=500, detail=f"youtube info failed: {exc}") from exc


@app.get("/youtube/transcript")
def youtube_transcript(video_id: str, language: str = "fr-fr") -> dict:
    """Return sentence-level transcript with word timings for a YouTube video."""
    if not video_id or not video_id.strip():
        raise HTTPException(status_code=400, detail="empty video_id")
    try:
        return fetch_transcript(video_id.strip(), language=language)
    except TranscriptError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("youtube transcript failed")
        raise HTTPException(status_code=500, detail=f"youtube transcript failed: {exc}") from exc


@app.get("/word_ipa")
def word_ipa(word: str, language: str = "fr-fr") -> dict:
    """Return the IPA pronunciation for a single *word*."""
    if not word or not word.strip():
        raise HTTPException(status_code=400, detail="empty word")
    try:
        pairs = reference_ipa_per_word(word.strip(), language=language)
        if not pairs:
            return {"word": word, "ipa": ""}
        return {"word": pairs[0][0], "ipa": "".join(pairs[0][1])}
    except Exception as exc:
        logger.exception("word_ipa failed")
        raise HTTPException(status_code=500, detail=f"word_ipa failed: {exc}") from exc


@app.get("/mouth_diagram")
def mouth_diagram_endpoint(phone: str) -> Response:
    """Return an SVG articulatory diagram for an IPA phone."""
    if not phone or not phone.strip() or phone.strip() in {"-", "_", "?"}:
        raise HTTPException(status_code=400, detail="empty or invalid phone")
    if not has_diagram(phone):
        raise HTTPException(status_code=404, detail=f"no diagram for phone {phone}")
    try:
        svg = mouth_diagram(phone)
    except Exception as exc:
        logger.exception("mouth diagram failed")
        raise HTTPException(status_code=500, detail=f"diagram failed: {exc}") from exc
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return Response(content=svg, media_type="image/svg+xml", headers=headers)


def _synthesize_to_b64(text: str, language: str) -> str | None:
    """Synthesize *text* and return base64-encoded WAV bytes, or None on failure."""
    try:
        data = synthesize(text, language=language)
        return base64.b64encode(data).decode("ascii")
    except Exception as exc:
        logger.warning("pre-bake TTS failed for %r: %s", text, exc)
        return None


@app.get("/prebake_reference")
async def prebake_reference(sentence: str, language: str = "fr-fr") -> dict:
    """Pre-synthesize reference audio for every word in *sentence*.

    Returns a JSON object `{audios: {word: base64wav, ...}}`.  The frontend can
    turn each base64 blob into an object URL and play it instantly when the user
    clicks the reference-speaker button.
    """
    if not sentence or not sentence.strip():
        raise HTTPException(status_code=400, detail="empty sentence")

    words = sentence.split()
    if not words:
        raise HTTPException(status_code=400, detail="empty sentence")

    async def _bake(word: str) -> tuple[str, str | None]:
        tts_text = reference_text_for_word(sentence, word, language)
        b64 = await asyncio.to_thread(_synthesize_to_b64, tts_text, language)
        return word, b64

    baked = await asyncio.gather(*[_bake(w) for w in words])
    audios = {word: b64 for word, b64 in baked if b64}
    return {"sentence": sentence, "language": language, "audios": audios}


@app.get("/reference_audio")
def reference_audio(
    text: str,
    language: str = "en-us",
    sentence: str | None = None,
) -> Response:
    """Synthesize *text* with a high-quality neural voice and return a WAV file.

    If *sentence* is provided and *text* is part of a French liaison pair, the
    synthesized text includes the following word so the liaison consonant is
    audible.
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    if not shutil.which("edge-tts") and not shutil.which("say"):
        raise HTTPException(status_code=503, detail="no TTS backend available (edge-tts or macOS say)")

    tts_text = reference_text_for_word(sentence or text, text, language)
    try:
        data = synthesize(tts_text, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

    return Response(content=data, media_type="audio/wav")

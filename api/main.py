"""FastAPI backend for shadow-reader pronunciation practice."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse

from src.analyzer import analyze, reference_ipa_per_word
from src.articulatory import attach_tips
from src.gop import gop_scores
from src.intelligibility import score as intelligibility_score
from src.grapheme import mark_graphemes
from src.phoneme_audio import phoneme_wav
from src.models import TARGET_SR, load_audio, load_model, transcribe
from src.liaison import detect_liaisons, reference_text_for_word
from src.diagrams import diagram as mouth_diagram, has_diagram
from src.tts import synthesize
from src.translate import translate_sentences
from src.youtube import TranscriptError, extract_video_id, fetch_transcript, fetch_video_info
from src.storage import (
    get_attempts,
    get_recording_path,
    get_recent_videos,
    get_stats,
    get_video_progress,
    save_attempt,
    touch_video,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

processor: Any | None = None
model: Any | None = None
model_device: str = "cpu"

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


@app.middleware("http")
async def log_requests(request, call_next):
    """Log every request so the 8767 backend logs are actionable."""
    start = asyncio.get_event_loop().time()
    response = await call_next(request)
    elapsed = (asyncio.get_event_loop().time() - start) * 1000
    logger.info(
        "%s %s %s - %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response

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

    logger.info(
        "transcribe request: %d bytes, filename=%s, content_type=%s",
        len(raw),
        audio.filename or "",
        audio.content_type or "",
    )

    # No extra noise reduction: the browser mic already applies noiseSuppression,
    # and a second spectral-gating pass drops phones the acoustic model would catch.
    try:
        wav = await asyncio.to_thread(load_audio, raw)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")[:500]
        logger.warning("audio decode (ffmpeg) failed: %s", stderr)
        raise HTTPException(
            status_code=400,
            detail=f"无法识别这段录音，请确认麦克风正常工作并多读一会儿。详情：{stderr}",
        ) from e
    except ValueError as e:
        logger.warning("audio decode validation failed: %s", e)
        raise HTTPException(
            status_code=400,
            detail=f"录音内容为空或太短，请重新跟读。详情：{e}",
        ) from e
    except Exception as e:
        logger.exception("audio decode failed")
        raise HTTPException(status_code=400, detail=f"audio decode failed: {e}") from e

    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    duration_s = len(wav) / TARGET_SR
    try:
        # CPU-bound inference off the event loop.
        result = await asyncio.to_thread(transcribe, wav, processor, model)
    except Exception as e:
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    response = {
        "duration_s": round(duration_s, 2),
        "model": "facebook/wav2vec2-lv-60-espeak-cv-ft",
        "tokens": result["tokens"],
    }

    if target_text:
        def _analyze():
            # GOP: force-align the canonical phones to the audio posteriors and
            # score each by acoustic confidence (partial credit, accent-tolerant).
            phones = [p for _, ps in reference_ipa_per_word(target_text, language=language) for p in ps]
            gop = gop_scores(result["logp"], phones, processor, result["blank_id"])
            a = analyze(target_text, result["tokens"], language=language, gop=gop)
            attach_tips(a, language=language)
            _attach_word_times(a, duration_s, result["token_times"])
            # 达意: context-aware intelligibility via Whisper (what a listener hears).
            # Optional — silently skipped if the Whisper model isn't present.
            try:
                intel = intelligibility_score(wav, target_text, language)
                if intel:
                    a["intelligibility"] = intel["overall"]
                    a["heard"] = intel["heard"]
                    for i, w in enumerate(a["words"]):
                        if i < len(intel["per_word"]):
                            w["intelligibility"] = intel["per_word"][i]
            except Exception:
                logger.exception("intelligibility failed")
            return a

        try:
            response["analysis"] = await asyncio.to_thread(_analyze)
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


@app.post("/attempts")
async def create_attempt(
    audio: UploadFile = File(...),
    video_id: str = Form(...),
    sentence_idx: int = Form(...),
    sentence_text: str = Form(...),
    language: str = Form("fr-fr"),
    analysis: str = Form(...),
    duration_s: float = Form(0),
    title: str = Form(""),
    thumbnail: str = Form(""),
    total_sentences: int = Form(0),
) -> dict:
    """Persist a practice attempt (audio + analysis) and return its id."""
    try:
        analysis_data = json.loads(analysis)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid analysis JSON: {exc}") from exc

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio")

    try:
        touch_video(
            video_id=video_id,
            title=title,
            thumbnail=thumbnail,
            language=language,
            total_sentences=total_sentences,
            last_sentence_idx=sentence_idx,
        )
        attempt = save_attempt(
            video_id=video_id,
            sentence_idx=sentence_idx,
            sentence_text=sentence_text,
            language=language,
            audio_bytes=audio_bytes,
            analysis=analysis_data,
            duration_s=duration_s,
        )
    except Exception as exc:
        logger.exception("save_attempt failed")
        raise HTTPException(status_code=500, detail=f"save failed: {exc}") from exc

    return {
        "id": attempt.id,
        "video_id": attempt.video_id,
        "sentence_idx": attempt.sentence_idx,
        "overall_score": attempt.overall_score,
        "created_at": attempt.created_at,
    }


@app.get("/stats")
def stats() -> dict:
    """Return aggregate practice statistics for the dashboard."""
    try:
        return get_stats()
    except Exception as exc:
        logger.exception("stats failed")
        raise HTTPException(status_code=500, detail=f"stats failed: {exc}") from exc


@app.get("/recent_videos")
def recent_videos(limit: int = 20) -> dict:
    """Return recently practiced videos for the dashboard."""
    try:
        videos = get_recent_videos(limit=limit)
    except Exception as exc:
        logger.exception("recent_videos failed")
        raise HTTPException(status_code=500, detail=f"recent_videos failed: {exc}") from exc

    return {
        "videos": [
            {
                "video_id": v.video_id,
                "title": v.title,
                "thumbnail": v.thumbnail,
                "language": v.language,
                "total_sentences": v.total_sentences,
                "last_sentence_idx": v.last_sentence_idx,
                "last_practiced_at": v.last_practiced_at,
                "attempt_count": v.attempt_count,
                "sentence_attempt_count": v.sentence_attempt_count,
            }
            for v in videos
        ]
    }


@app.get("/videos/{video_id}/progress")
def video_progress(video_id: str) -> dict:
    """Return progress for a single video."""
    progress = get_video_progress(video_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="video not found")
    return {
        "video_id": progress.video_id,
        "title": progress.title,
        "thumbnail": progress.thumbnail,
        "language": progress.language,
        "total_sentences": progress.total_sentences,
        "last_sentence_idx": progress.last_sentence_idx,
        "last_practiced_at": progress.last_practiced_at,
        "attempt_count": progress.attempt_count,
        "sentence_attempt_count": progress.sentence_attempt_count,
    }


@app.get("/attempts")
def list_attempts(video_id: str, sentence_idx: int | None = None) -> dict:
    """List persisted attempts for a video, optionally filtered by sentence."""
    if not video_id:
        raise HTTPException(status_code=400, detail="empty video_id")
    try:
        attempts = get_attempts(video_id, sentence_idx=sentence_idx)
    except Exception as exc:
        logger.exception("list_attempts failed")
        raise HTTPException(status_code=500, detail=f"list failed: {exc}") from exc

    return {
        "video_id": video_id,
        "sentence_idx": sentence_idx,
        "attempts": [
            {
                "id": a.id,
                "sentence_idx": a.sentence_idx,
                "sentence_text": a.sentence_text,
                "overall_score": a.overall_score,
                "analysis": a.analysis,
                "created_at": a.created_at,
            }
            for a in attempts
        ],
    }


@app.get("/attempts/{attempt_id}/audio")
def attempt_audio(attempt_id: str) -> FileResponse:
    """Stream the recorded audio for a persisted attempt."""
    path = get_recording_path(attempt_id)
    if path is None:
        raise HTTPException(status_code=404, detail="recording not found")
    return FileResponse(path, media_type="audio/webm", filename=path.name)


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


class PrebakeItem(BaseModel):
    text: str
    sentence: str = ""


class PrebakeRequest(BaseModel):
    language: str = "fr-fr"
    items: list[PrebakeItem]


@app.post("/prebake")
async def prebake(req: PrebakeRequest) -> dict:
    """Warm the TTS cache for a batch of words/phrases.

    The frontend fires this after loading a transcript so individual word
    reference audio plays instantly when the user clicks a word.  Synthesis is
    cached on disk, so duplicate requests are no-ops.
    """
    if not req.items:
        return {"queued": 0}

    seen: set[str] = set()

    def _bake(item: PrebakeItem) -> None:
        key = (req.language, item.text, item.sentence)
        if key in seen:
            return
        seen.add(key)
        try:
            tts_text = reference_text_for_word(item.sentence or item.text, item.text, req.language)
            synthesize(tts_text, language=req.language)
        except Exception as exc:
            logger.debug("prebake failed for %r: %s", item.text, exc)

    # Run the batch in a thread pool: TTS is I/O + subprocess bound, not async.
    await asyncio.to_thread(lambda: [_bake(i) for i in req.items])
    return {"queued": len(seen)}


class TranslateRequest(BaseModel):
    sentences: list[str]
    target: str = "zh"
    source_hint: str = "auto"


@app.post("/translate")
async def translate(req: TranslateRequest) -> dict:
    """Translate a list of sentences to the requested target language."""
    cleaned = [s.strip() for s in req.sentences]
    if not any(cleaned):
        return {"translations": [""] * len(cleaned)}
    translations = await asyncio.to_thread(
        translate_sentences, cleaned, req.source_hint, req.target
    )
    return {"translations": translations}


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


# --- Grapheme highlighting + isolated/syllable phoneme audio -----------------

class GraphemePair(BaseModel):
    word: str
    phone: str


class GraphemeRequest(BaseModel):
    pairs: list[GraphemePair]
    language: str = "fr-fr"


@app.post("/grapheme")
async def grapheme(req: GraphemeRequest) -> dict:
    """For each (word, phone), return the word with that sound's letters marked."""
    pairs = [(p.word, p.phone) for p in req.pairs]
    marks = await asyncio.to_thread(mark_graphemes, pairs, req.language)
    return {"marks": marks}


@app.get("/phoneme_audio")
def phoneme_audio(ipa: str, language: str = "fr-fr") -> Response:
    """Play an IPA phone, or a space-separated syllable (e.g. 'm ɑ̃'), via espeak."""
    if not ipa or not ipa.strip():
        raise HTTPException(status_code=400, detail="empty ipa")
    data = phoneme_wav(ipa.strip(), language)
    if data is None:
        raise HTTPException(status_code=404, detail="no audio for phone")
    return Response(content=data, media_type="audio/wav")

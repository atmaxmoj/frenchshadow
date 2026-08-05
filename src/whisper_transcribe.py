"""Whisper-based word-level transcript alignment for shadow-reading.

Downloads YouTube audio, runs the local Whisper model, and aligns the real
audio word timings to the caption text. This keeps the caption's accurate
punctuation/text while replacing its evenly-divided durations with timings from
the actual speech.
"""

from __future__ import annotations

import difflib
import json
import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from src.youtube import Sentence, WordToken

logger = logging.getLogger(__name__)

_MODELS = Path(__file__).resolve().parent.parent / "models"
_CACHE = Path(__file__).resolve().parent.parent / "audio_cache"
_LOCK = threading.Lock()
_PROC: Any | None = None
_MODEL: Any | None = None
_LOADED_DIR: str | None = None

# Prefer small for accuracy; fall back to base if small is missing/corrupt.
_CANDIDATE_NAMES = ("whisper-small", "whisper-base")

# Cap any single word duration after alignment to avoid garbage from bad matches.
_MAX_WORD_DURATION_S = 2.5

# Sentence segmentation parameters (same semantics as src.youtube rule-based splitter).
_MAX_SENTENCE_WORDS = 20
_PAUSE_THRESHOLD_S = 1.2

# Whisper timestamp token IDs for the 30-second model are in this range.
_TIMESTAMP_TOKEN_RANGE = range(50365, 51866)


def _model_dir() -> Path | None:
    for name in _CANDIDATE_NAMES:
        d = _MODELS / name
        if (d / "model.safetensors").exists():
            return d
    return None


def available() -> bool:
    return _model_dir() is not None


def _ensure_model() -> tuple[Any, Any]:
    global _PROC, _MODEL, _LOADED_DIR
    if _PROC is not None and _MODEL is not None:
        return _PROC, _MODEL
    with _LOCK:
        if _PROC is not None and _MODEL is not None:
            return _PROC, _MODEL
        d = _model_dir()
        if d is None:
            raise RuntimeError("no local whisper model found")
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("Loading Whisper model from %s", d)
        proc = WhisperProcessor.from_pretrained(str(d))
        model = WhisperForConditionalGeneration.from_pretrained(str(d)).eval()
        _PROC, _MODEL, _LOADED_DIR = proc, model, d.name
        logger.info("Whisper model loaded: %s", d.name)
        return _PROC, _MODEL


def _find_node() -> str | None:
    return shutil.which("node")


def _download_audio(video_id: str) -> Path:
    """Download a video's audio to WAV, cached under audio_cache/."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    cached = _CACHE / f"{video_id}.wav"
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    out_template = str(_CACHE / f"{video_id}.tmp")
    node = _find_node()
    cmd = [
        "yt-dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", out_template,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    if node:
        cmd.extend(["--js-runtimes", node])

    logger.info("Downloading audio for %s", video_id)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error("yt-dlp failed: %s", result.stderr[-1000:])
        raise RuntimeError(f"failed to download audio for {video_id}: {result.stderr[-500:]}")

    tmp = Path(f"{out_template}.wav")
    if not tmp.exists():
        raise RuntimeError(f"yt-dlp did not produce expected file {tmp}")
    tmp.rename(cached)
    logger.info("Audio cached: %s (%s bytes)", cached, cached.stat().st_size)
    return cached


def _load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32)


def _extract_words(segment: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    """Convert a Whisper segment with token timestamps into word-level entries."""
    tokens = segment["tokens"].tolist() if hasattr(segment["tokens"], "tolist") else segment["tokens"]
    timestamps = (
        segment["token_timestamps"].tolist()
        if hasattr(segment["token_timestamps"], "tolist")
        else segment["token_timestamps"]
    )
    segment_end = float(segment["end"])
    special_ids = set(tokenizer.all_special_ids)

    content: list[tuple[int, float]] = []
    for tid, ts in zip(tokens, timestamps):
        if tid in special_ids or tid in _TIMESTAMP_TOKEN_RANGE:
            continue
        content.append((tid, ts))

    words: list[dict[str, Any]] = []
    word_tids: list[int] = []
    word_times: list[float] = []

    for tid, ts in content:
        tok = tokenizer.convert_ids_to_tokens(tid)
        is_word_start = tok.startswith("Ġ") or not word_tids
        if is_word_start and word_tids:
            text = tokenizer.decode(word_tids, skip_special_tokens=True).strip()
            if text:
                words.append({"text": text, "start": word_times[0], "end": word_times[-1]})
            word_tids = []
            word_times = []
        word_tids.append(tid)
        word_times.append(ts)

    if word_tids:
        text = tokenizer.decode(word_tids, skip_special_tokens=True).strip()
        if text:
            words.append({"text": text, "start": word_times[0], "end": segment_end})

    # Each word's end is the next word's start; the last word keeps segment_end.
    for i in range(len(words) - 1):
        words[i]["end"] = words[i + 1]["start"]

    return words


def _get_whisper_words(audio: np.ndarray, language: str) -> list[dict[str, Any]]:
    """Run Whisper on audio in 30s chunks and return all word-level timings."""
    proc, model = _ensure_model()
    lang_code = {"fr-fr": "fr", "fr-ca": "fr", "en-us": "en", "en-gb": "en"}.get(
        language.lower(), language.split("-")[0]
    )

    sample_rate = 16000
    chunk_samples = 30 * sample_rate
    all_words: list[dict[str, Any]] = []

    for chunk_idx, offset in enumerate(range(0, len(audio), chunk_samples)):
        chunk = audio[offset : offset + chunk_samples]
        if len(chunk) < 0.5 * sample_rate:
            break

        inputs = proc(
            chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
                language=lang_code,
                task="transcribe",
                return_timestamps=True,
                return_token_timestamps=True,
                return_dict_in_generate=True,
            )

        time_offset = offset / sample_rate
        for segment_group in outputs["segments"]:
            for segment in segment_group:
                for w in _extract_words(segment, proc.tokenizer):
                    all_words.append({
                        "text": w["text"],
                        "start": w["start"] + time_offset,
                        "end": w["end"] + time_offset,
                    })

    return all_words


def _load_whisper_words(video_id: str, language: str) -> list[dict[str, Any]]:
    """Load cached Whisper words or run Whisper on the video audio."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE / f"{video_id}.whisper.json"
    if cache_path.exists():
        logger.info("Using cached Whisper words for %s", video_id)
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    audio_path = _download_audio(video_id)
    audio = _load_audio(audio_path)
    whisper_words = _get_whisper_words(audio, language)
    cache_path.write_text(json.dumps(whisper_words, ensure_ascii=False, indent=2), encoding="utf-8")
    return whisper_words


def _clean_word(text: str) -> str:
    """Lowercase and strip trailing punctuation for alignment matching."""
    return text.strip(".,!?;:\"'()[]{}«»…").lower()


def align_sentences(
    video_id: str,
    sentences: list[Sentence],
    language: str = "fr-fr",
) -> list[Sentence]:
    """Align pre-segmented caption sentences to Whisper word timings.

    Uses a global sequence alignment between the caption words and Whisper words,
    which handles overlapping YouTube caption entries better than per-sentence
    alignment. Returns a new list of sentences with accurate per-word start/end
    times and recomputed sentence boundaries.
    """
    if not sentences:
        return []

    whisper_words = _load_whisper_words(video_id, language)
    if not whisper_words:
        return sentences

    # Flatten base sentence words and remember which sentence each belongs to.
    base_words: list[WordToken] = []
    sentence_word_counts: list[int] = []
    for s in sentences:
        base_words.extend(s.words)
        sentence_word_counts.append(len(s.words))

    yt_clean = [_clean_word(w.text) for w in base_words]
    wh_clean = [_clean_word(w["text"]) for w in whisper_words]

    matcher = difflib.SequenceMatcher(None, yt_clean, wh_clean)
    matches = matcher.get_opcodes()

    # Map each base word index to a whisper index if matched.
    yt_to_whisper: dict[int, int] = {}
    for op, i1, i2, j1, j2 in matches:
        if op == "equal":
            for k, j in enumerate(range(j1, j2)):
                yt_to_whisper[i1 + k] = j

    matched_indices = sorted(yt_to_whisper.keys())
    aligned: list[WordToken] = []

    for i, wt in enumerate(base_words):
        if i in yt_to_whisper:
            w = whisper_words[yt_to_whisper[i]]
            start, end = w["start"], w["end"]
        else:
            prev_idx = next((k for k in reversed(matched_indices) if k < i), None)
            next_idx = next((k for k in matched_indices if k > i), None)
            if prev_idx is not None and next_idx is not None:
                p = whisper_words[yt_to_whisper[prev_idx]]
                n = whisper_words[yt_to_whisper[next_idx]]
                ratio = (i - prev_idx) / (next_idx - prev_idx)
                start = p["start"] + ratio * (n["start"] - p["start"])
                end = p["end"] + ratio * (n["end"] - p["end"])
            elif prev_idx is not None:
                p = whisper_words[yt_to_whisper[prev_idx]]
                start = p["end"]
                end = p["end"] + (wt.end - wt.start)
            elif next_idx is not None:
                n = whisper_words[yt_to_whisper[next_idx]]
                start = n["start"] - (wt.end - wt.start)
                end = n["start"]
            else:
                start = wt.start
                end = wt.end

        # Clamp overly long words caused by alignment failures.
        if end - start > _MAX_WORD_DURATION_S:
            end = start + _MAX_WORD_DURATION_S
        if end <= start:
            end = start + 0.05
        aligned.append(WordToken(text=wt.text, start=start, end=end))

    # Enforce global monotonic non-overlapping word boundaries.
    for i in range(len(aligned) - 1):
        if aligned[i].end > aligned[i + 1].start:
            aligned[i] = WordToken(
                text=aligned[i].text,
                start=aligned[i].start,
                end=aligned[i + 1].start,
            )

    # Distribute aligned words back into sentences using original word counts.
    result: list[Sentence] = []
    cursor = 0
    for idx, s in enumerate(sentences):
        count = sentence_word_counts[idx]
        sent_words = aligned[cursor : cursor + count]
        cursor += count
        if not sent_words:
            result.append(s)
            continue
        result.append(
            Sentence(
                text=s.text,
                start=sent_words[0].start,
                end=sent_words[-1].end,
                words=sent_words,
            )
        )

    return result


def _ends_with_weak_boundary(buffer: list[WordToken]) -> bool:
    if not buffer:
        return False
    return buffer[-1].text.rstrip()[-1] in ",;:"


def segment_whisper_words(words: list[dict[str, Any]]) -> list[Sentence]:
    """Group Whisper word-level timings into sentences.

    Mirrors the rule-based logic in src.youtube: terminal punctuation always
    ends a sentence, weak punctuation (comma/semicolon/colon) is ignored unless
    the buffer would exceed the comfortable word limit, and long pauses split
    only when we are not mid-comma-phrase.
    """
    if not words:
        return []

    sentences: list[Sentence] = []
    buffer: list[WordToken] = []
    text_buffer: list[str] = []
    prev_end = 0.0

    def flush() -> None:
        nonlocal buffer, text_buffer
        if not buffer:
            return
        sentences.append(
            Sentence(
                text=" ".join(text_buffer),
                start=buffer[0].start,
                end=buffer[-1].end,
                words=list(buffer),
            )
        )
        buffer = []
        text_buffer = []

    for w in words:
        raw_text = w["text"].strip()
        if not raw_text:
            continue
        start = float(w["start"])
        end = float(w["end"])

        # Long silence starts a new sentence unless we're in a comma-separated phrase.
        if buffer and start - prev_end > _PAUSE_THRESHOLD_S and not _ends_with_weak_boundary(buffer):
            flush()

        # Store display text with punctuation, but a clean token for analysis/highlighting.
        clean_text = raw_text.rstrip(".,!?;:\"'()[]{}«»…")
        if not clean_text:
            clean_text = raw_text
        buffer.append(WordToken(text=clean_text, start=start, end=end))
        text_buffer.append(raw_text)

        # Hard cap: flush before the next chunk would overflow.
        if len(buffer) >= _MAX_SENTENCE_WORDS:
            flush()
        elif raw_text[-1] in ".!?…":
            flush()

        prev_end = end

    flush()
    return sentences


def transcribe_video(video_id: str, language: str = "fr-fr") -> dict[str, Any]:
    """Return a Whisper-only transcript dict for a YouTube video (legacy/test path)."""
    whisper_words = _load_whisper_words(video_id, language)
    sentences = segment_whisper_words(whisper_words)
    return {
        "video_id": video_id,
        "language": language,
        "sentence_count": len(sentences),
        "sentences": sentences,
    }

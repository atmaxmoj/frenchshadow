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
import re
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
# Keep shadow-reading chunks short enough to hold in working memory.
_MAX_SENTENCE_WORDS = 12
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


_STRONG_END = re.compile(r"(?<=[.!?…])\s+")
_WEAK_END = re.compile(r"(?<=[,;])\s+")
_WORD_SPLIT = re.compile(r"\s+")


def _is_punctuation_sparse(text: str, threshold: float = 0.15) -> bool:
    """Return True if *text* has very few terminal punctuation marks."""
    words = text.split()
    if not words:
        return False
    terminals = sum(1 for w in words if w and w[-1] in ".!?…")
    return terminals / len(words) < threshold


def _split_punctuated_text(text: str) -> list[tuple[str, int, bool]]:
    """Split restored/annotated text into sentence-sized pieces.

    Strong boundaries (. ! ? …) always end a sentence. Weak boundaries (, ;)
    are used to keep chunks short enough for shadow-reading: we flush at a weak
    boundary once the buffer is at least half the maximum comfortable size.
    Returns (piece_text, word_count, is_terminal).
    """
    text = text.replace("\n", " ").strip()
    if not text:
        return []

    words = text.split()
    result: list[tuple[str, int, bool]] = []
    buffer: list[str] = []
    soft_limit = _MAX_SENTENCE_WORDS // 2

    for i, word in enumerate(words):
        buffer.append(word)
        buf_count = len(buffer)
        last_char = word[-1] if word else ""
        is_terminal = last_char in ".!?"
        is_weak = last_char in ",;:"

        # Terminal punctuation or hard cap always flushes.
        if is_terminal or buf_count >= _MAX_SENTENCE_WORDS:
            joined = " ".join(buffer)
            result.append((joined, buf_count, is_terminal))
            buffer = []
            continue

        # At a weak boundary, flush once we have a comfortable chunk so the
        # next comma phrase does not make the sentence too long.
        if is_weak and buf_count >= soft_limit:
            joined = " ".join(buffer)
            result.append((joined, buf_count, False))
            buffer = []

    if buffer:
        joined = " ".join(buffer)
        result.append((joined, len(buffer), joined[-1] in ".!?"))

    return result


def segment_whisper_words(words: list[dict[str, Any]]) -> list[Sentence]:
    """Group Whisper word-level timings into sentences.

    If Whisper produced sparse punctuation (common with base/small on clean
    speech), restore punctuation with the local model first, then split at
    real sentence boundaries. This prevents long unpunctuated runs from
    becoming one giant sentence.
    """
    if not words:
        return []

    # Build both raw and clean texts. Raw keeps Whisper's own punctuation; clean
    # is what we feed the restoration model if the raw output is too sparse.
    raw_texts: list[str] = []
    clean_texts: list[str] = []
    for w in words:
        t = w["text"].strip()
        raw_texts.append(t)
        clean = t.rstrip(".,!?;:\"'()[]{}«»…")
        clean_texts.append(clean if clean else t)

    raw_full_text = " ".join(raw_texts)
    clean_full_text = " ".join(clean_texts)

    # Restore punctuation if Whisper didn't provide enough.
    try:
        from src.punct import has_model, restore_punctuation

        if has_model() and _is_punctuation_sparse(raw_full_text):
            punctuated = restore_punctuation(clean_full_text)
        else:
            punctuated = raw_full_text
    except Exception as exc:
        logger.warning("punctuation restoration failed: %s", exc)
        punctuated = raw_full_text

    pieces = _split_punctuated_text(punctuated)
    if not pieces:
        return []

    sentences: list[Sentence] = []
    word_cursor = 0
    for piece_text, word_count, _ in pieces:
        if word_cursor >= len(words):
            break
        piece_words = words[word_cursor : word_cursor + word_count]
        word_cursor += word_count
        if not piece_words:
            continue

        clean_tokens: list[WordToken] = []
        for w in piece_words:
            raw_text = w["text"].strip()
            clean_text = raw_text.rstrip(".,!?;:\"'()[]{}«»…")
            if not clean_text:
                clean_text = raw_text
            clean_tokens.append(WordToken(text=clean_text, start=w["start"], end=w["end"]))

        sentences.append(
            Sentence(
                text=piece_text,
                start=clean_tokens[0].start,
                end=clean_tokens[-1].end,
                words=clean_tokens,
            )
        )

    # Append any leftover words (e.g. if punctuation model dropped some).
    if word_cursor < len(words):
        leftover = words[word_cursor:]
        clean_tokens = []
        display_texts: list[str] = []
        for w in leftover:
            raw_text = w["text"].strip()
            display_texts.append(raw_text)
            clean_text = raw_text.rstrip(".,!?;:\"'()[]{}«»…")
            if not clean_text:
                clean_text = raw_text
            clean_tokens.append(WordToken(text=clean_text, start=w["start"], end=w["end"]))
        sentences.append(
            Sentence(
                text=" ".join(display_texts),
                start=clean_tokens[0].start,
                end=clean_tokens[-1].end,
                words=clean_tokens,
            )
        )

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

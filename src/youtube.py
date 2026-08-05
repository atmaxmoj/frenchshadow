"""YouTube transcript extraction and sentence segmentation for shadow-reading.

Tries to fetch human or auto-generated captions via youtube-transcript-api.  If
no captions are available, falls back to language detection on the video title
so the UI can ask the user for a transcript or run STT later.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from src.punct import has_model as _has_punct_model, punctuation_is_sparse, restore_punctuation

logger = logging.getLogger(__name__)


_STRONG_END = re.compile(r"(?<=[.!?…])\s+")
_WEAK_END = re.compile(r"(?<=[,;])\s+")
_WORD_SPLIT = re.compile(r"\s+")
_MAX_SENTENCE_WORDS = 15
_PAUSE_THRESHOLD_S = 1.2


class TranscriptError(Exception):
    """Raised when a transcript cannot be retrieved."""


@dataclass(frozen=True)
class WordToken:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Sentence:
    text: str
    start: float
    end: float
    words: list[WordToken]


def extract_video_id(url: str) -> str | None:
    """Return the 11-character YouTube video ID, or None."""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"youtube\.com/watch\?.*[?&]v=([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _fetch_oembed(video_id: str) -> dict[str, Any]:
    """Fetch public oEmbed metadata for a video (title, author, thumbnail)."""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def _iso_639_1(language: str) -> str:
    """Normalize 'fr-fr' -> 'fr', 'en-us' -> 'en', etc."""
    return language.lower().split("-")[0]


def _split_text_into_sentence_pieces(text: str) -> list[tuple[str, int, bool]]:
    """Split text at sentence boundaries.

    Strong boundaries (. ! ? …) always end a sentence. Weak boundaries (, ;)
    are only used when a sentence would otherwise exceed the comfortable word
    limit, so short comma-separated phrases stay together.

    Returns a list of (piece_text, word_count, is_terminal).
    """
    text = text.replace("\n", " ").strip()
    if not text:
        return []

    result: list[tuple[str, int, bool]] = []
    for strong in _STRONG_END.split(text):
        strong = strong.strip()
        if not strong:
            continue
        word_count = len(_WORD_SPLIT.split(strong))
        if word_count <= _MAX_SENTENCE_WORDS:
            is_terminal = strong[-1] in ".!?"
            result.append((strong, word_count, is_terminal))
            continue

        # Long sentence: also split at weak boundaries so it stays readable.
        for weak in _WEAK_END.split(strong):
            weak = weak.strip()
            if not weak:
                continue
            is_terminal = weak[-1] in ".!?"
            word_count = len(_WORD_SPLIT.split(weak))
            result.append((weak, word_count, is_terminal))

    return result


def _chunk_piece(
    piece: str, word_count: int, is_terminal: bool, max_words: int
) -> list[tuple[str, int, bool]]:
    """Split a long piece into word-sized chunks, preserving terminal flag on last."""
    if word_count <= max_words:
        return [(piece, word_count, is_terminal)]
    words = _WORD_SPLIT.split(piece)
    chunks: list[tuple[str, int, bool]] = []
    for i in range(0, len(words), max_words):
        chunk_words = words[i : i + max_words]
        chunk_text = " ".join(chunk_words)
        chunks.append((chunk_text, len(chunk_words), False))
    if chunks:
        chunks[-1] = (chunks[-1][0], chunks[-1][1], is_terminal)
    return chunks


def _segment_sentences(raw_entries: list[dict[str, Any]], use_punctuation_model: bool = True) -> list[Sentence]:
    """Group transcript entries into sentences and assign approximate word times.

    If the punctuation restoration model is available and the transcript is
    sparse on terminal punctuation, restore punctuation first so we can split
    at real sentence boundaries instead of relying on pauses or word counts.
    Otherwise fall back to the rule-based segmenter.
    """
    if (
        use_punctuation_model
        and _has_punct_model()
        and punctuation_is_sparse(raw_entries)
    ):
        sentences = _segment_with_restored_punctuation(raw_entries)
        if sentences:
            return sentences
    return _segment_sentences_rule_based(raw_entries)


def _segment_with_restored_punctuation(raw_entries: list[dict[str, Any]]) -> list[Sentence]:
    """Restore punctuation globally, then split into sentences with timings."""
    raw_words: list[WordToken] = []
    for entry in raw_entries:
        entry_start = float(entry["start"])
        entry_duration = float(entry.get("duration", 0))
        entry_end = entry_start + entry_duration
        text = str(entry.get("text", "")).replace("\n", " ").strip()
        if not text:
            continue
        words = _WORD_SPLIT.split(text)
        per_word_duration = entry_duration / max(len(words), 1)
        for i, w in enumerate(words):
            w_clean = w.strip(".,!?;:\"'()[]{}«»")
            if not w_clean:
                continue
            w_start = entry_start + i * per_word_duration
            w_end = min(w_start + per_word_duration, entry_end)
            if w_end <= w_start:
                w_end = w_start + 0.01
            raw_words.append(WordToken(w_clean, w_start, w_end))

    if not raw_words:
        return []

    full_text = " ".join(w.text for w in raw_words)
    punctuated = restore_punctuation(full_text)

    sentences: list[Sentence] = []
    word_cursor = 0
    for sent_text, sent_word_count, _ in _split_text_into_sentence_pieces(punctuated):
        if not sent_text:
            continue
        sent_words: list[WordToken] = []
        for _ in range(sent_word_count):
            if word_cursor < len(raw_words):
                sent_words.append(raw_words[word_cursor])
                word_cursor += 1
        if sent_words:
            sentences.append(
                Sentence(
                    text=sent_text,
                    start=sent_words[0].start,
                    end=sent_words[-1].end,
                    words=sent_words,
                )
            )

    # Append any leftover words (e.g. if the model produced fewer sentences).
    if word_cursor < len(raw_words):
        leftover = raw_words[word_cursor:]
        sentences.append(
            Sentence(
                text=" ".join(w.text for w in leftover),
                start=leftover[0].start,
                end=leftover[-1].end,
                words=leftover,
            )
        )

    return sentences


def _segment_sentences_rule_based(raw_entries: list[dict[str, Any]]) -> list[Sentence]:
    """Group transcript entries into sentences and assign approximate word times.

    Sentences end at punctuation (. ! ? …), after a long pause between entries,
    or when the buffer reaches a maximum word count. This handles transcripts
    that lack punctuation or contain very long phrases.
    """
    sentences: list[Sentence] = []
    buffer_texts: list[str] = []
    buffer_words: list[WordToken] = []
    buffer_start: float | None = None
    buffer_end: float | None = None
    prev_end = 0.0

    def flush() -> None:
        nonlocal buffer_texts, buffer_words, buffer_start, buffer_end
        if not buffer_texts:
            return
        sentences.append(
            Sentence(
                text=" ".join(buffer_texts),
                start=buffer_start or 0,
                end=buffer_end or 0,
                words=buffer_words,
            )
        )
        buffer_texts = []
        buffer_words = []
        buffer_start = None
        buffer_end = None

    for entry in raw_entries:
        entry_start = float(entry["start"])
        entry_duration = float(entry.get("duration", 0))
        entry_end = entry_start + entry_duration
        text = str(entry.get("text", "")).replace("\n", " ").strip()
        if not text:
            continue

        # Start a new sentence after a long silence.
        if buffer_texts and entry_start - prev_end > _PAUSE_THRESHOLD_S:
            flush()

        pieces = _split_text_into_sentence_pieces(text)
        if not pieces:
            continue

        total_entry_words = sum(wc for _, wc, _ in pieces) or 1
        elapsed = 0.0
        for piece, word_count, is_terminal in pieces:
            piece_duration = entry_duration * (word_count / total_entry_words)
            piece_start = entry_start + elapsed
            piece_end = piece_start + piece_duration
            elapsed += piece_duration
            per_word_duration = piece_duration / max(word_count, 1)
            word_offset = 0

            for chunk_text, chunk_count, chunk_terminal in _chunk_piece(
                piece, word_count, is_terminal, _MAX_SENTENCE_WORDS
            ):
                chunk_start = piece_start + word_offset * per_word_duration
                chunk_end = piece_start + (word_offset + chunk_count) * per_word_duration
                if buffer_start is None:
                    buffer_start = chunk_start
                buffer_end = chunk_end
                buffer_texts.append(chunk_text)

                raw_words = _WORD_SPLIT.split(chunk_text)
                for i, w in enumerate(raw_words):
                    w_clean = w.strip(".,!?;:\"'()[]{}«»")
                    if not w_clean:
                        continue
                    w_start = chunk_start + i * per_word_duration
                    w_end = min(w_start + per_word_duration, piece_end)
                    if w_end <= w_start:
                        w_end = w_start + 0.01
                    buffer_words.append(WordToken(w_clean, w_start, w_end))

                word_offset += chunk_count

                # Flush at terminal punctuation, max word count, or long chunk.
                if chunk_terminal or len(buffer_words) >= _MAX_SENTENCE_WORDS:
                    flush()

        prev_end = entry_end

    flush()
    return sentences


def _list_available_languages(video_id: str) -> list[dict[str, str]]:
    """Return available transcript languages for a video."""
    try:
        transcripts = YouTubeTranscriptApi().list(video_id)
    except Exception:
        return []
    result = []
    for t in transcripts:
        result.append({"code": t.language_code, "name": t.language, "generated": t.is_generated})
    return result


def fetch_video_info(url: str, preferred_language: str = "fr-fr") -> dict[str, Any]:
    """Return metadata and available transcript languages for a YouTube URL."""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("invalid YouTube URL")

    meta = _fetch_oembed(video_id)
    languages = _list_available_languages(video_id)
    target_code = _iso_639_1(preferred_language)
    has_target = any(_iso_639_1(lang["code"]) == target_code for lang in languages)

    return {
        "video_id": video_id,
        "title": meta.get("title", ""),
        "author": meta.get("author_name", ""),
        "thumbnail": meta.get("thumbnail_url", ""),
        "available_languages": languages,
        "preferred_language": preferred_language,
        "has_preferred_language": has_target,
    }


def fetch_transcript(video_id: str, language: str = "fr-fr") -> dict[str, Any]:
    """Fetch and segment a transcript into sentences with word timings.

    Prefer the local Whisper model for real audio-aligned word timestamps; fall
    back to YouTube's caption entries if Whisper is unavailable or fails.

    Raises:
        TranscriptError: if no captions are available and Whisper fails.
    """
    target_code = _iso_639_1(language)

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=[target_code])
    except TranscriptsDisabled as exc:
        raise TranscriptError("transcripts are disabled for this video") from exc
    except NoTranscriptFound as exc:
        raise TranscriptError(f"no {language} transcript found") from exc
    except Exception as exc:
        raise TranscriptError(f"failed to load transcript: {exc}") from exc

    raw = [{"text": s.text, "start": s.start, "duration": s.duration} for s in transcript]

    # Base segmentation from YouTube captions (with punctuation restoration if available).
    base_sentences = _segment_sentences(raw)

    # Try Whisper alignment: real audio word timings mapped onto caption text.
    try:
        from src.whisper_transcribe import align_sentences, available as whisper_available

        if whisper_available():
            sentences = align_sentences(video_id, base_sentences, language)
            return {
                "video_id": video_id,
                "language": language,
                "sentence_count": len(sentences),
                "sentences": [
                    {
                        "text": s.text,
                        "start": round(s.start, 2),
                        "end": round(s.end, 2),
                        "words": [
                            {"text": w.text, "start": round(w.start, 2), "end": round(w.end, 2)}
                            for w in s.words
                        ],
                    }
                    for s in sentences
                ],
            }
    except Exception as exc:
        logger.warning("Whisper transcript failed, falling back to YouTube captions: %s", exc)

    # Fallback: YouTube captions with evenly-divided word durations.
    sentences = base_sentences

    return {
        "video_id": video_id,
        "language": language,
        "sentence_count": len(sentences),
        "sentences": [
            {
                "text": s.text,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "words": [
                    {"text": w.text, "start": round(w.start, 2), "end": round(w.end, 2)}
                    for w in s.words
                ],
            }
            for s in sentences
        ],
    }

"""YouTube transcript extraction and sentence segmentation for shadow-reading.

Tries to fetch human or auto-generated captions via youtube-transcript-api.  If
no captions are available, falls back to language detection on the video title
so the UI can ask the user for a transcript or run STT later.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD_SPLIT = re.compile(r"\s+")


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

    Returns a list of (piece_text, word_count, is_terminal).
    """
    pieces = _SENTENCE_END.split(text.replace("\n", " ").strip())
    result: list[tuple[str, int, bool]] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        is_terminal = piece[-1] in ".!?"
        word_count = len(_WORD_SPLIT.split(piece))
        result.append((piece, word_count, is_terminal))
    return result


def _segment_sentences(raw_entries: list[dict[str, Any]]) -> list[Sentence]:
    """Group transcript entries into sentences and assign approximate word times."""
    sentences: list[Sentence] = []
    buffer_texts: list[str] = []
    buffer_words: list[WordToken] = []
    buffer_start: float | None = None
    buffer_end: float | None = None

    for entry in raw_entries:
        entry_start = float(entry["start"])
        entry_duration = float(entry.get("duration", 0))
        entry_end = entry_start + entry_duration
        text = str(entry.get("text", "")).replace("\n", " ").strip()
        if not text:
            continue

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

            if buffer_start is None:
                buffer_start = piece_start
            buffer_end = piece_end
            buffer_texts.append(piece)

            # Distribute word timings within this piece linearly.
            raw_words = _WORD_SPLIT.split(piece)
            word_duration = piece_duration / max(len(raw_words), 1)
            for i, w in enumerate(raw_words):
                w_clean = w.strip(".,!?;:\"'()[]{}«»")
                if not w_clean:
                    continue
                w_start = piece_start + i * word_duration
                w_end = min(w_start + word_duration, piece_end)
                buffer_words.append(WordToken(w_clean, w_start, w_end))

            if is_terminal:
                sentences.append(
                    Sentence(
                        text=" ".join(buffer_texts),
                        start=buffer_start,
                        end=buffer_end,
                        words=buffer_words,
                    )
                )
                buffer_texts = []
                buffer_words = []
                buffer_start = None
                buffer_end = None

    if buffer_texts:
        sentences.append(
            Sentence(
                text=" ".join(buffer_texts),
                start=buffer_start or 0,
                end=buffer_end or 0,
                words=buffer_words,
            )
        )

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
    """Fetch and segment a YouTube transcript into sentences with word timings.

    Raises:
        TranscriptError: if no captions are available.
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
    sentences = _segment_sentences(raw)

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

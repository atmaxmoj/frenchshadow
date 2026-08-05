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
_MAX_SENTENCE_WORDS = 20
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

    def _ends_with_weak_boundary(texts: list[str]) -> bool:
        """True if the pending sentence ends with a comma/semicolon/colon."""
        if not texts:
            return False
        return texts[-1].rstrip()[-1] in ",;:"

    for entry in raw_entries:
        entry_start = float(entry["start"])
        entry_duration = float(entry.get("duration", 0))
        entry_end = entry_start + entry_duration
        text = str(entry.get("text", "")).replace("\n", " ").strip()
        if not text:
            continue

        # A long silence starts a new sentence only if we are not in the middle
        # of a comma-separated phrase. In slow-spoken videos each comma phrase
        # often arrives as its own caption entry with a breath-pause between.
        if (
            buffer_texts
            and entry_start - prev_end > _PAUSE_THRESHOLD_S
            and not _ends_with_weak_boundary(buffer_texts)
        ):
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
                # Hard cap: flush before the incoming chunk would overflow, so
                # no sentence exceeds the comfortable word limit.
                if (
                    buffer_texts
                    and len(buffer_words) + chunk_count > _MAX_SENTENCE_WORDS
                ):
                    flush()

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

                # Flush at terminal punctuation or when the hard cap is reached.
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


def _sentences_to_response(
    video_id: str, language: str, sentences: list[Sentence]
) -> dict[str, Any]:
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


def _fetch_manual_caption(video_id: str, language: str) -> list[dict[str, Any]] | None:
    """Return manually-authored captions for *language* if they exist.

    Auto-generated captions are treated as absent and return None so that we
    fall back to Whisper, which has more accurate timings for shadow-reading.
    """
    target_code = _iso_639_1(language)
    try:
        transcripts = YouTubeTranscriptApi().list(video_id)
    except Exception as exc:
        logger.debug("could not list transcripts for %s: %s", video_id, exc)
        return None

    for t in transcripts:
        if _iso_639_1(t.language_code) == target_code and not t.is_generated:
            logger.info("Using manual caption: %s", t.language_code)
            return [{"text": s.text, "start": s.start, "duration": s.duration} for s in t.fetch()]

    logger.info("No manual caption found for %s, will use Whisper", language)
    return None


def fetch_transcript(video_id: str, language: str = "fr-fr") -> dict[str, Any]:
    """Fetch and segment a transcript into sentences with word timings.

    Priority:
        1. Manually-authored YouTube captions in the requested language.
           These are aligned with Whisper word timings for accurate playback.
        2. Whisper-only transcription (auto-generated captions are ignored).

    Raises:
        TranscriptError: if no manual captions are available and Whisper fails
                         or is not present.
    """
    # 1. Prefer manual captions and align them to real audio timings.
    manual_raw = _fetch_manual_caption(video_id, language)
    if manual_raw is not None:
        base_sentences = _segment_sentences(manual_raw)
        try:
            from src.whisper_transcribe import align_sentences, available as whisper_available

            if whisper_available():
                sentences = align_sentences(video_id, base_sentences, language)
                return _sentences_to_response(video_id, language, sentences)
        except Exception as exc:
            logger.warning("Whisper alignment failed, using caption timings: %s", exc)
        return _sentences_to_response(video_id, language, base_sentences)

    # 2. No manual captions: fall back to Whisper-only transcription.
    try:
        from src.whisper_transcribe import available as whisper_available, segment_whisper_words

        if not whisper_available():
            raise TranscriptError("no manual captions and no local Whisper model")

        from src.whisper_transcribe import _load_whisper_words

        sentences = segment_whisper_words(_load_whisper_words(video_id, language))
        return _sentences_to_response(video_id, language, sentences)
    except TranscriptError:
        raise
    except Exception as exc:
        raise TranscriptError(f"Whisper transcription failed: {exc}") from exc

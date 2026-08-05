"""Tests for Whisper-based transcript alignment."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.youtube import Sentence, WordToken
from src.whisper_transcribe import align_sentences, available, segment_whisper_words


@pytest.mark.skipif(not available(), reason="no local whisper model")
def test_align_sentences_maps_whisper_timings():
    """Word timings from Whisper should replace approximate caption timings."""
    sentences = [
        Sentence(
            text="bonjour les amis",
            start=0.0,
            end=3.0,
            words=[
                WordToken(text="bonjour", start=0.0, end=1.0),
                WordToken(text="les", start=1.0, end=2.0),
                WordToken(text="amis", start=2.0, end=3.0),
            ],
        ),
        Sentence(
            text="et bienvenue",
            start=3.0,
            end=5.0,
            words=[
                WordToken(text="et", start=3.0, end=4.0),
                WordToken(text="bienvenue", start=4.0, end=5.0),
            ],
        ),
    ]

    whisper_words = [
        {"text": "bonjour", "start": 0.5, "end": 1.1},
        {"text": "les", "start": 1.1, "end": 1.5},
        {"text": "amis", "start": 1.5, "end": 2.2},
        {"text": "et", "start": 2.8, "end": 3.0},
        {"text": "bienvenue", "start": 3.0, "end": 3.8},
    ]

    with patch("src.whisper_transcribe._load_whisper_words", return_value=whisper_words):
        aligned = align_sentences("dummy", sentences, "fr-fr")

    assert len(aligned) == 2
    assert aligned[0].text == "bonjour les amis"
    assert aligned[0].words[0].start == pytest.approx(0.5, rel=0.01)
    assert aligned[0].words[0].end == pytest.approx(1.1, rel=0.01)
    assert aligned[0].words[1].start == pytest.approx(1.1, rel=0.01)
    assert aligned[0].words[-1].end == pytest.approx(2.2, rel=0.01)
    assert aligned[1].words[0].start == pytest.approx(2.8, rel=0.01)
    assert aligned[1].words[-1].end == pytest.approx(3.8, rel=0.01)


@pytest.mark.skipif(not available(), reason="no local whisper model")
def test_align_sentences_is_monotonic():
    """Aligned words must never go backwards in time."""
    sentences = [
        Sentence(
            text="a b c",
            start=0.0,
            end=3.0,
            words=[
                WordToken(text="a", start=0.0, end=1.0),
                WordToken(text="b", start=1.0, end=2.0),
                WordToken(text="c", start=2.0, end=3.0),
            ],
        ),
    ]
    whisper_words = [
        {"text": "a", "start": 0.0, "end": 0.5},
        {"text": "c", "start": 0.5, "end": 1.0},
    ]

    with patch("src.whisper_transcribe._load_whisper_words", return_value=whisper_words):
        aligned = align_sentences("dummy", sentences, "fr-fr")

    words = aligned[0].words
    for i in range(len(words) - 1):
        assert words[i].end <= words[i + 1].start


def test_segment_whisper_words_merges_comma_phrases():
    """Whisper words should be grouped by terminal punctuation, not commas."""
    whisper_words = [
        {"text": "aujourd'hui,", "start": 0.0, "end": 0.8},
        {"text": "nous", "start": 2.0, "end": 2.3},
        {"text": "allons", "start": 2.3, "end": 2.6},
        {"text": "avoir", "start": 2.6, "end": 2.9},
        {"text": "une", "start": 2.9, "end": 3.1},
        {"text": "conversation", "start": 3.1, "end": 3.6},
        {"text": "tout", "start": 3.6, "end": 3.9},
        {"text": "à", "start": 3.9, "end": 4.0},
        {"text": "fait", "start": 4.0, "end": 4.2},
        {"text": "ordinaire,", "start": 4.2, "end": 4.8},
        {"text": "parler", "start": 5.0, "end": 5.4},
        {"text": "de", "start": 5.4, "end": 5.5},
        {"text": "la", "start": 5.5, "end": 5.6},
        {"text": "météo,", "start": 5.6, "end": 6.0},
        {"text": "mais", "start": 7.0, "end": 7.3},
        {"text": "en", "start": 7.3, "end": 7.4},
        {"text": "français", "start": 7.4, "end": 7.8},
        {"text": "lent.", "start": 7.8, "end": 8.3},
    ]
    sentences = segment_whisper_words(whisper_words)
    assert len(sentences) == 1
    assert sentences[0].text.endswith("lent.")
    assert all(len(s.words) <= 20 for s in sentences)

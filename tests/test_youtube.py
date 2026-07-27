"""Tests for YouTube transcript segmentation."""

from __future__ import annotations

import pytest

from src.youtube import (
    _segment_sentences,
    _split_text_into_sentence_pieces,
    extract_video_id,
)


def test_extract_video_id_watch_url():
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_short_url():
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid():
    assert extract_video_id("https://example.com") is None


def test_split_text_into_sentence_pieces_basic():
    pieces = _split_text_into_sentence_pieces("Hello world. How are you? I'm fine.")
    texts = [p[0] for p in pieces]
    assert texts == ["Hello world.", "How are you?", "I'm fine."]


def test_split_text_into_sentence_pieces_french():
    pieces = _split_text_into_sentence_pieces("Bonjour les amis. Bienvenue ! Comment ça va ?")
    texts = [p[0] for p in pieces]
    assert texts == ["Bonjour les amis.", "Bienvenue !", "Comment ça va ?"]


def test_split_text_into_sentence_pieces_no_punctuation():
    pieces = _split_text_into_sentence_pieces("bonjour les amis et bienvenue")
    assert len(pieces) == 1
    assert pieces[0][0] == "bonjour les amis et bienvenue"


def test_segment_sentences_splits_on_punctuation():
    raw = [
        {"text": "Hello world. How are you?", "start": 0.0, "duration": 3.0},
        {"text": "I'm fine.", "start": 3.5, "duration": 1.5},
    ]
    sentences = _segment_sentences(raw)
    assert len(sentences) == 3
    assert sentences[0].text == "Hello world."
    assert sentences[1].text == "How are you?"
    assert sentences[2].text == "I'm fine."


def test_segment_sentences_splits_on_long_pause():
    raw = [
        {"text": "bonjour les amis", "start": 0.0, "duration": 2.0},
        {"text": "et bienvenue", "start": 5.0, "duration": 2.0},
    ]
    sentences = _segment_sentences(raw)
    assert len(sentences) == 2
    assert sentences[0].text == "bonjour les amis"
    assert sentences[1].text == "et bienvenue"


def test_segment_sentences_splits_on_max_words():
    raw = [
        {"text": "one two three four five six seven eight nine ten eleven twelve", "start": 0.0, "duration": 6.0},
    ]
    sentences = _segment_sentences(raw)
    assert len(sentences) >= 2
    assert all(len(s.words) <= 10 for s in sentences)


def test_segment_sentences_no_punctuation_splits_on_pause_and_max_words():
    raw = [
        {"text": "bonjour les amis et bienvenue dans ce nouvel épisode aujourd'hui nous allons avoir", "start": 0.0, "duration": 6.0},
        {"text": "une conversation tout à fait ordinaire parler de la météo", "start": 7.5, "duration": 4.0},
    ]
    sentences = _segment_sentences(raw)
    assert len(sentences) >= 3
    assert all(len(s.text.split()) <= 10 for s in sentences)


def test_segment_sentences_word_times_are_monotonic():
    raw = [
        {"text": "Hello world. How are you?", "start": 0.0, "duration": 4.0},
    ]
    sentences = _segment_sentences(raw)
    for s in sentences:
        for i in range(1, len(s.words)):
            assert s.words[i].start >= s.words[i - 1].start
            assert s.words[i].end > s.words[i].start

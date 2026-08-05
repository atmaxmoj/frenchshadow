"""Tests for YouTube transcript segmentation."""

from __future__ import annotations

import pytest

from src.punct import has_model
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


def test_fetch_manual_caption_prefers_manual_over_generated():
    """Auto-generated captions must be ignored; manual captions are returned."""
    from unittest.mock import MagicMock, patch

    from src.youtube import _fetch_manual_caption

    manual_snippet = MagicMock()
    manual_snippet.text = "bonjour"
    manual_snippet.start = 0.0
    manual_snippet.duration = 1.0

    generated_snippet = MagicMock()
    generated_snippet.text = "salut"
    generated_snippet.start = 0.0
    generated_snippet.duration = 1.0

    manual_transcript = MagicMock()
    manual_transcript.language_code = "fr"
    manual_transcript.is_generated = False
    manual_transcript.fetch.return_value = [manual_snippet]

    generated_transcript = MagicMock()
    generated_transcript.language_code = "fr"
    generated_transcript.is_generated = True
    generated_transcript.fetch.return_value = [generated_snippet]

    api = MagicMock()
    api.list.return_value = [generated_transcript, manual_transcript]

    with patch("src.youtube.YouTubeTranscriptApi", return_value=api):
        result = _fetch_manual_caption("dummy", "fr-fr")

    assert result is not None
    assert result[0]["text"] == "bonjour"
    manual_transcript.fetch.assert_called_once()
    generated_transcript.fetch.assert_not_called()


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


def test_split_text_into_sentence_pieces_commas_stay_together_when_short():
    # Short comma-separated phrases should not be split at the comma.
    pieces = _split_text_into_sentence_pieces("Bonjour, comment ça va ?")
    texts = [p[0] for p in pieces]
    assert texts == ["Bonjour, comment ça va ?"]


def test_split_text_into_sentence_pieces_commas_split_when_long():
    # Long sentences with commas should be split at weak boundaries.
    text = (
        "Bonjour les amis, et bienvenue dans ce nouvel épisode, "
        "aujourd'hui nous allons avoir une conversation tout à fait ordinaire, "
        "parler de la météo, de nos projets pour le weekend, "
        "bref, faire un peu de small talk, mais en français lent."
    )
    pieces = _split_text_into_sentence_pieces(text)
    texts = [p[0] for p in pieces]
    assert len(texts) >= 2
    assert all(len(t.split()) <= 12 for t in texts)
    # The final piece should keep the terminal punctuation.
    assert texts[-1][-1] == "."


# Rule-based segmentation tests: keep the punctuation model out of the loop so
# they stay fast and deterministic.


def test_segment_sentences_splits_on_punctuation():
    raw = [
        {"text": "Hello world. How are you?", "start": 0.0, "duration": 3.0},
        {"text": "I'm fine.", "start": 3.5, "duration": 1.5},
    ]
    sentences = _segment_sentences(raw, use_punctuation_model=False)
    assert len(sentences) == 3
    assert sentences[0].text == "Hello world."
    assert sentences[1].text == "How are you?"
    assert sentences[2].text == "I'm fine."


def test_segment_sentences_splits_on_long_pause():
    raw = [
        {"text": "bonjour les amis", "start": 0.0, "duration": 2.0},
        {"text": "et bienvenue", "start": 5.0, "duration": 2.0},
    ]
    sentences = _segment_sentences(raw, use_punctuation_model=False)
    assert len(sentences) == 2
    assert sentences[0].text == "bonjour les amis"
    assert sentences[1].text == "et bienvenue"


def test_segment_sentences_splits_on_max_words():
    raw = [
        {"text": "one two three four five six seven eight nine ten eleven twelve thirteen", "start": 0.0, "duration": 6.0},
    ]
    sentences = _segment_sentences(raw, use_punctuation_model=False)
    assert len(sentences) >= 2
    assert all(len(s.words) <= 12 for s in sentences)


def test_segment_sentences_no_punctuation_splits_on_pause_and_max_words():
    raw = [
        {"text": "bonjour les amis et bienvenue dans ce nouvel épisode aujourd'hui nous allons avoir", "start": 0.0, "duration": 6.0},
        {"text": "une conversation tout à fait ordinaire parler de la météo", "start": 7.5, "duration": 4.0},
    ]
    sentences = _segment_sentences(raw, use_punctuation_model=False)
    assert len(sentences) >= 3
    assert all(len(s.text.split()) <= 12 for s in sentences)


def test_segment_sentences_commas_stay_together_when_short():
    # A short comma-separated phrase should remain one sentence.
    raw = [
        {"text": "Bonjour, comment ça va ?", "start": 0.0, "duration": 2.0},
    ]
    sentences = _segment_sentences(raw, use_punctuation_model=False)
    assert len(sentences) == 1
    assert sentences[0].text == "Bonjour, comment ça va ?"


def test_segment_sentences_merge_comma_phrases_across_long_pauses():
    # YouTube captions for slow-spoken videos often emit one comma phrase per
    # entry with >1s gaps. Those gaps are breaths, not sentence boundaries, so
    # phrases ending in ',' should merge until a terminal '.' is reached.
    raw = [
        {"text": "aujourd'hui,", "start": 0.0, "duration": 0.8},
        {"text": "nous allons avoir une conversation tout à fait ordinaire,", "start": 2.0, "duration": 2.0},
        {"text": "parler de la météo,", "start": 5.0, "duration": 1.2},
        {"text": "mais en français lent.", "start": 7.0, "duration": 1.5},
        {"text": "et avant de commencer,", "start": 9.5, "duration": 1.0},
        {"text": "on vous rappelle que le 1er avril,", "start": 11.5, "duration": 1.5},
        {"text": "notre club de lecture commence sur discord.", "start": 14.0, "duration": 2.0},
    ]
    sentences = _segment_sentences(raw, use_punctuation_model=False)
    texts = [s.text for s in sentences]
    assert len(sentences) == 4, f"expected 4 sentences, got {len(sentences)}: {texts}"
    assert texts[0].endswith("ordinaire,")
    assert texts[1].endswith("lent.")
    assert texts[2].endswith("avril,")
    assert texts[3].endswith("discord.")
    assert all(len(s.text.split()) <= 12 for s in sentences)


def test_segment_sentences_word_times_are_monotonic():
    raw = [
        {"text": "Hello world. How are you?", "start": 0.0, "duration": 4.0},
    ]
    sentences = _segment_sentences(raw, use_punctuation_model=False)
    for s in sentences:
        for i in range(1, len(s.words)):
            assert s.words[i].start >= s.words[i - 1].start
            assert s.words[i].end > s.words[i].start


# Punctuation-model integration tests (slow because they load the 2 GB model).


@pytest.mark.skipif(not has_model(), reason="punctuation model not downloaded")
@pytest.mark.slow
def test_segment_sentences_model_restores_punctuation():
    raw = [
        {"text": "bonjour les amis et bienvenue", "start": 0.0, "duration": 3.0},
        {"text": "aujourd'hui nous allons parler de la météo", "start": 4.0, "duration": 4.0},
    ]
    sentences = _segment_sentences(raw, use_punctuation_model=True)
    # The model should introduce at least one sentence boundary.
    assert len(sentences) >= 2
    # All returned sentences should have words and monotonic timings.
    for s in sentences:
        assert s.words
        assert s.words[0].start <= s.words[-1].end
        for i in range(1, len(s.words)):
            assert s.words[i].start >= s.words[i - 1].start


@pytest.mark.skipif(not has_model(), reason="punctuation model not downloaded")
def test_punctuation_model_is_sparse_check():
    from src.punct import punctuation_is_sparse

    punctuated = [
        {"text": "Hello world.", "start": 0.0, "duration": 1.0},
        {"text": "How are you?", "start": 2.0, "duration": 1.0},
    ]
    assert punctuation_is_sparse(punctuated) is False

    unpunctuated = [
        {"text": "bonjour les amis", "start": 0.0, "duration": 1.0},
        {"text": "et bienvenue", "start": 2.0, "duration": 1.0},
    ]
    assert punctuation_is_sparse(unpunctuated) is True

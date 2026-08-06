"""Tests for punctuation restoration."""

from __future__ import annotations

import pytest

from src import punct


def test_has_model_reports_presence():
    # The model directory either exists in this checkout or it doesn't.
    assert punct.has_model() == punct.MODEL_DIR.is_dir()


@pytest.mark.skipif(not punct.has_model(), reason="punctuation model not downloaded")
@pytest.mark.slow
def test_restore_punctuation_preserves_words():
    text = "bonjour les amis et bienvenue"
    restored = punct.restore_punctuation(text)
    # Words should be preserved; only punctuation may be inserted.
    restored_words = restored.split()
    # Strip trailing punctuation from each word for comparison.
    cleaned = [w.rstrip(".,!?;:-") for w in restored_words]
    assert cleaned == text.split()


@pytest.mark.skipif(not punct.has_model(), reason="punctuation model not downloaded")
@pytest.mark.slow
def test_restore_punctuation_adds_terminal_marks():
    text = "bonjour les amis et bienvenue aujourd'hui nous allons parler de la météo"
    restored = punct.restore_punctuation(text)
    # A useful restoration should introduce at least one sentence boundary.
    assert any(p in restored for p in (".", "?", "!"))


def test_punctuation_is_sparse_empty():
    assert punct.punctuation_is_sparse([]) is False


def test_punctuation_is_sparse_threshold():
    entries = [
        {"text": "hello world", "start": 0.0, "duration": 1.0},
        {"text": "how are you", "start": 2.0, "duration": 1.0},
        {"text": "i am fine", "start": 4.0, "duration": 1.0},
    ]
    assert punct.punctuation_is_sparse(entries) is True

    punctuated = [
        {"text": "Hello world.", "start": 0.0, "duration": 1.0},
        {"text": "How are you?", "start": 2.0, "duration": 1.0},
        {"text": "I am fine.", "start": 4.0, "duration": 1.0},
    ]
    assert punct.punctuation_is_sparse(punctuated) is False

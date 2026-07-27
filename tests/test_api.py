"""Tests for FastAPI helpers (no model loading)."""

from __future__ import annotations

from api.main import _attach_word_times


def test_attach_word_times_basic():
    analysis = {
        "words": [
            {"word": "hello", "learner_start": 0, "learner_end": 3},
            {"word": "world", "learner_start": 3, "learner_end": 6},
        ]
    }
    token_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    _attach_word_times(analysis, duration_s=0.6, token_times=token_times)

    assert analysis["words"][0]["start_time"] == 0.0
    assert analysis["words"][0]["end_time"] == 0.3
    assert analysis["words"][1]["start_time"] == 0.3
    assert analysis["words"][1]["end_time"] <= 0.6


def test_attach_word_times_monotonic():
    analysis = {"words": [{"word": "a", "learner_start": 5, "learner_end": 5}]}
    _attach_word_times(analysis, duration_s=1.0, token_times=[0.0, 0.2, 0.4])
    assert analysis["words"][0]["start_time"] <= analysis["words"][0]["end_time"]

"""Tests for persistent storage (videos + attempts + dashboard stats)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src import storage


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Use a temporary DB and recordings directory for each test."""
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "attempts.db")
    monkeypatch.setattr(storage, "RECORDINGS_DIR", tmp_path / "recordings")
    storage.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    # Reset the singleton connection so _init_db runs against the new path.
    monkeypatch.setattr(storage, "_CONN", None)
    yield tmp_path
    # Close the connection if one was opened.
    if storage._CONN is not None:
        storage._CONN.close()
    monkeypatch.setattr(storage, "_CONN", None)


def test_touch_video_creates_record(fresh_db):
    storage.touch_video(
        video_id="abc123",
        title="Test Video",
        thumbnail="https://example.com/thumb.jpg",
        language="fr-fr",
        total_sentences=20,
        last_sentence_idx=0,
    )
    progress = storage.get_video_progress("abc123")
    assert progress is not None
    assert progress.video_id == "abc123"
    assert progress.title == "Test Video"
    assert progress.thumbnail == "https://example.com/thumb.jpg"
    assert progress.language == "fr-fr"
    assert progress.total_sentences == 20
    assert progress.last_sentence_idx == 0


def test_touch_video_updates_existing(fresh_db):
    storage.touch_video("v1", "Old", "http://old", "fr-fr", 10, 0)
    storage.touch_video("v1", "New", "http://new", "en-us", 12, 5)
    progress = storage.get_video_progress("v1")
    assert progress.title == "New"
    assert progress.language == "en-us"
    assert progress.total_sentences == 12
    assert progress.last_sentence_idx == 5


def test_save_attempt_persists_audio_and_analysis(fresh_db):
    storage.touch_video("v1", "Title", "http://thumb", "fr-fr", 10, 0)
    attempt = storage.save_attempt(
        video_id="v1",
        sentence_idx=2,
        sentence_text="bonjour le monde",
        language="fr-fr",
        audio_bytes=b"fake-webm-audio",
        analysis={"overall_score": 0.85, "words": []},
        duration_s=3.5,
    )
    assert attempt.video_id == "v1"
    assert attempt.sentence_idx == 2
    assert attempt.overall_score == 0.85
    assert attempt.duration_s == 3.5
    assert Path(attempt.recording_path).exists()

    attempts = storage.get_attempts("v1")
    assert len(attempts) == 1
    assert attempts[0].id == attempt.id


def test_get_attempts_filters_by_sentence(fresh_db):
    storage.touch_video("v1", "Title", "http://thumb", "fr-fr", 10, 0)
    storage.save_attempt("v1", 0, "a", "fr-fr", b"x", {"overall_score": 1.0})
    storage.save_attempt("v1", 1, "b", "fr-fr", b"x", {"overall_score": 0.9})
    storage.save_attempt("v1", 1, "c", "fr-fr", b"x", {"overall_score": 0.8})

    all_attempts = storage.get_attempts("v1")
    assert len(all_attempts) == 3

    sentence_1 = storage.get_attempts("v1", sentence_idx=1)
    assert len(sentence_1) == 2


def test_get_video_progress_counts_attempts(fresh_db):
    storage.touch_video("v1", "Title", "http://thumb", "fr-fr", 10, 0)
    storage.save_attempt("v1", 0, "a", "fr-fr", b"x", {"overall_score": 1.0})
    storage.save_attempt("v1", 1, "b", "fr-fr", b"x", {"overall_score": 0.9})
    storage.save_attempt("v1", 1, "c", "fr-fr", b"x", {"overall_score": 0.8})

    progress = storage.get_video_progress("v1")
    assert progress.attempt_count == 3
    assert progress.sentence_attempt_count == 2


def test_get_recent_videos_orders_by_practice_time(fresh_db):
    storage.touch_video("v1", "First", "http://a", "fr-fr", 10, 0)
    storage.touch_video("v2", "Second", "http://b", "fr-fr", 10, 0)
    storage.save_attempt("v1", 0, "a", "fr-fr", b"x", {"overall_score": 1.0})
    storage.save_attempt("v2", 0, "b", "fr-fr", b"x", {"overall_score": 1.0})

    recent = storage.get_recent_videos(limit=10)
    assert len(recent) == 2
    # v2 was touched/saved last, so it should appear first.
    assert recent[0].video_id == "v2"
    assert recent[1].video_id == "v1"


def test_get_stats_aggregates(fresh_db):
    storage.touch_video("v1", "Title", "http://thumb", "fr-fr", 10, 0)
    storage.save_attempt("v1", 0, "a", "fr-fr", b"x", {"overall_score": 1.0}, duration_s=2.0)
    storage.save_attempt("v1", 1, "b", "fr-fr", b"x", {"overall_score": 0.9}, duration_s=3.0)

    stats = storage.get_stats()
    assert stats["videos"] == 1
    assert stats["attempts"] == 2
    assert stats["sentences"] == 2
    assert abs(stats["total_minutes"] - 5.0 / 60) < 0.02
    assert stats["days"] == 1


def test_get_recording_path(fresh_db):
    storage.touch_video("v1", "Title", "http://thumb", "fr-fr", 10, 0)
    attempt = storage.save_attempt("v1", 0, "a", "fr-fr", b"x", {"overall_score": 1.0})
    path = storage.get_recording_path(attempt.id)
    assert path is not None
    assert path.read_bytes() == b"x"

    assert storage.get_recording_path("nonexistent") is None


def test_save_attempt_updates_last_sentence_idx(fresh_db):
    storage.touch_video("v1", "Title", "http://thumb", "fr-fr", 10, 0)
    storage.save_attempt("v1", 3, "a", "fr-fr", b"x", {"overall_score": 1.0})
    progress = storage.get_video_progress("v1")
    assert progress.last_sentence_idx == 3

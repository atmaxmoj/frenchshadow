"""Tests for the dashboard HTTP endpoints (no model loading)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import storage


def _fake_load_model():
    return None, None, "cpu"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a TestClient with a fresh storage backend and a stubbed model."""
    monkeypatch.setattr("src.models.load_model", _fake_load_model)
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "attempts.db")
    monkeypatch.setattr(storage, "RECORDINGS_DIR", tmp_path / "recordings")
    storage.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "_CONN", None)

    from api.main import app

    with TestClient(app) as c:
        yield c

    if storage._CONN is not None:
        storage._CONN.close()
    monkeypatch.setattr(storage, "_CONN", None)


def test_stats_empty(client):
    res = client.get("/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["videos"] == 0
    assert data["attempts"] == 0
    assert data["total_minutes"] == 0.0


def test_recent_videos_empty(client):
    res = client.get("/recent_videos")
    assert res.status_code == 200
    assert res.json()["videos"] == []


def test_post_attempt_persists_and_updates_dashboard(client):
    audio = io.BytesIO(b"fake-audio")
    res = client.post(
        "/attempts",
        data={
            "video_id": "vid1",
            "sentence_idx": 0,
            "sentence_text": "hello world",
            "language": "en-us",
            "analysis": '{"overall_score": 0.9, "words": []}',
            "duration_s": "120",
            "title": "Hello Video",
            "thumbnail": "http://thumb",
            "total_sentences": "10",
        },
        files={"audio": ("recording.webm", audio, "audio/webm")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["video_id"] == "vid1"
    assert body["overall_score"] == 0.9

    stats = client.get("/stats").json()
    assert stats["videos"] == 1
    assert stats["attempts"] == 1
    assert stats["total_minutes"] > 0

    recent = client.get("/recent_videos").json()
    assert len(recent["videos"]) == 1
    video = recent["videos"][0]
    assert video["video_id"] == "vid1"
    assert video["title"] == "Hello Video"
    assert video["thumbnail"] == "http://thumb"
    assert video["total_sentences"] == 10
    assert video["last_sentence_idx"] == 0
    assert video["attempt_count"] == 1


def test_video_progress_endpoint(client):
    client.post(
        "/attempts",
        data={
            "video_id": "vid2",
            "sentence_idx": 3,
            "sentence_text": "test",
            "language": "fr-fr",
            "analysis": '{"overall_score": 0.7}',
            "duration_s": "1.0",
            "title": "French Video",
            "thumbnail": "http://french",
            "total_sentences": "20",
        },
        files={"audio": ("recording.webm", io.BytesIO(b"x"), "audio/webm")},
    )

    res = client.get("/videos/vid2/progress")
    assert res.status_code == 200
    data = res.json()
    assert data["video_id"] == "vid2"
    assert data["last_sentence_idx"] == 3
    assert data["total_sentences"] == 20


def test_video_progress_404(client):
    res = client.get("/videos/unknown/progress")
    assert res.status_code == 404


def test_list_attempts(client):
    client.post(
        "/attempts",
        data={
            "video_id": "vid3",
            "sentence_idx": 0,
            "sentence_text": "one",
            "language": "en-us",
            "analysis": '{"overall_score": 0.8}',
            "duration_s": "1.0",
            "title": "",
            "thumbnail": "",
            "total_sentences": "5",
        },
        files={"audio": ("recording.webm", io.BytesIO(b"a"), "audio/webm")},
    )
    client.post(
        "/attempts",
        data={
            "video_id": "vid3",
            "sentence_idx": 1,
            "sentence_text": "two",
            "language": "en-us",
            "analysis": '{"overall_score": 0.6}',
            "duration_s": "1.0",
            "title": "",
            "thumbnail": "",
            "total_sentences": "5",
        },
        files={"audio": ("recording.webm", io.BytesIO(b"b"), "audio/webm")},
    )

    res = client.get("/attempts?video_id=vid3")
    assert res.status_code == 200
    data = res.json()
    assert len(data["attempts"]) == 2

    res = client.get("/attempts?video_id=vid3&sentence_idx=1")
    assert len(res.json()["attempts"]) == 1
    assert res.json()["attempts"][0]["sentence_text"] == "two"


def test_attempt_audio_stream(client):
    res = client.post(
        "/attempts",
        data={
            "video_id": "vid4",
            "sentence_idx": 0,
            "sentence_text": "audio test",
            "language": "en-us",
            "analysis": '{"overall_score": 1.0}',
            "duration_s": "0.5",
            "title": "",
            "thumbnail": "",
            "total_sentences": "1",
        },
        files={"audio": ("recording.webm", io.BytesIO(b"webm-bytes"), "audio/webm")},
    )
    attempt_id = res.json()["id"]

    audio_res = client.get(f"/attempts/{attempt_id}/audio")
    assert audio_res.status_code == 200
    assert audio_res.content == b"webm-bytes"
    assert audio_res.headers["content-type"] == "audio/webm"

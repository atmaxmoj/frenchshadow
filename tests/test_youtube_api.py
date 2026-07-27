"""Tests for the YouTube info/transcript endpoints (with mocked fetchers)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _fake_load_model():
    return None, None, "cpu"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("src.models.load_model", _fake_load_model)

    from api.main import app

    with TestClient(app) as c:
        yield c


def test_youtube_info(client, monkeypatch):
    def fake_info(url, preferred_language):
        return {
            "video_id": "abc123",
            "title": "Test Video",
            "author": "Test Author",
            "thumbnail": "http://thumb",
            "available_languages": [{"code": "fr", "name": "French", "generated": False}],
            "preferred_language": preferred_language,
            "has_preferred_language": True,
        }

    monkeypatch.setattr("api.main.fetch_video_info", fake_info)

    res = client.get("/youtube/info?url=https://www.youtube.com/watch?v=abc123&language=fr-fr")
    assert res.status_code == 200
    data = res.json()
    assert data["video_id"] == "abc123"
    assert data["title"] == "Test Video"
    assert data["has_preferred_language"] is True


def test_youtube_info_empty_url(client):
    res = client.get("/youtube/info?url=&language=fr-fr")
    assert res.status_code == 400


def test_youtube_transcript(client, monkeypatch):
    def fake_transcript(video_id, language):
        return {
            "video_id": video_id,
            "language": language,
            "sentence_count": 2,
            "sentences": [
                {
                    "text": "Bonjour.",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [{"text": "Bonjour", "start": 0.0, "end": 1.0}],
                },
                {
                    "text": "Comment ça va ?",
                    "start": 1.5,
                    "end": 3.0,
                    "words": [
                        {"text": "Comment", "start": 1.5, "end": 2.0},
                        {"text": "ça", "start": 2.0, "end": 2.4},
                        {"text": "va", "start": 2.4, "end": 3.0},
                    ],
                },
            ],
        }

    monkeypatch.setattr("api.main.fetch_transcript", fake_transcript)

    res = client.get("/youtube/transcript?video_id=abc123&language=fr-fr")
    assert res.status_code == 200
    data = res.json()
    assert data["sentence_count"] == 2
    assert data["sentences"][0]["text"] == "Bonjour."
    assert len(data["sentences"][1]["words"]) == 3


def test_youtube_transcript_empty_video_id(client):
    res = client.get("/youtube/transcript?video_id=&language=fr-fr")
    assert res.status_code == 400


def test_youtube_transcript_not_found(client, monkeypatch):
    from src.youtube import TranscriptError

    def fake_transcript(_video_id, language=None):
        raise TranscriptError("no transcript")

    monkeypatch.setattr("api.main.fetch_transcript", fake_transcript)

    res = client.get("/youtube/transcript?video_id=missing&language=fr-fr")
    assert res.status_code == 404

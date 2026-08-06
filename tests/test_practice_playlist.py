"""Tests for /videos/{id}/practice_playlist (history whole-video replay data)."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import api.main as main
from src import storage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "attempts.db")
    monkeypatch.setattr(storage, "RECORDINGS_DIR", tmp_path / "recordings")
    storage.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "_CONN", None)
    monkeypatch.setattr(main, "COSY3_CACHE_DIR", tmp_path / "cosy3_cache")

    with TestClient(main.app) as c:
        yield c

    if storage._CONN is not None:
        storage._CONN.close()
    monkeypatch.setattr(storage, "_CONN", None)


def _create_attempt(client, sentence_idx: int, text: str, audio: bytes, video_id: str = "vid_pl") -> str:
    res = client.post(
        "/attempts",
        data={
            "video_id": video_id,
            "sentence_idx": sentence_idx,
            "sentence_text": text,
            "language": "fr-fr",
            "analysis": '{"overall_score": 0.7, "words": []}',
            "duration_s": "1.0",
            "title": "PL",
            "thumbnail": "http://thumb",
            "total_sentences": "3",
        },
        files={"audio": ("recording.webm", io.BytesIO(audio), "audio/webm")},
    )
    assert res.status_code == 200
    return res.json()["id"]


def _bake(client, **form):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = b"RIFF\x00\x00\x00\x00WAVEfake"
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value = mock_client
        res = client.post("/cosy_clone", data=form, files=form.pop("_files", {}) or None)
    assert res.status_code == 200
    return res


def test_playlist_latest_attempt_per_sentence_and_clone_lookup(client):
    a0_old = _create_attempt(client, 0, "Bonjour", b"audio-zero-v1")
    a0_new = _create_attempt(client, 0, "Bonjour", b"audio-zero-v2")
    a1 = _create_attempt(client, 1, "Comment ça va", b"audio-one")

    # Bake clones: sentence 0 via its OWN latest recording (blob path),
    # sentence 1 via sentence 0's OLD recording as the pinned first sample
    # (simulates the rolling pre-bake using another take as the voice sample).
    _bake(client, target_text="Bonjour", prompt_text="Bonjour", language="fr-fr",
          _files={"audio": ("r.webm", io.BytesIO(b"audio-zero-v2"), "audio/webm")})
    _bake(client, target_text="Comment ça va", prompt_text="Comment ça va", language="fr-fr",
          _files={"audio": ("r.webm", io.BytesIO(b"audio-zero-v1"), "audio/webm")})

    res = client.get("/videos/vid_pl/practice_playlist?language=fr-fr")
    assert res.status_code == 200
    items = res.json()["items"]
    assert [i["sentence_idx"] for i in items] == [0, 1]

    # Latest attempt wins for sentence 0.
    assert items[0]["attempt_id"] == a0_new
    assert items[0]["clone_key"] is not None
    # Clone baked with a DIFFERENT take as the sample is still found (pre-bake).
    assert items[1]["attempt_id"] == a1
    assert items[1]["clone_key"] is not None
    assert items[0]["clone_key"] != items[1]["clone_key"]
    assert a0_old != a0_new

    # The clone audio is actually servable.
    wav = client.get(f"/cosy_clone_audio/{items[0]['clone_key']}.wav")
    assert wav.status_code == 200


def test_playlist_clone_key_null_when_not_baked(client):
    _create_attempt(client, 0, "Bonjour", b"some-audio")
    res = client.get("/videos/vid_pl/practice_playlist?language=fr-fr")
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 1
    assert items[0]["clone_key"] is None


def test_playlist_empty_for_unknown_video(client):
    res = client.get("/videos/nope/practice_playlist?language=fr-fr")
    assert res.status_code == 200
    assert res.json()["items"] == []

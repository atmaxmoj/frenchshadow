"""Tests for the processed playback variant of /attempts/{id}/audio."""

from __future__ import annotations

import io
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from src import storage


@pytest.fixture
def client(tmp_path, monkeypatch):
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


def _tone_wav() -> bytes:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    res = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-f", "wav", "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return res.stdout


def _create_attempt(client, audio: bytes) -> str:
    res = client.post(
        "/attempts",
        data={
            "video_id": "vid_pb",
            "sentence_idx": 0,
            "sentence_text": "bonjour",
            "language": "fr-fr",
            "analysis": '{"overall_score": 0.8, "words": []}',
            "duration_s": "1.0",
            "title": "T",
            "thumbnail": "",
            "total_sentences": "1",
        },
        files={"audio": ("recording.webm", io.BytesIO(audio), "audio/webm")},
    )
    assert res.status_code == 200
    return res.json()["id"]


def test_playback_audio_is_denoised_normalized_wav_and_cached(client):
    attempt_id = _create_attempt(client, _tone_wav())

    # Raw is still the default.
    raw = client.get(f"/attempts/{attempt_id}/audio")
    assert raw.status_code == 200

    res = client.get(f"/attempts/{attempt_id}/audio?playback=1")
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.content[:4] == b"RIFF"

    # The processed copy is cached next to the raw recording.
    cached = storage.RECORDINGS_DIR / "processed" / f"{attempt_id}.wav"
    assert cached.exists()
    assert cached.read_bytes() == res.content

    # Second request serves the cache just fine.
    again = client.get(f"/attempts/{attempt_id}/audio?playback=1")
    assert again.status_code == 200
    assert again.content == res.content


def test_playback_audio_falls_back_to_raw_when_processing_fails(client):
    attempt_id = _create_attempt(client, b"not-real-audio")
    res = client.get(f"/attempts/{attempt_id}/audio?playback=1")
    assert res.status_code == 200  # raw bytes, no crash
    assert res.content == b"not-real-audio"

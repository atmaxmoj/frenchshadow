"""Tests for the /cosy_clone proxy endpoint (no real CosyVoice3 loading)."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src import storage


def _fake_load_model():
    return None, None, "cpu"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a TestClient with fresh storage and a stubbed model."""
    monkeypatch.setattr("src.models.load_model", _fake_load_model)
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "attempts.db")
    monkeypatch.setattr(storage, "RECORDINGS_DIR", tmp_path / "recordings")
    storage.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "_CONN", None)

    from api.main import app, COSY3_CACHE_DIR

    # Redirect cache to tmp_path so tests do not pollute the repo cache.
    monkeypatch.setattr("api.main.COSY3_CACHE_DIR", tmp_path / "cosy3_cache")

    with TestClient(app) as c:
        yield c

    if storage._CONN is not None:
        storage._CONN.close()
    monkeypatch.setattr(storage, "_CONN", None)


def _create_attempt(client, text: str = "bonjour les amis") -> str:
    """Persist a fake attempt and return its id."""
    res = client.post(
        "/attempts",
        data={
            "video_id": "vid_cosy",
            "sentence_idx": 0,
            "sentence_text": text,
            "language": "fr-fr",
            "analysis": '{"overall_score": 0.8, "words": []}',
            "duration_s": "1.0",
            "title": "Cosy Test",
            "thumbnail": "http://thumb",
            "total_sentences": "1",
        },
        files={"audio": ("recording.webm", io.BytesIO(b"fake-audio"), "audio/webm")},
    )
    assert res.status_code == 200
    return res.json()["id"]


def test_cosy_clone_forwards_to_service_and_caches_wav(client):
    """The backend should forward the request to the CosyVoice3 service, save the
    returned WAV, and expose it under /cosy_clone_audio/{id}.wav."""
    attempt_id = _create_attempt(client)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = b"RIFF\x00\x00\x00\x00WAVEfake-wav-data"

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        res = client.post(
            "/cosy_clone",
            data={
                "attempt_id": attempt_id,
                "target_text": "bonjour les amis",
                "prompt_text": "bonjour les amis",
            },
        )

    assert res.status_code == 200
    body = res.json()
    assert body["url"].startswith("/cosy_clone_audio/")
    assert body["url"].endswith(".wav")
    assert body["cached"] is True

    # Ensure the forwarded payload matches what CosyVoice3 expects.
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[1]["json"]["target_text"] == "bonjour les amis"
    assert call_args[1]["json"]["prompt_text"] == "bonjour les amis"
    assert call_args[1]["json"]["ref_path"].endswith(".webm")

    # The cached audio endpoint should stream the mocked WAV bytes.
    audio_res = client.get(body["url"])
    assert audio_res.status_code == 200
    assert audio_res.content == mock_response.content
    assert audio_res.headers["content-type"] == "audio/wav"


def test_cosy_clone_defaults_to_attempt_sentence_text(client):
    """If target_text/prompt_text are omitted, the backend uses the sentence
    stored on the attempt."""
    attempt_id = _create_attempt(client)

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = b"RIFF\x00\x00\x00\x00WAVEfake-wav-data"

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        res = client.post(
            "/cosy_clone",
            data={"attempt_id": attempt_id},
        )

    assert res.status_code == 200
    call_args = mock_client.post.call_args
    assert call_args[1]["json"]["target_text"] == "bonjour les amis"
    assert call_args[1]["json"]["prompt_text"] == "bonjour les amis"


def test_cosy_clone_accepts_raw_audio_blob(client):
    """The /cosy_clone endpoint can receive the raw recording blob directly so
    the frontend can start cloning in parallel with analysis."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = b"RIFF\x00\x00\x00\x00WAVEfake-wav-data"

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        res = client.post(
            "/cosy_clone",
            data={"target_text": "bonjour les amis", "prompt_text": "bonjour les amis"},
            files={"audio": ("recording.webm", io.BytesIO(b"fake-audio"), "audio/webm")},
        )

    assert res.status_code == 200
    body = res.json()
    assert body["url"].startswith("/cosy_clone_audio/")
    assert body["url"].endswith(".wav")

    call_args = mock_client.post.call_args
    assert call_args[1]["json"]["target_text"] == "bonjour les amis"
    assert call_args[1]["json"]["prompt_text"] == "bonjour les amis"
    assert call_args[1]["json"]["ref_path"].endswith(".webm")


def test_cosy_clone_returns_404_for_missing_attempt(client):
    res = client.post(
        "/cosy_clone",
        data={"attempt_id": "no-such-id", "target_text": "hello", "prompt_text": "hello"},
    )
    assert res.status_code == 404


def test_cosy_clone_returns_503_when_service_down(client):
    """When the CosyVoice3 service is unreachable, the backend should surface a
    clear 503 instead of crashing."""
    attempt_id = _create_attempt(client)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        from httpx import ConnectError
        mock_client.post = AsyncMock(side_effect=ConnectError("Connection refused"))
        mock_client_cls.return_value = mock_client

        res = client.post(
            "/cosy_clone",
            data={"attempt_id": attempt_id, "target_text": "hello", "prompt_text": "hello"},
        )

    assert res.status_code == 503
    assert "not running" in res.json()["detail"]
    assert res.status_code == 503
    assert "not running" in res.json()["detail"]


def _mock_cosy_ok():
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = b"RIFF\x00\x00\x00\x00WAVEfake-wav-data"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


def test_cosy_clone_cache_key_includes_target_text(client):
    """The same attempt recording is reused for the sentence clone AND for
    per-word clones. Different target texts must produce different cache keys,
    otherwise a word clone would overwrite (or be shadowed by) the sentence."""
    attempt_id = _create_attempt(client)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = _mock_cosy_ok()
        mock_client_cls.return_value = mock_client

        sentence_res = client.post(
            "/cosy_clone",
            data={"attempt_id": attempt_id, "target_text": "bonjour les amis"},
        )
        word_res = client.post(
            "/cosy_clone",
            data={"attempt_id": attempt_id, "target_text": "bonjour"},
        )

    assert sentence_res.status_code == 200
    assert word_res.status_code == 200
    assert sentence_res.json()["url"] != word_res.json()["url"]
    assert mock_client.post.call_count == 2


def test_cosy_clone_serves_from_cache_without_calling_service(client):
    """A repeat request for the same (recording, target, prompt) must be served
    from disk — the CosyVoice3 service must NOT be called again."""
    attempt_id = _create_attempt(client)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = _mock_cosy_ok()
        mock_client_cls.return_value = mock_client
        first = client.post(
            "/cosy_clone",
            data={"attempt_id": attempt_id, "target_text": "bonjour"},
        )
    assert first.status_code == 200

    # Second call: service is DOWN, but the cached WAV must still be served.
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        from httpx import ConnectError
        mock_client.post = AsyncMock(side_effect=ConnectError("Connection refused"))
        mock_client_cls.return_value = mock_client

        second = client.post(
            "/cosy_clone",
            data={"attempt_id": attempt_id, "target_text": "bonjour"},
        )

    assert second.status_code == 200
    assert second.json()["url"] == first.json()["url"]
    assert second.json()["cached"] is True
    mock_client.post.assert_not_called()


def test_cosy_clone_word_level_forwards_word_and_sentence_prompt(client):
    """Word-level clone: the target is the single word, the prompt stays the
    full sentence the user actually read (it describes the reference audio)."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = _mock_cosy_ok()
        mock_client_cls.return_value = mock_client

        res = client.post(
            "/cosy_clone",
            data={
                "target_text": "épisode",
                "prompt_text": "Bonjour les amis et bienvenue dans ce nouvel épisode",
                "language": "fr-fr",
            },
            files={"audio": ("recording.webm", io.BytesIO(b"fake-audio"), "audio/webm")},
        )

    assert res.status_code == 200
    call_args = mock_client.post.call_args
    assert call_args[1]["json"]["target_text"] == "épisode"
    assert call_args[1]["json"]["prompt_text"] == (
        "Bonjour les amis et bienvenue dans ce nouvel épisode"
    )
    assert call_args[1]["json"]["language"] == "fr-fr"


def test_cosy_clone_cache_key_includes_language(client):
    """The same recording + text in a different language is a different audio:
    the cache keys must differ and the service must be called for each."""
    attempt_id = _create_attempt(client)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = _mock_cosy_ok()
        mock_client_cls.return_value = mock_client

        fr = client.post(
            "/cosy_clone",
            data={"attempt_id": attempt_id, "target_text": "bonjour", "language": "fr-fr"},
        )
        en = client.post(
            "/cosy_clone",
            data={"attempt_id": attempt_id, "target_text": "bonjour", "language": "en-us"},
        )

    assert fr.status_code == 200
    assert en.status_code == 200
    assert fr.json()["url"] != en.json()["url"]
    assert mock_client.post.call_count == 2
    langs = [c[1]["json"]["language"] for c in mock_client.post.call_args_list]
    assert langs == ["fr-fr", "en-us"]


def test_cosy_clone_same_blob_twice_reuses_recording_and_cache(client):
    """Uploading the same recording blob for the same word twice must hit the
    cache on the second call (content-addressed recording + cache key)."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = _mock_cosy_ok()
        mock_client_cls.return_value = mock_client

        first = client.post(
            "/cosy_clone",
            data={"target_text": "bonjour", "prompt_text": "bonjour les amis"},
            files={"audio": ("recording.webm", io.BytesIO(b"same-audio"), "audio/webm")},
        )
        second = client.post(
            "/cosy_clone",
            data={"target_text": "bonjour", "prompt_text": "bonjour les amis"},
            files={"audio": ("recording.webm", io.BytesIO(b"same-audio"), "audio/webm")},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["url"] == second.json()["url"]
    # The service was called once; the second request was a cache hit.
    assert mock_client.post.call_count == 1

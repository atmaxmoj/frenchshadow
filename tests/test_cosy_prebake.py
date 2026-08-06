"""Tests for /cosy_prebake: background clone baking from one voice sample."""

from __future__ import annotations

import io
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import api.main as main
from src import storage


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with fresh storage and a redirected clone cache."""
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "attempts.db")
    monkeypatch.setattr(storage, "RECORDINGS_DIR", tmp_path / "recordings")
    storage.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "_CONN", None)
    monkeypatch.setattr(main, "COSY3_CACHE_DIR", tmp_path / "cosy3_cache")
    monkeypatch.setattr(main, "_prebake_task", None)

    with TestClient(main.app) as c:
        yield c

    if storage._CONN is not None:
        storage._CONN.close()
    monkeypatch.setattr(storage, "_CONN", None)


def _mock_cosy_ok():
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = b"RIFF\x00\x00\x00\x00WAVEfake-wav-data"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


def _wait_task(timeout: float = 10.0) -> None:
    task = main._prebake_task
    assert task is not None
    deadline = time.time() + timeout
    while not task.done() and time.time() < deadline:
        time.sleep(0.05)
    assert task.done(), "prebake task did not finish"


def _post_prebake(client, items, language="fr-fr", audio=b"voice-sample"):
    return client.post(
        "/cosy_prebake",
        data={"items": json.dumps(items), "language": language},
        files={"audio": ("recording.webm", io.BytesIO(audio), "audio/webm")},
    )


def test_prebake_bakes_each_item_and_caches(client):
    items = [
        {"target_text": "Bonjour les amis", "prompt_text": "Bonjour les amis"},
        {"target_text": "Comment ça va", "prompt_text": "Comment ça va"},
    ]
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = _mock_cosy_ok()
        mock_client_cls.return_value = mock_client

        res = _post_prebake(client, items)
        assert res.status_code == 200
        assert res.json()["queued"] == 2
        _wait_task()

    assert mock_client.post.call_count == 2
    texts = [c[1]["json"]["target_text"] for c in mock_client.post.call_args_list]
    assert texts == ["Bonjour les amis", "Comment ça va"]
    # Every baked item landed in the cache directory.
    cached = list(main.COSY3_CACHE_DIR.glob("*.wav"))
    assert len(cached) == 2


def test_prebake_skips_already_cached_items(client):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = _mock_cosy_ok()
        mock_client_cls.return_value = mock_client
        _post_prebake(client, [{"target_text": "Bonjour", "prompt_text": "Bonjour"}])
        _wait_task()
        assert mock_client.post.call_count == 1

        # Second prebake of the same sample+text: fully served from cache.
        _post_prebake(client, [{"target_text": "Bonjour", "prompt_text": "Bonjour"}])
        _wait_task()
        assert mock_client.post.call_count == 1


def test_prebake_continues_past_failures(client):
    """A failing item must not block the rest of the queue."""
    ok_response = AsyncMock()
    ok_response.status_code = 200
    ok_response.content = b"RIFF\x00\x00\x00\x00WAVEfake"
    bad_response = AsyncMock()
    bad_response.status_code = 500
    bad_response.text = "boom"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=[bad_response, ok_response])

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value = mock_client
        _post_prebake(client, [
            {"target_text": "bad item", "prompt_text": "bad item"},
            {"target_text": "good item", "prompt_text": "good item"},
        ])
        _wait_task()

    assert mock_client.post.call_count == 2
    assert len(list(main.COSY3_CACHE_DIR.glob("*.wav"))) == 1  # only the good one


def test_prebake_replaces_previous_queue(client):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = _mock_cosy_ok()
        mock_client_cls.return_value = mock_client
        _post_prebake(client, [{"target_text": "first", "prompt_text": "first"}])
        first_task = main._prebake_task
        _post_prebake(client, [{"target_text": "second", "prompt_text": "second"}])
        second_task = main._prebake_task

        assert second_task is not first_task
        _wait_task()
        assert first_task.done()  # cancelled or finished; no longer running


def test_prebake_rejects_bad_input(client):
    res = client.post(
        "/cosy_prebake",
        data={"items": "not json", "language": "fr-fr"},
        files={"audio": ("recording.webm", io.BytesIO(b"x"), "audio/webm")},
    )
    assert res.status_code == 400

    res = client.post(
        "/cosy_prebake",
        data={"items": "[]", "language": "fr-fr"},
        files={"audio": ("recording.webm", io.BytesIO(b""), "audio/webm")},
    )
    assert res.status_code == 400


def test_prebake_queue_is_capped(client, monkeypatch):
    """SHADOW_READER_COSY_PREBAKE_MAX bounds how many items one queue holds."""
    monkeypatch.setattr(main, "COSY_PREBAKE_MAX", 2)
    items = [{"target_text": f"s{i}", "prompt_text": f"s{i}"} for i in range(5)]
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = _mock_cosy_ok()
        mock_client_cls.return_value = mock_client
        res = _post_prebake(client, items)
        assert res.status_code == 200
        assert res.json()["queued"] == 2
        _wait_task()
        assert mock_client.post.call_count == 2


def test_bake_evicts_oldest_when_cache_over_cap(client, monkeypatch):
    """SHADOW_READER_COSY_CACHE_MAX_MB bounds the on-disk clone cache."""
    clip = b"RIFF\x00\x00\x00\x00WAVEfake-wav-data"
    monkeypatch.setattr(main, "COSY_CACHE_MAX_BYTES", len(clip))

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = _mock_cosy_ok()
        mock_client_cls.return_value = mock_client
        _post_prebake(client, [
            {"target_text": "first clip", "prompt_text": "first clip"},
            {"target_text": "second clip", "prompt_text": "second clip"},
        ])
        _wait_task()

    remaining = list(main.COSY3_CACHE_DIR.glob("*.wav"))
    assert len(remaining) == 1  # the oldest was evicted to make room
    assert remaining[0].stat().st_size == len(clip)

"""The backend must warm the Whisper intelligibility model at startup.

First-use lazy loading takes ~15s, which once pushed /transcribe past the
Next.js dev-proxy timeout and surfaced as a bogus 500 to the learner even
though the backend eventually returned 200. These tests pin the lifespan
preload behaviour without loading any real model.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import api.main as main
from src import intelligibility


def _wait_for(calls: list, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while not calls and time.time() < deadline:
        time.sleep(0.05)


def test_lifespan_preloads_intelligibility_model(monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(main, "load_model", lambda: (None, None, "cpu"))
    monkeypatch.setattr(main, "intelligibility_available", lambda: True)
    monkeypatch.setattr(
        main, "intelligibility_preload", lambda: calls.append(True) or True
    )

    with TestClient(main.app):
        _wait_for(calls)

    assert calls, "lifespan should kick off intelligibility preload at startup"


def test_lifespan_skips_preload_when_model_missing(monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(main, "load_model", lambda: (None, None, "cpu"))
    monkeypatch.setattr(main, "intelligibility_available", lambda: False)
    monkeypatch.setattr(
        main, "intelligibility_preload", lambda: calls.append(True) or True
    )

    with TestClient(main.app):
        time.sleep(0.3)

    assert not calls, "preload must be skipped when no whisper model is present"


def test_preload_delegates_to_ensure_model(monkeypatch):
    seen: list[bool] = []
    monkeypatch.setattr(
        intelligibility, "_ensure_model", lambda: seen.append(True) or True
    )
    assert intelligibility.preload() is True
    assert seen == [True]


@pytest.mark.slow
def test_lifespan_preload_loads_real_whisper_model(tmp_path, monkeypatch):
    """Real startup, real models: after lifespan, whisper is actually in memory.

    This is the end-to-end proof of the fix: we reset the module's model
    globals first, so the model can only become non-None if the lifespan
    preload task really ran and succeeded.
    """
    if not intelligibility.available():
        pytest.skip("no local whisper model")

    from src import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "attempts.db")
    monkeypatch.setattr(storage, "RECORDINGS_DIR", tmp_path / "recordings")
    storage.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "_CONN", None)

    # Force a cold start: the model must be loaded by the lifespan preload.
    monkeypatch.setattr(intelligibility, "_proc", None)
    monkeypatch.setattr(intelligibility, "_model", None)
    monkeypatch.setattr(intelligibility, "_loaded_dir", None)

    with TestClient(main.app):
        task = main._preload_task
        assert task is not None, "lifespan should have started the preload task"
        deadline = time.time() + 120
        while not task.done() and time.time() < deadline:
            time.sleep(0.5)
        assert task.done(), "preload task did not finish within 120s"
        assert task.exception() is None, f"preload raised: {task.exception()}"
        assert intelligibility._model is not None
        assert intelligibility._proc is not None

    if storage._CONN is not None:
        storage._CONN.close()
        storage._CONN = None

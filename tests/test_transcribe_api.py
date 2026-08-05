"""End-to-end tests for the /transcribe endpoint.

These tests use the real wav2vec2 model and a real webm/opus container (built
from espeak-ng speech via ffmpeg), so they are marked as slow. They prove that
browser-like audio uploads decode, transcribe and produce an analysis.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import storage

SENTENCE = "The quick brown fox jumps over the lazy dog."


def _synthesize_webm_sample() -> bytes:
    """Return a webm/opus blob containing synthesized English speech."""
    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng not installed")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wav_path = tmp_path / "speech.wav"
        webm_path = tmp_path / "speech.webm"

        subprocess.run(
            ["espeak-ng", "-v", "en-us", SENTENCE, "-w", str(wav_path)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-c:a",
                "libopus",
                "-b:a",
                "24k",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "webm",
                str(webm_path),
            ],
            check=True,
            capture_output=True,
        )
        return webm_path.read_bytes()


@pytest.fixture(scope="module")
def webm_sample() -> bytes:
    return _synthesize_webm_sample()


@pytest.fixture(scope="module")
def client_module():
    """TestClient that loads the real model once for the module."""
    import tempfile

    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    storage.DB_PATH = tmp_path / "attempts.db"
    storage.RECORDINGS_DIR = tmp_path / "recordings"
    storage.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    storage._CONN = None

    from api.main import app

    with TestClient(app) as c:
        yield c

    if storage._CONN is not None:
        storage._CONN.close()
        storage._CONN = None
    tmp.cleanup()


@pytest.mark.slow
def test_transcribe_real_webm_returns_analysis(client_module, webm_sample):
    """A real webm/opus recording should decode, transcribe and analyse."""
    response = client_module.post(
        "/transcribe",
        data={
            "target_text": SENTENCE,
            "language": "en-us",
        },
        files={"audio": ("speech.webm", webm_sample, "audio/webm")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["duration_s"] > 0
    assert isinstance(body["tokens"], list)
    assert len(body["tokens"]) > 0
    assert "analysis" in body
    analysis = body["analysis"]
    assert "overall_score" in analysis
    assert isinstance(analysis["words"], list)
    assert len(analysis["words"]) > 0


@pytest.mark.slow
def test_transcribe_real_webm_falls_back_when_stdin_decode_fails(client_module, webm_sample, monkeypatch):
    """Even when ffmpeg cannot decode from stdin, the temp-file fallback works."""
    import src.models as models

    def _failing_stdin(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"stdin decode failed")

    monkeypatch.setattr(models, "_ffmpeg_decode_stdin", _failing_stdin)

    response = client_module.post(
        "/transcribe",
        data={
            "target_text": SENTENCE,
            "language": "en-us",
        },
        files={"audio": ("speech.webm", webm_sample, "audio/webm")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["duration_s"] > 0
    assert "analysis" in body
    assert len(body["analysis"]["words"]) > 0

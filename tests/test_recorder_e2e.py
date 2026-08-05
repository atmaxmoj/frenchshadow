"""True end-to-end test: browser MediaRecorder → backend /transcribe → analysis.

This test drives a real Chromium instance, records a synthetic voice clip via a
fake microphone, uploads it through the same /transcribe endpoint the UI uses,
and verifies the backend decodes the container and returns an analysis.

It is marked slow because it loads the wav2vec2 model and starts a browser.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _FakeMicWav:
    def __init__(self, path: Path, tmp: tempfile.TemporaryDirectory):
        self.path = path
        self._tmp = tmp


def _make_fake_mic_wav(text: str = "The quick brown fox jumps over the lazy dog.") -> _FakeMicWav:
    """Create a 16 kHz mono 16-bit PCM WAV suitable for Chromium fake audio capture."""
    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng not installed")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")

    tmp = tempfile.TemporaryDirectory()
    wav_path = Path(tmp.name) / "speech.wav"
    out_path = Path(tmp.name) / "fake_mic.wav"

    subprocess.run(
        ["espeak-ng", "-v", "en-us", text, "-w", str(wav_path)],
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
            "-ar",
            "16000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            "-y",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return _FakeMicWav(out_path, tmp)


@pytest.fixture(scope="session")
def backend_url():
    """Start a real backend process and return its URL."""
    project_root = Path(__file__).resolve().parent.parent
    port = _free_port()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [str(project_root / ".venv" / "bin" / "uvicorn"), "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"

    try:
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("backend failed to start")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.slow
def test_browser_recording_uploads_and_returns_analysis(backend_url):
    """Chromium records a fake mic clip, uploads it, and gets a 200 + analysis."""
    fake_wav = _make_fake_mic_wav()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-audio-capture={fake_wav.path}",
            ]
        )
        page = browser.new_page()
        # A blank localhost page satisfies the secure-context requirement for
        # getUserMedia while keeping the test independent of the Next.js dev server.
        page.goto("http://localhost:8768")

        result = page.evaluate(
            """async ({ backendUrl, targetText }) => {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const recorder = new MediaRecorder(stream);
                const chunks = [];
                recorder.ondataavailable = (e) => {
                    if (e.data.size > 0) chunks.push(e.data);
                };

                const blob = await new Promise((resolve, reject) => {
                    recorder.onstop = () => {
                        // Wait a tick for the final dataavailable event, exactly like
                        // the application does, so this test exercises the real path.
                        setTimeout(() => {
                            resolve(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }));
                        }, 100);
                    };
                    recorder.onerror = (e) => reject(e);
                    recorder.start();
                    setTimeout(() => recorder.stop(), 3500);
                });

                const header = new Uint8Array(await blob.slice(0, 16).arrayBuffer());
                const headerHex = Array.from(header)
                    .map((b) => b.toString(16).padStart(2, "0"))
                    .join("");

                const form = new FormData();
                form.append("audio", blob, "recording.webm");
                form.append("target_text", targetText);
                form.append("language", "en-us");

                const res = await fetch(`${backendUrl}/transcribe`, {
                    method: "POST",
                    body: form,
                });
                const body = await res.json();
                return {
                    status: res.status,
                    headerHex,
                    blobSize: blob.size,
                    mimeType: blob.type,
                    body,
                };
            }""",
            {"backendUrl": backend_url, "targetText": "The quick brown fox jumps over the lazy dog."},
        )

        browser.close()

    assert result["status"] == 200, result
    assert result["blobSize"] > 0
    header = result["headerHex"]
    assert (
        header.startswith("1a45dfa3")  # WebM/Matroska EBML header
        or header.startswith("4f676753")  # Ogg
        or header[8:16] == "66747970"  # MP4 ftyp at offset 4
    ), f"unexpected container header: {header}"
    assert "analysis" in result["body"]
    assert isinstance(result["body"]["analysis"]["words"], list)

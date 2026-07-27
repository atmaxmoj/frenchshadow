"""Tests for the TTS module."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src import tts


def _clear_cache():
    for p in tts.CACHE_DIR.glob("*.wav"):
        p.unlink()
    for p in tts.CACHE_DIR.glob("*.mp3"):
        p.unlink()


def test_synthesize_raises_on_empty_text():
    with pytest.raises(ValueError, match="empty text"):
        tts.synthesize("   ")


def test_synthesize_returns_wav_and_caches(monkeypatch, tmp_path):
    _clear_cache()
    fake_wav = b"RIFF\x00\x00\x00\x00WAVE"

    def fake_edge_tts(text: str, voice: str, out_path: Path) -> bool:
        # Write a fake mp3; synthesize will convert it through ffmpeg,
        # so instead we monkeypatch _convert_to_wav to just write the wav.
        out_path.write_bytes(b"fake-mp3")
        return True

    monkeypatch.setattr(tts, "_edge_tts", fake_edge_tts)
    monkeypatch.setattr(
        tts,
        "_convert_to_wav",
        lambda src, dst: dst.write_bytes(fake_wav),
    )

    data = tts.synthesize("hello", language="en-us")
    assert data == fake_wav

    # Second call should hit the cache without invoking the backend.
    calls = {"n": 0}
    def counted_fake(*args, **kwargs):
        calls["n"] += 1
        return fake_edge_tts(*args, **kwargs)
    monkeypatch.setattr(tts, "_edge_tts", counted_fake)
    data2 = tts.synthesize("hello", language="en-us")
    assert data2 == fake_wav
    assert calls["n"] == 0

    _clear_cache()


def test_synthesize_falls_back_to_say(monkeypatch):
    _clear_cache()
    fake_wav = b"RIFF\x00\x00\x00\x00WAVE"

    monkeypatch.setattr(tts, "_edge_tts", lambda *a, **k: False)

    def fake_say(text: str, voice: str, out_path: Path) -> bool:
        out_path.write_bytes(fake_wav)
        return True

    monkeypatch.setattr(tts, "_say_tts", fake_say)

    data = tts.synthesize("bonjour", language="fr-fr")
    assert data == fake_wav
    _clear_cache()


def test_synthesize_raises_when_no_backend(monkeypatch):
    _clear_cache()
    monkeypatch.setattr(tts, "_edge_tts", lambda *a, **k: False)
    monkeypatch.setattr(tts, "_say_tts", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match="no TTS backend available"):
        tts.synthesize("hello")
    _clear_cache()


@pytest.mark.slow
def test_synthesize_integration():
    """Actually call edge-tts; skipped implicitly if the binary is missing."""
    if not shutil.which("edge-tts"):
        pytest.skip("edge-tts not installed")
    _clear_cache()
    data = tts.synthesize("hello", language="en-us")
    assert data.startswith(b"RIFF")
    assert len(data) > 44
    _clear_cache()

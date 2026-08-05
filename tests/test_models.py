"""Tests for model/audio utilities (no heavy model loading)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.models import TARGET_SR, load_audio


@pytest.fixture
def webm_sample() -> bytes:
    candidates = list(Path(__file__).with_suffix("").parent.parent.glob("recordings/*.webm"))
    if not candidates:
        pytest.skip("no recorded webm samples available")
    return candidates[0].read_bytes()


def test_load_audio_empty_blob_raises():
    with pytest.raises(ValueError, match="empty audio blob"):
        load_audio(b"")


def test_load_audio_invalid_blob_raises():
    with pytest.raises(subprocess.CalledProcessError):
        load_audio(b"this is not audio")


def test_load_audio_decodes_webm_sample(webm_sample: bytes):
    """A real webm/opus blob should decode to a mono 16kHz float32 array."""
    audio = load_audio(webm_sample)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.size > 0


def test_load_audio_target_sr_respected(webm_sample: bytes):
    audio = load_audio(webm_sample, target_sr=TARGET_SR)
    assert audio.size > 0


def test_load_audio_falls_back_to_temp_file_when_stdin_fails(webm_sample: bytes, monkeypatch):
    """If ffmpeg cannot decode from stdin, it should decode from a temp file."""
    import src.models as models

    def _failing_stdin(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"stdin decode failed")

    monkeypatch.setattr(models, "_ffmpeg_decode_stdin", _failing_stdin)
    audio = load_audio(webm_sample)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.size > 0

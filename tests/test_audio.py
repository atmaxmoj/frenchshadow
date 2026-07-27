"""Tests for audio preprocessing utilities."""

from __future__ import annotations

import numpy as np

from src.audio import reduce_noise


def test_reduce_noise_keeps_shape_and_dtype():
    """reduce_noise returns a float32 mono array of the same length."""
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    cleaned = reduce_noise(audio)
    assert cleaned.shape == audio.shape
    assert cleaned.dtype == np.float32

"""Audio utilities: file loading, microphone recording, and preprocessing."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    import noisereduce as nr

    _HAS_NR = True
except Exception:  # pragma: no cover
    _HAS_NR = False

logger = logging.getLogger(__name__)

TARGET_SR = 16000


def reduce_noise(audio: np.ndarray, samplerate: int = TARGET_SR) -> np.ndarray:
    """Apply stationary noise reduction to a mono float32 array.

    Falls back to the original audio if noisereduce is unavailable or fails.
    """
    if not _HAS_NR:
        return audio
    try:
        # Use a short leading window to estimate the noise profile when the
        # recording starts before the user speaks.
        cleaned = nr.reduce_noise(y=audio, sr=samplerate, stationary=True, prop_decrease=1.0)
        return cleaned.astype(np.float32)
    except Exception:
        logger.exception("noise reduction failed, returning original audio")
        return audio


def record_audio(duration: float = 3.0, samplerate: int = TARGET_SR) -> np.ndarray:
    """Record *duration* seconds from the default microphone.

    Returns a 1-D float32 mono array at *samplerate*.
    """
    logger.info("Recording %.1fs from microphone...", duration)
    frames = int(duration * samplerate)
    recording = sd.rec(frames, samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()
    logger.info("Recording finished")
    return recording.squeeze()


def save_audio(path: str, audio: np.ndarray, samplerate: int = TARGET_SR) -> None:
    """Save a mono float32 array to a WAV file."""
    sf.write(path, audio, samplerate)


def load_audio_file(path: str, samplerate: int = TARGET_SR) -> np.ndarray:
    """Load a WAV/FLAC/OGG/etc. file with soundfile, resampling to *samplerate*."""
    import librosa

    audio, _ = librosa.load(path, sr=samplerate, mono=True)
    return audio.astype(np.float32)

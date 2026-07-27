"""Audio utilities: file loading and microphone recording."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SR = 16000


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

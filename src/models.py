"""wav2vec2 phoneme recognition model wrapper.

Loads a single Apache-2.0 licensed model that outputs eSpeak IPA phones:
    facebook/wav2vec2-lv-60-espeak-cv-ft

This avoids the unlicensed slplab model while keeping the same error-location
pipeline (reference vs learner IPA alignment).
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
from typing import Any

import librosa
import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get(
    "SHADOW_READER_MODEL",
    "facebook/wav2vec2-lv-60-espeak-cv-ft",
)
TARGET_SR = 16000

_processor: AutoProcessor | None = None
_model: AutoModelForCTC | None = None


def load_model(device: str | None = None) -> tuple[AutoProcessor, AutoModelForCTC, str]:
    """Load the wav2vec2 model once and cache it globally.

    Parameters
    ----------
    device : str | None
        "cpu", "mps", or None to auto-select. Defaults to cpu for stability.

    Returns
    -------
    (processor, model, device_name)
    """
    global _processor, _model

    if _processor is not None and _model is not None:
        return _processor, _model, str(_model.device)

    if device is None:
        device = "cpu"

    logger.info("Loading %s on %s", MODEL_NAME, device)
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    _model = AutoModelForCTC.from_pretrained(MODEL_NAME).to(device).eval()
    logger.info("Model loaded on %s", device)
    return _processor, _model, device


def load_audio(raw: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode arbitrary audio bytes to a mono float32 array at *target_sr*.

    Tries librosa/soundfile first, then falls back to ffmpeg.
    """
    try:
        audio, _ = librosa.load(io.BytesIO(raw), sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except Exception:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ar",
                str(target_sr),
                "-ac",
                "1",
                "-f",
                "wav",
                "pipe:1",
            ],
            input=raw,
            capture_output=True,
            check=True,
        )
        audio, _ = librosa.load(io.BytesIO(proc.stdout), sr=target_sr, mono=True)
        return audio.astype(np.float32)


def transcribe(audio: np.ndarray, processor: Any | None = None, model: Any | None = None) -> dict:
    """Run CTC inference and return IPA tokens plus approximate token timings.

    Parameters
    ----------
    audio : np.ndarray
        1-D float32 array at 16 kHz.
    processor, model : optional
        Pre-loaded transformer objects. If omitted, ``load_model()`` is called.

    Returns
    -------
    dict with ``raw`` (space-separated IPA string), ``tokens`` (list), and
    ``token_times`` (list of seconds, same length as ``tokens``).
    """
    if processor is None or model is None:
        processor, model, _ = load_model()

    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(model.device)
    with torch.no_grad():
        logits = model(input_values).logits
    logp = torch.log_softmax(logits, dim=-1)  # (1, T, V) — for GOP forced-align
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    tokens = text.split()

    # Approximate token timestamps from CTC alignment.
    blank_id = getattr(processor.tokenizer, "pad_token_id", 0)
    frame_ids = pred_ids[0].cpu().tolist()
    duration_s = len(audio) / TARGET_SR
    n_frames = len(frame_ids)
    seconds_per_frame = duration_s / n_frames if n_frames else 0.0

    token_times: list[float] = []
    prev_id = -1
    for frame_idx, token_id in enumerate(frame_ids):
        if token_id != prev_id and token_id != blank_id:
            token_times.append(frame_idx * seconds_per_frame)
        prev_id = token_id

    # Safety: timings length may differ from tokens by one due to decoding quirks.
    if len(token_times) != len(tokens):
        token_times = [i * duration_s / max(len(tokens), 1) for i in range(len(tokens))]

    # Normalize tokens (strip stress/syllable markers) and keep only those with
    # a non-empty normalized form. The times stay aligned with the surviving
    # tokens so that analyzer indices map directly to seconds.
    normalized_pairs: list[tuple[str, float]] = []
    for tok, t in zip(tokens, token_times):
        norm = tok.rstrip(".0123456789")
        if norm:
            normalized_pairs.append((norm, t))

    return {
        "raw": text,
        "tokens": [p[0] for p in normalized_pairs],
        "token_times": [p[1] for p in normalized_pairs],
        "logp": logp.cpu(),
        "blank_id": blank_id,
    }

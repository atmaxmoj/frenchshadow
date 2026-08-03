"""达意 (intelligibility): would a listener recover the target words?

Operationalized the Munro-Derwing way — a "listener transcription" — using a
context-aware ASR (Whisper) instead of a human: run Whisper on the learner
audio, then compare what it HEARD to the target at the phoneme level. Whisper's
decoder softmax already runs the noisy-channel competition over its whole learned
vocabulary + language prior, so we neither hand-build a lexicon nor treat it as a
black box — we read its recovered transcript and measure phoneme recall.

This is decorrelated from GOP-based 发音: heavily-accented but understandable
speech scores high 达意 / low 发音. Whisper runs lazily and off the event loop.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from src.analyzer import align, reference_ipa_per_word

logger = logging.getLogger(__name__)

_MODELS = Path(__file__).resolve().parent.parent / "models"
_lock = threading.Lock()
_proc = None
_model = None
_loaded_dir: str | None = None

# Credit for how well each target phone survived into what Whisper heard.
_MATCH, _SUB, _DEL = 1.0, 0.25, 0.0

_LANG = {"fr-fr": "fr", "fr-ca": "fr", "en-us": "en", "en-gb": "en"}


def _candidate_dirs() -> list[Path]:
    """Whisper model dirs to try, best first. Prefer small; fall back to base."""
    return [
        d for name in ("whisper-small", "whisper-base")
        if (d := _MODELS / name) and (d / "model.safetensors").exists()
    ]


def available() -> bool:
    return bool(_candidate_dirs())


def _ensure_model() -> bool:
    global _proc, _model, _loaded_dir
    if _model is not None:
        return True
    with _lock:
        if _model is not None:
            return True
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        # Try each candidate; skip one that's still downloading / corrupt.
        for d in _candidate_dirs():
            try:
                p = WhisperProcessor.from_pretrained(str(d))
                m = WhisperForConditionalGeneration.from_pretrained(str(d)).eval()
                p.tokenizer.set_prefix_tokens(language="french", task="transcribe")
                _proc, _model, _loaded_dir = p, m, d.name
                logger.info("intelligibility model loaded: %s", d.name)
                return True
            except Exception as exc:
                logger.warning("skip whisper model %s: %s", d.name, exc)
        return False


def transcribe_heard(wav: np.ndarray, language: str = "fr-fr") -> str | None:
    """What a context-aware listener (Whisper) hears from the learner audio."""
    if not _ensure_model():
        return None
    import torch

    lang = _LANG.get(language.lower(), language.split("-")[0])
    feat = _proc(wav, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        gen = _model.generate(feat, language=lang, task="transcribe", max_new_tokens=160)
    return _proc.batch_decode(gen, skip_special_tokens=True)[0].strip()


def _flat_phones(text: str, language: str) -> list[str]:
    try:
        return [p for _, ph in reference_ipa_per_word(text, language) for p in ph]
    except Exception:
        return []


def score(wav: np.ndarray, target_text: str, language: str = "fr-fr") -> dict | None:
    """Return {heard, overall, per_word: [float]} or None if Whisper unavailable.

    per_word is aligned to reference_ipa_per_word(target_text) word order.
    """
    heard = transcribe_heard(wav, language)
    if heard is None:
        return None

    ref = reference_ipa_per_word(target_text, language)  # [(word, [phones])]
    target_flat = [p for _, ph in ref for p in ph]
    heard_flat = _flat_phones(heard, language)

    # Credit per target phone: recovered (match), near (sub), or lost (del).
    credit: dict[int, float] = {}
    for op in align(target_flat, heard_flat):
        if op.ref_index is None:
            continue
        credit[op.ref_index] = {"match": _MATCH, "sub": _SUB, "del": _DEL}.get(op.kind, _DEL)

    per_word: list[float] = []
    idx = 0
    total = 0.0
    for _, phones in ref:
        got = sum(credit.get(idx + k, _DEL) for k in range(len(phones)))
        per_word.append(round(got / max(len(phones), 1), 2))
        total += got
        idx += len(phones)

    overall = round(total / max(len(target_flat), 1), 2)
    return {"heard": heard, "overall": overall, "per_word": per_word}

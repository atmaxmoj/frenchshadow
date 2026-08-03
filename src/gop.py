"""Goodness-of-Pronunciation (GOP) from the CTC phoneme model's posteriors.

Instead of taking the recognizer's argmax phones and edit-distancing them (which
zeroes a phone the model merely failed to *label*), we force-align the canonical
phone sequence to the audio's frame posteriors and read off, per phone, how
confident the acoustic model is that it was actually pronounced. This gives
continuous, accent-tolerant, partial-credit scores.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


def gop_scores(
    logp: torch.Tensor,
    phones: list[str],
    processor: Any,
    blank_id: int,
) -> list[float] | None:
    """Per-phone GOP in [0,1], aligned to *phones*. None if it can't align.

    *logp*: (1, T, V) log-softmax emissions. *phones*: canonical IPA phones.
    """
    if not phones:
        return None
    try:
        tok = processor.tokenizer
        unk = getattr(tok, "unk_token_id", None)
        ids = [tok.convert_tokens_to_ids(p) for p in phones]
        if any(i is None or i == unk for i in ids):
            return None  # OOV phone — can't force-align a token we don't model

        emission = logp if logp.dim() == 3 else logp.unsqueeze(0)
        emission = emission.to(torch.float32).cpu()
        targets = torch.tensor([ids], dtype=torch.int32)
        if emission.shape[1] < len(ids):
            return None  # fewer frames than phones — unalignable

        aligned, scores = torchaudio.functional.forced_align(
            emission, targets, blank=blank_id
        )
        spans = torchaudio.functional.merge_tokens(
            aligned[0], scores[0], blank=blank_id
        )
        # merge_tokens averages the frame log-probs per span → GOP = exp(mean).
        gop = [float(np.exp(min(0.0, s.score))) for s in spans]
        if len(gop) != len(phones):
            return None
        return gop
    except Exception as exc:  # pragma: no cover — alignment can fail on edge cases
        logger.warning("gop alignment failed: %s", exc)
        return None

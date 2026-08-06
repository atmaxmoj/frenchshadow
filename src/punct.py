"""Punctuation restoration using a local XLM-RoBERTa-large token-classifier.

The model in ``models/punct/`` predicts the punctuation mark that should follow
each word (``0`` = none, ``.``, ``,``, ``?``, ``-``, ``:``).  Restoring
punctuation before sentence segmentation lets us split unpunctuated YouTube
auto-captions at real sentence boundaries instead of relying on pauses or a
hard word-count limit.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).with_suffix("").parent.parent / "models" / "punct"
MODEL_DIR = Path(os.environ.get("SHADOW_READER_PUNCT_MODEL", DEFAULT_MODEL_DIR))

# Sentence-breaking punctuation emitted by the model.
_TERMINAL_PUNCT = {".", "?", "!", "…"}

_tokenizer: Any | None = None
_model: Any | None = None
_id2label: dict[int, str] | None = None


def has_model() -> bool:
    """Return True if the punctuation model files are present."""
    return MODEL_DIR.is_dir() and (MODEL_DIR / "config.json").exists()


def _load() -> None:
    global _tokenizer, _model, _id2label
    if _tokenizer is not None:
        return
    if not has_model():
        raise RuntimeError(f"punctuation model not found at {MODEL_DIR}")

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    logger.info("Loading punctuation model from %s", MODEL_DIR)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    _model.eval()
    # id2label may be keyed by str or int depending on transformers version.
    raw_id2label = dict(_model.config.id2label)
    _id2label = {int(k): str(v) for k, v in raw_id2label.items()}


def _label_to_punct(label_id: int) -> str:
    if _id2label is None:
        return ""
    label = _id2label.get(label_id, "0")
    return "" if label == "0" else label


def _split_long_text(text: str, max_tokens: int = 450) -> list[str]:
    """Split *text* into chunks that fit the model's context window.

    Splits at sentence-like boundaries when possible; otherwise at whitespace.
    """
    if not _tokenizer:
        return [text]
    # Quick token-count estimate.  If it fits, no need to split.
    tokens = _tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) <= max_tokens:
        return [text]

    # Try to split on existing punctuation first.
    rough_splits = re.split(r"(?<=[.!?…])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for part in rough_splits:
        part_tokens = len(_tokenizer.encode(part, add_special_tokens=False))
        if current_tokens + part_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(part)
        current_tokens += part_tokens
    if current:
        chunks.append(" ".join(current))

    # If a single rough split is still too long, fall back to word chunks.
    final_chunks: list[str] = []
    for chunk in chunks:
        if len(_tokenizer.encode(chunk, add_special_tokens=True)) <= max_tokens:
            final_chunks.append(chunk)
            continue
        words = chunk.split()
        current_words: list[str] = []
        current_tokens = 0
        for word in words:
            word_tokens = len(_tokenizer.encode(word, add_special_tokens=False))
            if current_tokens + word_tokens > max_tokens and current_words:
                final_chunks.append(" ".join(current_words))
                current_words = []
                current_tokens = 0
            current_words.append(word)
            current_tokens += word_tokens
        if current_words:
            final_chunks.append(" ".join(current_words))
    return final_chunks or [text]


def restore_punctuation(text: str) -> str:
    """Return *text* with punctuation marks restored by the local model.

    Words are preserved; only spaces between words may gain a punctuation mark.
    """
    if not has_model():
        return text
    _load()
    assert _tokenizer is not None and _model is not None

    text = " ".join(text.split())
    if not text:
        return text

    chunks = _split_long_text(text)
    punctuated_chunks: list[str] = []

    for chunk in chunks:
        encoding = _tokenizer(
            chunk,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v for k, v in encoding.items() if k != "offset_mapping"}
        with torch.no_grad():
            logits = _model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)[0].tolist()
        offset_mapping = encoding["offset_mapping"][0].tolist()

        words = chunk.split()
        result_words: list[str] = []
        char_pos = 0
        for word in words:
            word_start = char_pos
            word_end = char_pos + len(word)

            # Use the last sub-word token that falls inside the word's span.
            last_label = 0
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == tok_end:
                    continue  # <s> / </s>
                if tok_start >= word_start and tok_end <= word_end:
                    last_label = predictions[i]
                elif tok_start >= word_end:
                    break

            punct = _label_to_punct(last_label)
            result_words.append(word + punct)
            char_pos = word_end + 1  # +1 for the space between words

        punctuated_chunks.append(" ".join(result_words))

    return " ".join(punctuated_chunks)


def punctuation_is_sparse(raw_entries: list[dict[str, Any]], threshold: float = 0.3) -> bool:
    """Return True if fewer than *threshold* of the entries end with terminal punctuation."""
    texts = [str(e.get("text", "")).strip() for e in raw_entries if str(e.get("text", "")).strip()]
    if not texts:
        return False
    terminals = sum(1 for t in texts if t[-1] in _TERMINAL_PUNCT)
    return terminals / len(texts) < threshold

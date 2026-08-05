"""Lightweight GLM-backed sentence translation for the shadow-reader UI.

Reads the same Zhipu GLM key used by ``src.grapheme`` and provides a tiny
batch-translation helper.  Failures are logged and returned as empty strings so
that missing translations never break the practice flow.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_KEY_PATH = Path(__file__).resolve().parent.parent / ".secrets" / "glm_key"
_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
_MODEL = "glm-4.6"


def _key() -> str | None:
    try:
        return _KEY_PATH.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None


def _glm(prompt: str, key: str, retries: int = 3) -> str:
    """Call GLM and return the raw assistant content."""
    body = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(
            _ENDPOINT,
            data=data,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.load(resp)["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise
    raise RuntimeError("GLM retries exhausted")


def translate_sentences(sentences: list[str], source_hint: str = "auto", target: str = "zh") -> list[str]:
    """Translate *sentences* to *target* using GLM.

    Returns a list order-aligned with the input.  On any failure individual
    sentences may be returned as empty strings.
    """
    key = _key()
    if not key:
        logger.warning("no GLM key found, skipping translation")
        return [""] * len(sentences)

    if not sentences:
        return []

    source = source_hint if source_hint != "auto" else "the source language of each sentence"
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    prompt = (
        f"Translate the following {source} sentences to {target}. "
        "Return ONLY a JSON object mapping integer indices to translations, "
        "for example: {\"0\": \"...\", \"1\": \"...\"}. "
        "Do not include markdown, explanations, or any other text.\n\n"
        f"{numbered}"
    )

    try:
        raw = _glm(prompt, key)
    except Exception as exc:
        logger.warning("GLM translation request failed: %s", exc)
        return [""] * len(sentences)

    # Extract a JSON object from the response leniently.
    obj: dict | None = None
    text = raw.strip()
    if "{" in text and "}" in text:
        try:
            obj = json.loads(text[text.find("{") : text.rfind("}") + 1])
        except Exception as exc:
            logger.debug("failed to parse GLM translation JSON: %s", exc)
    if not isinstance(obj, dict):
        logger.warning("GLM translation returned non-JSON: %r", raw)
        return [""] * len(sentences)

    return [str(obj.get(str(i), "")).strip() for i in range(len(sentences))]

"""Small shared helpers for the CosyVoice3 clone pipeline.

Kept dependency-free so both the heavyweight cosy3_service (separate venv)
and plain unit tests can import it.

- instruct_for(): pin the synthesis language — zero-shot guesses, and French
  clones came out sounding German.
- evict_cache(): bound the on-disk clone cache (performance is capped via
  SHADOW_READER_COSY_PREBAKE_MAX on the main backend).
"""

from __future__ import annotations

from pathlib import Path

_NAMES = {
    "fr": "French",
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
}


def language_name(language: str) -> str | None:
    """Map a BCP-47-ish code ('fr-fr', 'en-us') to CosyVoice3's language name."""
    if not language:
        return None
    return _NAMES.get(language.lower().split("-")[0].strip())


def instruct_for(language: str) -> str | None:
    """The instruct text that pins the synthesis language, or None if unknown."""
    name = language_name(language)
    if name is None:
        return None
    return f"You are a helpful assistant. Please speak in {name}.<|endofprompt|>"


def evict_cache(cache_dir: Path, max_bytes: int, pattern: str = "*.wav") -> int:
    """Delete oldest files (by mtime) until the cache fits max_bytes.

    Returns the number of files evicted. Files are only ever re-baked on
    demand, so eviction is safe — worst case the next request regenerates.
    """
    if max_bytes <= 0:
        return 0
    files = [p for p in cache_dir.glob(pattern) if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    evicted = 0
    while total > max_bytes and files:
        victim = files.pop(0)
        size = victim.stat().st_size
        victim.unlink(missing_ok=True)
        total -= size
        evicted += 1
    return evicted

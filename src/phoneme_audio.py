"""Isolated IPA phoneme audio via espeak-ng (Kirshenbaum phoneme input).

Lets the UI play a single target/produced sound so the learner can A/B them,
instead of hearing the whole word. Cached on disk.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE = Path(__file__).resolve().parent.parent / "phoneme_cache"
_CACHE.mkdir(parents=True, exist_ok=True)
_ESPEAK = shutil.which("espeak-ng") or "/opt/homebrew/bin/espeak-ng"

# French IPA → espeak-ng Kirshenbaum phoneme codes (validated to synthesize).
IPA_TO_ESPEAK = {
    "a": "a", "ɑ": "A", "ɑ̃": "A~", "e": "e", "ɛ": "E", "ɛ̃": "E~", "i": "i",
    "o": "o", "ɔ": "O", "ɔ̃": "O~", "ø": "2", "œ": "9", "œ̃": "9~", "ə": "@",
    "u": "u", "y": "y", "j": "j", "w": "w", "ɥ": "H",
    "b": "b", "d": "d", "f": "f", "g": "g", "k": "k", "l": "l", "m": "m",
    "n": "n", "ɲ": "n^", "ŋ": "N", "p": "p", "ʁ": "R", "s": "s", "ʃ": "S",
    "t": "t", "v": "v", "z": "z", "ʒ": "Z",
}

_LANG = {"fr-fr": "fr", "fr-ca": "fr", "en-us": "en-us", "en-gb": "en", "es-es": "es", "de-de": "de"}


def phoneme_wav(ipa: str, language: str = "fr-fr") -> bytes | None:
    """WAV bytes for an IPA phone, or a space-separated sequence (a syllable).

    A bare vowel — especially a nasal like /ɑ̃/ — is unnatural to hear and imitate,
    so the UI plays the whole syllable (e.g. ``"m ɑ̃"`` → *mɑ̃*). Passing a single
    phone still works. Returns None if any phone is unmapped or synthesis fails.
    """
    phones = ipa.split()
    codes = [IPA_TO_ESPEAK.get(p) for p in phones]
    if not codes or any(c is None for c in codes):
        return None
    esp = "".join(codes)
    key = hashlib.sha256(f"{language}:{ipa}".encode("utf-8")).hexdigest()
    path = _CACHE / f"{key}.wav"
    if path.exists():
        return path.read_bytes()
    lang = _LANG.get(language.lower(), "fr")
    try:
        proc = subprocess.run(
            [_ESPEAK, "-v", lang, f"[[{esp}]]", "--stdout", "-s", "120"],
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("phoneme synth failed for %r: %s", ipa, exc)
        return None
    if proc.returncode != 0 or len(proc.stdout) < 200:
        return None
    path.write_bytes(proc.stdout)
    return proc.stdout

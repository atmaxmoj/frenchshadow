"""French liaison detection and reference-text helpers.

Liaison is the pronunciation of a normally-silent final consonant when the
next word begins with a vowel or silent *h*. We detect it by comparing the
contextual phonemization of a phrase with the isolated phonemization of each
word.
"""

from __future__ import annotations

import os
import shutil
from typing import Sequence


def _ensure_espeak_env() -> None:
    """Set phonemizer's espeak environment variables if binaries are found."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY") and os.environ.get("PHONEMIZER_ESPEAK_PATH"):
        return
    candidates = [
        ("/opt/homebrew/bin/espeak-ng", "/opt/homebrew/lib/libespeak-ng.dylib"),
        ("/opt/homebrew/bin/espeak", "/opt/homebrew/lib/libespeak.dylib"),
        ("/usr/local/bin/espeak-ng", "/usr/local/lib/libespeak-ng.dylib"),
        ("/usr/bin/espeak-ng", "/usr/lib/x86_64-linux-gnu/libespeak-ng.so"),
        ("/usr/bin/espeak", "/usr/lib/x86_64-linux-gnu/libespeak.so"),
    ]
    for binary, library in candidates:
        if shutil.which(binary) and os.path.exists(library):
            os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", binary)
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", library)
            return


_ensure_espeak_env()


def _phonemize(text: str, language: str) -> list[list[str]]:
    """Return per-word IPA phone lists for *text*."""
    from phonemizer import phonemize as _phonemize_fn
    from phonemizer.separator import Separator

    sep = Separator(word=" | ", phone=" ")
    raw = _phonemize_fn(
        text,
        language=language,
        backend="espeak",
        with_stress=False,
        preserve_punctuation=False,
        separator=sep,
        strip=True,
    )
    return [part.strip().split() for part in raw.split("|") if part.strip()]


def _strip_punctuation(word: str) -> str:
    return word.strip(".,!?;:\"'()[]{}").lower()


def _words(text: str) -> list[str]:
    return [w for w in (_strip_punctuation(w) for w in text.split()) if w]


def detect_liaisons(text: str, language: str = "fr-fr") -> list[tuple[str, str]]:
    """Return liaison word pairs in *text*.

    A pair (w1, w2) means w1 gains a final consonant in context that is not
    present when w1 is phonemized in isolation, and w2 starts with a vowel
    sound.
    """
    language = language.lower()
    if not language.startswith("fr"):
        return []

    words = _words(text)
    if len(words) < 2:
        return []

    try:
        contextual = _phonemize(" ".join(words), language)
        isolated = [_phonemize(w, language)[0] for w in words]
    except Exception:
        return []

    if len(contextual) != len(isolated):
        return []

    pairs: list[tuple[str, str]] = []
    for i in range(len(words) - 1):
        ctx = contextual[i]
        iso = isolated[i]
        # A liaison consonant is one or more phones appended to the isolated form.
        if len(ctx) > len(iso) and ctx[: len(iso)] == iso:
            # Verify the next word starts with a vowel in context.
            next_ctx = contextual[i + 1] if i + 1 < len(contextual) else []
            if next_ctx and next_ctx[0] in "aeiouyɛœøəɑɔɛ̃œ̃ɑ̃ø̃":
                pairs.append((words[i], words[i + 1]))
    return pairs


def reference_text_for_word(
    sentence: str,
    target_word: str,
    language: str = "fr-fr",
) -> str:
    """Return the text to synthesize for *target_word*'s reference audio.

    If *target_word* initiates a liaison with the next word, the returned text
    includes both words so the liaison consonant is audible.
    """
    language = language.lower()
    words = _words(sentence)
    target = _strip_punctuation(target_word)
    if language.startswith("fr") and target in words:
        idx = words.index(target)
        liaisons = detect_liaisons(sentence, language)
        for w1, w2 in liaisons:
            if w1 == target and idx + 1 < len(words):
                return f"{words[idx]} {words[idx + 1]}"
    return target or target_word

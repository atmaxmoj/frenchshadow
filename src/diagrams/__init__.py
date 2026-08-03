"""Unified articulatory diagram generation.

Vowels and consonants are rendered from declarative JSON configurations by a
single renderer.  This keeps the visual language consistent and makes adding a
new phone a matter of editing data, not drawing code.
"""

from __future__ import annotations

import re

from .config import load_consonants, load_vowels
from .dynartmo import has_sagittal, render_sagittal
from .renderer import render_consonant, render_vowel

# Phones that are commonly written as digraphs or ASCII variants but should
# be treated as a single articulatory target.
_ALIASES: dict[str, str] = {
    "ɹ": "r",
    "ɾ": "r",
    "ʀ": "ʁ",
    "g": "ɡ",
    "tʃ": "ʃ",
    "dʒ": "ʒ",
    "ts": "s",
    "dz": "z",
    "ɥ": "j",
}

_VOWELS = load_vowels()
_CONSONANTS = load_consonants()


def _base_phone(phone: str) -> str:
    """Strip length, stress, syllable markers and nasalisation diacritics."""
    return re.sub(r"[\u0300-\u036Fːˑ̆̋́̄̀̏̌̂᷄᷅᷇᷆᷈᷉.0123456789]", "", phone).strip()


def _canonical_phone(phone: str) -> str:
    """Return a canonical phone suitable for looking up diagrams."""
    base = _base_phone(phone)
    return _ALIASES.get(base, base)


def has_diagram(phone: str) -> bool:
    """Return True if *phone* can be illustrated."""
    if has_sagittal(phone):
        return True
    canonical = _canonical_phone(phone)
    return canonical in _VOWELS or canonical in _CONSONANTS


def diagram(phone: str) -> str:
    """Return an SVG string illustrating the articulation of *phone*.

    Primary path is the parametric DYNARTmo midsagittal renderer (every French
    phoneme); the legacy schematic renderer and a placeholder remain as fallbacks.
    """
    sagittal = render_sagittal(phone)
    if sagittal is not None:
        return sagittal

    canonical = _canonical_phone(phone)

    vowel_cfg = _VOWELS.get(canonical)
    if vowel_cfg is not None:
        return render_vowel(vowel_cfg, label=phone)

    consonant_cfg = _CONSONANTS.get(canonical)
    if consonant_cfg is not None:
        return render_consonant(consonant_cfg, label=phone)

    return _placeholder(phone)


def _placeholder(phone: str) -> str:
    safe = phone.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160" width="200" height="160">
  <rect x="10" y="10" width="180" height="140" rx="12" fill="none" stroke="#5f6368" stroke-width="2" stroke-dasharray="6 4"/>
  <text x="100" y="75" text-anchor="middle" fill="#e8eaed" font-size="22" font-family="sans-serif">/{safe}/</text>
</svg>'''


def phones_for_diagrams(errors: list[dict]) -> list[str]:
    """Extract phones worth drawing from a list of error dicts."""
    phones: list[str] = []
    seen: set[str] = set()
    for e in errors:
        for key in ("expected", "actual"):
            phone = e.get(key)
            if phone and phone not in seen:
                seen.add(phone)
                phones.append(phone)
    return phones

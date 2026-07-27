"""Declarative diagram configurations.

Phone-specific diagram data lives in JSON files under data/; this module loads
and validates them.  The renderer only knows how to draw the generic shapes,
not which phone has which articulatory features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_DATA_DIR = Path(__file__).with_suffix("").parent / "data"


@dataclass(frozen=True)
class VowelConfig:
    phone: str
    x: float
    y: float
    rounded: bool


@dataclass(frozen=True)
class ConsonantConfig:
    phone: str
    place: Literal[
        "bilabial",
        "labiodental",
        "dental",
        "alveolar",
        "postalveolar",
        "retroflex",
        "alveolopalatal",
        "palatal",
        "velar",
        "uvular",
        "pharyngeal",
        "glottal",
    ]
    manner: Literal[
        "plosive",
        "fricative",
        "sibilant_fricative",
        "affricate",
        "nasal",
        "approximant",
        "lateral_approximant",
    ]
    voiced: bool


def _load_json(name: str) -> dict:
    path = _DATA_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def load_vowels() -> dict[str, VowelConfig]:
    raw = _load_json("vowels.json")
    return {
        phone: VowelConfig(phone=phone, **data)
        for phone, data in raw.items()
    }


def load_consonants() -> dict[str, ConsonantConfig]:
    raw = _load_json("consonants.json")
    return {
        phone: ConsonantConfig(phone=phone, **data)
        for phone, data in raw.items()
    }

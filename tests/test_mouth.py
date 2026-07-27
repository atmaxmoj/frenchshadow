"""Tests for articulatory diagram generation."""

from __future__ import annotations

import pytest

from src.diagrams import diagram, has_diagram, phones_for_diagrams


@pytest.mark.parametrize(
    "phone",
    ["b", "ʃ", "k", "g", "p", "t", "d", "m", "n", "l", "r", "h", "w", "j"],
)
def test_consonant_diagram_returns_schematic(phone):
    """All consonants return a unified feature-based sagittal schematic."""
    svg = diagram(phone)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert f"/{phone}/" in svg
    assert "M 20,180" in svg  # head outline path


@pytest.mark.parametrize(
    "phone",
    ["i", "e", "ɛ", "a", "u", "o", "ɔ", "ə", "y", "ø", "œ", "æ", "ɪ", "ʊ", "ɐ"],
)
def test_vowel_diagram_returns_mouth_profile(phone):
    """Vowels return a mouth-profile schematic matching the consonant style."""
    svg = diagram(phone)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "M 20,180" in svg  # same head outline as consonants
    assert f"/{phone}/" in svg


@pytest.mark.parametrize(
    "phone",
    ["w", "j", "l", "r", "ɹ", "h", "tʃ", "dʒ", "ts", "dz"],
)
def test_common_consonant_has_diagram(phone):
    """Common consonants that lack bundled assets still get a schematic."""
    svg = diagram(phone)
    assert "<svg" in svg
    assert "</svg>" in svg


def test_phones_for_diagrams():
    errors = [
        {"expected": "θ", "actual": "s"},
        {"expected": "ð", "actual": "z"},
    ]
    assert phones_for_diagrams(errors) == ["θ", "s", "ð", "z"]


def test_has_diagram_coverage():
    """Common phones are covered; truly unknown phones are not."""
    assert has_diagram("p") is True
    assert has_diagram("w") is True
    assert has_diagram("j") is True
    assert has_diagram("ə") is True
    assert has_diagram("\u0000") is False

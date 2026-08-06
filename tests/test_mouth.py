"""Tests for articulatory diagram generation."""

from __future__ import annotations

import pytest

from src.diagrams import diagram, has_diagram, phones_for_diagrams
from src.diagrams.dynartmo import has_sagittal


# Phones with a DYNARTmo parametric midsagittal contour.
_DYNARTMO_CONSONANTS = ["b", "ʃ", "k", "g", "p", "t", "d", "m", "n", "l", "r", "w", "j"]
_DYNARTMO_VOWELS = ["i", "e", "ɛ", "a", "u", "o", "ɔ", "ə", "y", "ø", "œ"]

# Phones that fall back to the legacy feature-based schematic.
_SCHEMATIC_CONSONANTS = ["h"]
_SCHEMATIC_VOWELS = ["æ", "ɪ", "ʊ", "ɐ"]


@pytest.mark.parametrize("phone", _DYNARTMO_CONSONANTS + _DYNARTMO_VOWELS)
def test_dynartmo_phones_return_polyline_svg(phone):
    """Phones supported by DYNARTmo render as parametric midsagittal contours."""
    assert has_sagittal(phone)
    svg = diagram(phone)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert f"/{phone}/" in svg
    assert "<polyline" in svg


@pytest.mark.parametrize("phone", _SCHEMATIC_CONSONANTS + _SCHEMATIC_VOWELS)
def test_fallback_phones_return_schematic_svg(phone):
    """Phones without DYNARTmo params fall back to the unified schematic."""
    assert not has_sagittal(phone)
    svg = diagram(phone)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert f"/{phone}/" in svg
    assert "M 20,180" in svg  # head outline path of the legacy renderer


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

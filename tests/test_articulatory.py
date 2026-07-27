"""Tests for articulatory tip generation."""

from __future__ import annotations

from src.articulatory import TIPS, get_tip


def test_custom_tip_for_known_pattern():
    tip = get_tip("ɹ", "l", "R → L")
    assert "/r/" in tip["description"] or "R" in tip["description"]
    assert tip["tongue"]
    assert tip["lips"]
    assert tip["jaw"]


def test_generic_tip_for_unknown_pair():
    tip = get_tip("z", "d", "sound mismatch (z → d)")
    assert "z" in tip["description"]
    assert "d" in tip["description"]
    assert tip["tongue"]


def test_insertion_tip():
    tip = get_tip(None, "ə", "extra sound (ə)")
    assert tip["description"]
    assert "多加" in tip["description"]


def test_deletion_tip():
    tip = get_tip("k", None, "missing sound (k)")
    assert "漏掉" in tip["description"]


def test_tip_includes_diagram_urls():
    tip = get_tip("θ", "s", "voiceless th → s")
    assert tip["diagram_expected"] == "/mouth_diagram?phone=θ"
    assert tip["diagram_actual"] == "/mouth_diagram?phone=s"


def test_insertion_tip_has_only_actual_diagram():
    tip = get_tip(None, "ə", "extra sound (ə)")
    assert tip["diagram_expected"] is None
    assert tip["diagram_actual"] == "/mouth_diagram?phone=ə"


def test_all_custom_tips_have_required_fields():
    for label, tip in TIPS.items():
        assert tip.description
        assert tip.tongue
        assert tip.lips
        assert tip.jaw
        assert tip.practice

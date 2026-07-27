"""Unit tests for the analyzer module.

These tests do not load the wav2vec2 model; they operate directly on IPA token
sequences to validate alignment and classification logic.
"""

from __future__ import annotations

import pytest

from src.analyzer import (
    L1_PATTERN_LABELS,
    AlignmentOp,
    VOWEL_REDUCTION_PAIRS,
    align,
    analyze,
    analyze_words,
    reference_ipa_per_word,
    split_sentences,
    sub_cost,
)


def test_sub_cost_match():
    assert sub_cost("ɑ", "ɑ") == 0


def test_sub_cost_l1_pattern():
    assert sub_cost("θ", "s") == 1
    assert sub_cost("v", "w") == 1
    assert sub_cost("ɹ", "l") == 1


def test_sub_cost_vowel_reduction():
    assert sub_cost("ə", "ʌ") == 1
    assert sub_cost("i", "ɪ") == 1


def test_sub_cost_unrelated():
    assert sub_cost("a", "z") == 3


def test_align_identical():
    ops = align(["a", "b", "c"], ["a", "b", "c"])
    assert len(ops) == 3
    assert all(op.kind == "match" for op in ops)


def test_align_substitution():
    ops = align(["ɹ", "ɪ", "l", "i"], ["l", "ɪ", "l", "i"])
    assert ops[0].kind == "sub"
    assert ops[0].ref == "ɹ"
    assert ops[0].learner == "l"
    assert all(op.kind == "match" for op in ops[1:])


def test_align_insertion():
    ops = align(["a", "b"], ["a", "x", "b"])
    kinds = [op.kind for op in ops]
    assert "ins" in kinds


def test_align_deletion():
    ops = align(["a", "b", "c"], ["a", "c"])
    kinds = [op.kind for op in ops]
    assert "del" in kinds


def test_align_l1_pattern_preferred_over_insdel():
    ops = align(["ɹ"], ["l"])
    assert len(ops) == 1
    assert ops[0].kind == "sub"


def test_analyze_words_really_rl():
    ref = [("really", ["ɹ", "ɪ", "l", "i"])]
    learner = ["l", "ɪ", "l", "i"]
    out = analyze_words(ref, learner)
    assert len(out) == 1
    w = out[0]
    assert w.word == "really"
    assert len(w.errors) == 1
    e = w.errors[0]
    assert e.expected == "ɹ"
    assert e.actual == "l"
    assert e.l1_pattern is True
    assert "R → L" in e.label
    assert w.score == 0.75


def test_analyze_words_th_to_s():
    ref = [("think", ["θ", "ɪ", "ŋ", "k"])]
    learner = ["s", "ɪ", "ŋ", "k"]
    out = analyze_words(ref, learner)
    e = out[0].errors[0]
    assert e.expected == "θ"
    assert e.actual == "s"
    assert "voiceless th → s" in e.label


def test_analyze_words_v_to_w():
    ref = [("very", ["v", "ɛ", "ɹ", "i"])]
    learner = ["w", "ɛ", "ɹ", "i"]
    out = analyze_words(ref, learner)
    e = out[0].errors[0]
    assert e.expected == "v"
    assert e.actual == "w"
    assert "V → W" in e.label


def test_analyze_words_clean():
    ref = [("blue", ["b", "l", "u"])]
    learner = ["b", "l", "u"]
    out = analyze_words(ref, learner)
    assert out[0].errors == []
    assert out[0].score == 1.0


def test_analyze_words_vowel_reduction_is_not_error():
    ref = [("the", ["ð", "ə"])]
    learner = ["ð", "ʌ"]
    out = analyze_words(ref, learner)
    assert out[0].errors == []
    assert out[0].score == 1.0


def test_analyze_top_level():
    text = "I think this is good."
    # Hypothetical learner output: "aɪ s ɪ ŋ k d ɪ s ɪ z ɡ ʊ d"
    learner = ["aɪ", "s", "ɪ", "ŋ", "k", "d", "ɪ", "s", "ɪ", "z", "ɡ", "ʊ", "d"]
    result = analyze(text, learner)
    assert 0.0 <= result["overall_score"] <= 1.0
    words = {w["word"]: w for w in result["words"]}
    assert "think" in words
    assert len(words["think"]["errors"]) >= 1


def test_split_sentences():
    assert split_sentences("Hello world. How are you?") == [
        "Hello world.",
        "How are you?",
    ]


def test_reference_ipa_basic():
    ref = reference_ipa_per_word("I like rice")
    words = [w for w, _ in ref]
    assert words == ["I", "like", "rice"]
    # Each word should have at least one phone.
    assert all(len(phones) >= 1 for _, phones in ref)


def test_reference_ipa_strips_punctuation():
    ref = reference_ipa_per_word("Hello, world!")
    words = [w for w, _ in ref]
    assert "Hello" in words
    assert "world" in words
    assert "Hello," not in words
    assert "world!" not in words

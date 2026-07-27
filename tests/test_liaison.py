"""Tests for French liaison detection."""

from __future__ import annotations

from unittest.mock import patch

from src.liaison import detect_liaisons, reference_text_for_word


def _mock_phonemize(contextual: dict[str, list[list[str]]], isolated: dict[str, list[list[str]]]):
    """Return a fake phonemize function."""

    def fn(text: str, language: str) -> list[list[str]]:
        # Normalize whitespace and punctuation for lookup.
        key = " ".join(text.strip(".,!?;:\"'()[]{}").split()).lower()
        if " " in key and key in contextual:
            return contextual[key]
        if key in isolated:
            return isolated[key]
        raise ValueError(f"unexpected text: {text!r}")

    return fn


def test_detect_liaisons_basic():
    contextual = {
        "les amis": [["l", "e", "z"], ["a", "m", "i"]],
    }
    isolated = {
        "les": [["l", "e"]],
        "amis": [["a", "m", "i"]],
    }
    with patch("src.liaison._phonemize", side_effect=_mock_phonemize(contextual, isolated)):
        assert detect_liaisons("les amis", "fr-fr") == [("les", "amis")]


def test_detect_liaisons_none():
    contextual = {
        "le chat": [["l", "ə"], ["ʃ", "a"]],
    }
    isolated = {
        "le": [["l", "ə"]],
        "chat": [["ʃ", "a"]],
    }
    with patch("src.liaison._phonemize", side_effect=_mock_phonemize(contextual, isolated)):
        assert detect_liaisons("le chat", "fr-fr") == []


def test_detect_liaisons_ignores_non_french():
    assert detect_liaisons("the apple", "en-us") == []


def test_reference_text_for_word_returns_pair_on_liaison():
    contextual = {
        "petit ami": [["p", "ə", "t", "i", "t"], ["a", "m", "i"]],
    }
    isolated = {
        "petit": [["p", "ə", "t", "i"]],
        "ami": [["a", "m", "i"]],
    }
    with patch("src.liaison._phonemize", side_effect=_mock_phonemize(contextual, isolated)):
        assert reference_text_for_word("petit ami", "petit", "fr-fr") == "petit ami"
        assert reference_text_for_word("petit ami", "ami", "fr-fr") == "ami"


def test_reference_text_for_word_non_french():
    assert reference_text_for_word("the apple", "the", "en-us") == "the"

"""CosyVoice3 pipeline helpers: language instruct + cache eviction (no model)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from cosy3_util import evict_cache, instruct_for, language_name


def test_french_maps_to_french():
    assert language_name("fr-fr") == "French"
    assert language_name("fr-ca") == "French"
    assert language_name("fr") == "French"


def test_english_and_others():
    assert language_name("en-us") == "English"
    assert language_name("en-gb") == "English"
    assert language_name("zh") == "Chinese"
    assert language_name("ja") == "Japanese"


def test_unknown_or_empty_returns_none():
    assert language_name("") is None
    assert language_name("xx-yy") is None


def test_instruct_format():
    assert (
        instruct_for("fr-fr")
        == "You are a helpful assistant. Please speak in French.<|endofprompt|>"
    )
    assert instruct_for("xx") is None


def test_evict_cache_removes_oldest_until_under_cap(tmp_path):
    old = tmp_path / "old.wav"
    new = tmp_path / "new.wav"
    old.write_bytes(b"x" * 100)
    time.sleep(0.01)
    new.write_bytes(b"x" * 100)
    # Force distinct mtimes even on coarse filesystems.
    import os
    os.utime(old, (1000, 1000))

    evicted = evict_cache(tmp_path, max_bytes=150)
    assert evicted == 1
    assert not old.exists()
    assert new.exists()


def test_evict_cache_noop_when_under_cap(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x" * 10)
    assert evict_cache(tmp_path, max_bytes=1000) == 0
    assert (tmp_path / "a.wav").exists()


def test_evict_cache_ignores_non_matching_files(tmp_path):
    (tmp_path / "keep.txt").write_bytes(b"x" * 500)
    (tmp_path / "a.wav").write_bytes(b"x" * 100)
    evict_cache(tmp_path, max_bytes=10)
    assert (tmp_path / "keep.txt").exists()
    assert not (tmp_path / "a.wav").exists()


def test_evict_cache_zero_cap_disables(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x" * 100)
    assert evict_cache(tmp_path, max_bytes=0) == 0
    assert (tmp_path / "a.wav").exists()

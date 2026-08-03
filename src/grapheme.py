"""Map an IPA phoneme back to the letter(s) that spell it in a French word.

Rule-based French G2P alignment is unreliable, so we ask GLM (glm-4.6, which is
accurate at this) and cache each (word, phone) forever. Returns the word with
the relevant letters wrapped in 【 】; on any failure returns the plain word so
the UI still shows the spelling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_KEY_PATH = Path(__file__).resolve().parent.parent / ".secrets" / "glm_key"
_CACHE_DIR = Path(__file__).resolve().parent.parent / "grapheme_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
_MODEL = "glm-4.6"


def _key() -> str | None:
    try:
        return _KEY_PATH.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None


def _cache_path(word: str, phone: str) -> Path:
    h = hashlib.sha256(f"{word}{phone}".encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{h}.txt"


def _glm(prompt: str, key: str, retries: int = 4) -> str:
    """Call GLM, retrying on rate-limit / transient server errors with backoff."""
    body = {"model": _MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(
            _ENDPOINT,
            data=data,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.load(resp)["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as exc:
            # 429 rate-limit and 5xx are transient; back off and retry.
            if exc.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(1.5 * (2 ** attempt))  # 1.5, 3, 6 s
                continue
            raise


def _parse_indexed(raw: str, n: int) -> list[str | None] | None:
    """Leniently parse GLM's ``{"0": "...", ...}`` reply.

    Returns a list of length *n* where entries GLM answered are strings and
    missing / non-string entries are None — so one unmarkable sound cannot void
    the whole batch.  Returns None only when nothing JSON-like parses at all.
    """
    s = raw.strip()
    if "{" not in s or "}" not in s:
        return None
    try:
        obj = json.loads(s[s.find("{") : s.rfind("}") + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return [v.strip() if isinstance((v := obj.get(str(i))), str) else None
            for i in range(n)]


def mark_graphemes(pairs: list[tuple[str, str]], language: str = "fr-fr") -> list[str]:
    """For each (word, phone), return the word with that sound's letters in 【 】.

    Order-aligned with *pairs*. Falls back to the plain word on any failure.
    """
    results: list[str | None] = [None] * len(pairs)
    todo: list[int] = []
    for i, (word, phone) in enumerate(pairs):
        p = _cache_path(word, phone)
        if p.exists():
            results[i] = p.read_text(encoding="utf-8")
        elif word and phone:
            todo.append(i)
        else:
            results[i] = word

    key = _key()
    if todo and key:
        lines = "\n".join(f"{k}. word=\"{pairs[i][0]}\" sound=[{pairs[i][1]}]" for k, i in enumerate(todo))
        prompt = (
            "For each French word below, wrap the letter(s) that spell the given IPA sound in 【 】, "
            "leaving all other letters unchanged. Output ONLY a JSON object mapping each item number "
            '(string) to the marked word, e.g. {"0":"aujou【r】d\'hui"}.\n\n' + lines
        )
        marked = None
        try:
            marked = _parse_indexed(_glm(prompt, key), len(todo))
        except Exception as exc:
            logger.warning("grapheme GLM failed: %s", exc)
        for pos, i in enumerate(todo):
            word = pairs[i][0]
            cand = marked[pos] if marked else None
            if cand and "【" in cand:
                results[i] = cand
                _cache_path(word, pairs[i][1]).write_text(cand, encoding="utf-8")
            else:
                results[i] = word  # plain fallback (not cached — retry next time)

    return [r if isinstance(r, str) else pairs[j][0] for j, r in enumerate(results)]

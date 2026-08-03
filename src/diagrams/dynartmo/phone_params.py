"""French IPA → DYNARTmo articulator parameters.

One entry per French phoneme; the DYNARTmo model interpolates the actual
contour, so this table alone gives full coverage (oral + nasal vowels, glides,
every consonant) with nothing missing.  Values are the non-zero fields of the
8-parameter vector documented in :mod:`.model`.

Conventions:
  toDors1  tongue high(+1000)/low(-1000)   toDors2  front(+1000)/back(-1000)
  toDors3  dorsum closure (velar/uvular)   lips1    lip closure (labial)
  lips2    rounding                        toTip1   tongue-tip closure
  toTip2   tip front(+)/back(-)            velum    nasal (0 oral .. 1000 nasal)
"""

from __future__ import annotations

# --- Oral vowels ----------------------------------------------------------
_VOWELS: dict[str, dict] = {
    "i": {"toDors1": 1000, "toDors2": 1000},
    "y": {"toDors1": 1000, "toDors2": 1000, "lips2": 1000},
    "e": {"toDors1": 500, "toDors2": 1000},
    "ɛ": {"toDors1": 100, "toDors2": 900},
    "a": {"toDors1": -1000, "toDors2": 400},
    "ɑ": {"toDors1": -1000, "toDors2": -400},
    "ø": {"toDors1": 500, "toDors2": 900, "lips2": 1000},
    "œ": {"toDors1": 100, "toDors2": 800, "lips2": 900},
    "ə": {"toDors1": 0, "toDors2": 0, "lips2": 300},
    "u": {"toDors1": 1000, "toDors2": -1000, "lips2": 1000},
    "o": {"toDors1": 500, "toDors2": -1000, "lips2": 1000},
    "ɔ": {"toDors1": 100, "toDors2": -900, "lips2": 800},
}

# --- Nasal vowels (velum lowered) -----------------------------------------
_NASAL_VOWELS: dict[str, dict] = {
    "ɑ̃": {"toDors1": -800, "toDors2": -400, "lips2": 300, "velum": 1000},
    "ɔ̃": {"toDors1": 100, "toDors2": -900, "lips2": 800, "velum": 1000},
    "ɛ̃": {"toDors1": 100, "toDors2": 800, "velum": 1000},
    "œ̃": {"toDors1": 100, "toDors2": 700, "lips2": 800, "velum": 1000},
}

# --- Glides ---------------------------------------------------------------
_GLIDES: dict[str, dict] = {
    "j": {"toDors1": 1000, "toDors2": 1000},
    "ɥ": {"toDors1": 1000, "toDors2": 1000, "lips2": 1000},
    "w": {"toDors1": 1000, "toDors2": -1000, "lips2": 1000},
}

# --- Consonants -----------------------------------------------------------
# A mid-low neutral tongue base; the closure parameter carries the place cue.
_CONSONANTS: dict[str, dict] = {
    "p": {"toDors1": -400, "lips1": 900},
    "b": {"toDors1": -400, "lips1": 900},
    "m": {"toDors1": -400, "lips1": 900, "velum": 1000},
    "f": {"toDors1": -400, "lips1": 560},
    "v": {"toDors1": -400, "lips1": 560},
    "t": {"toDors1": -400, "toTip1": 1000},
    "d": {"toDors1": -400, "toTip1": 1000},
    "n": {"toDors1": -400, "toTip1": 1000, "velum": 1000},
    "l": {"toDors1": -300, "toTip1": 820},
    "s": {"toDors1": -400, "toTip1": 720},
    "z": {"toDors1": -400, "toTip1": 720},
    "ʃ": {"toDors1": -300, "toTip1": 660, "toTip2": -500},
    "ʒ": {"toDors1": -300, "toTip1": 660, "toTip2": -500},
    "ɲ": {"toDors1": 1000, "toDors2": 800, "toTip1": 500, "velum": 1000},
    "k": {"toDors1": -500, "toDors2": -500, "toDors3": 950},
    "g": {"toDors1": -500, "toDors2": -500, "toDors3": 950},
    "ŋ": {"toDors1": -500, "toDors2": -500, "toDors3": 950, "velum": 1000},
    "ʁ": {"toDors1": -400, "toDors2": -800, "toDors3": 820},
    "χ": {"toDors1": -400, "toDors2": -800, "toDors3": 760},
}

PHONE_PARAMS: dict[str, dict] = {
    **_VOWELS,
    **_NASAL_VOWELS,
    **_GLIDES,
    **_CONSONANTS,
}

# ASCII / variant spellings that share a target with a canonical phone.
_ALIASES: dict[str, str] = {
    "ɡ": "g",   # IPA script-g → plain g
    "ʀ": "ʁ",   # uvular trill → French R
    "r": "ʁ",   # generic r in a French context → uvular
    "ɾ": "ʁ",
    "ɹ": "ʁ",
    "tʃ": "ʃ",
    "dʒ": "ʒ",
    "ts": "s",
    "dz": "z",
    "ɡ̃": "ŋ",
}


def params_for(phone: str) -> dict | None:
    """Return the articulator params for *phone*, or None if unsupported.

    Nasal diacritics are significant (they select the velum-lowered variant),
    so unlike the legacy renderer this does not strip the combining tilde.
    """
    p = phone.strip()
    if p in PHONE_PARAMS:
        return PHONE_PARAMS[p]
    if p in _ALIASES:
        return PHONE_PARAMS[_ALIASES[p]]
    # Strip length/stress marks only (keep nasal tilde U+0303).
    stripped = "".join(
        c for c in p if c not in "ːˑ.ˈ,ˌ0123456789"
    ).strip()
    if stripped in PHONE_PARAMS:
        return PHONE_PARAMS[stripped]
    if stripped in _ALIASES:
        return PHONE_PARAMS[_ALIASES[stripped]]
    return None

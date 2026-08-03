"""Phoneme-level pronunciation analyzer.

Clean-room implementation of a reference-vs-learner IPA comparison pipeline,
inspired by open speech-recognition research. It intentionally does not copy
any code from third-party backends.

Pipeline
--------
1. Target text -> espeak / phonemizer -> reference IPA per word.
2. Learner audio -> wav2vec2 CTC -> learner IPA token sequence.
3. Needleman-Wunsch alignment between reference and learner IPA.
4. Substitution / insertion / deletion classification.
5. Per-word score + Chinese-L1-specific feedback.
"""

from __future__ import annotations

import os
import re
import shutil
import functools
from dataclasses import asdict, dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# espeak-ng discovery (macOS Homebrew / Linux)
# ---------------------------------------------------------------------------


def _ensure_espeak_env() -> None:
    """Set phonemizer's espeak environment variables if binaries are found."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY") and os.environ.get("PHONEMIZER_ESPEAK_PATH"):
        return
    candidates = [
        ("/opt/homebrew/bin/espeak-ng", "/opt/homebrew/lib/libespeak-ng.dylib"),
        ("/opt/homebrew/bin/espeak", "/opt/homebrew/lib/libespeak.dylib"),
        ("/usr/local/bin/espeak-ng", "/usr/local/lib/libespeak-ng.dylib"),
        ("/usr/bin/espeak-ng", "/usr/lib/x86_64-linux-gnu/libespeak-ng.so"),
        ("/usr/bin/espeak", "/usr/lib/x86_64-linux-gnu/libespeak.so"),
    ]
    for binary, library in candidates:
        if shutil.which(binary) and os.path.exists(library):
            os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", binary)
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", library)
            return


_ensure_espeak_env()

# ---------------------------------------------------------------------------
# Reference IPA generation
# ---------------------------------------------------------------------------


def reference_ipa_per_word(text: str, language: str = "en-us") -> list[tuple[str, list[str]]]:
    """Return [(word, [ipa_phone, ...]), ...] for *text*.

    Uses the ``phonemizer`` package with the espeak backend. espeak-ng must be
    installed on the system.
    """
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    sep = Separator(word=" | ", phone=" ")
    raw = phonemize(
        text,
        language=language,
        backend="espeak",
        with_stress=False,
        preserve_punctuation=False,
        separator=sep,
        strip=True,
    )
    words = [w.strip(".,!?;:\"'()[]{}") for w in text.split()]
    words = [w for w in words if w]
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    return [(word, phones.split()) for word, phones in zip(words, parts)]


# ---------------------------------------------------------------------------
# Alignment costs
# ---------------------------------------------------------------------------

MATCH_COST = 0
PATTERN_COST = 1
INSDEL_COST = 2
DIFF_COST = 3

# Common Mandarin-L1 -> English substitutions. Values are human-readable labels.
L1_PATTERN_LABELS: dict[tuple[str, str], str] = {
    # th
    ("θ", "s"): "voiceless th → s",
    ("θ", "t"): "voiceless th → t",
    ("θ", "f"): "voiceless th → f",
    ("ð", "z"): "voiced th → z",
    ("ð", "d"): "voiced th → d",
    ("ð", "v"): "voiced th → v",
    # v/w
    ("v", "w"): "V → W",
    ("w", "v"): "W → V",
    # r/l
    ("ɹ", "l"): "R → L",
    ("l", "ɹ"): "L → R",
    ("ɹ", "ɻ"): "R → retroflex R",
    # fricatives / affricates
    ("ʃ", "s"): "SH → S",
    ("ʒ", "z"): "ZH → Z",
    ("tʃ", "ts"): "CH → TS",
    ("tʃ", "t"): "CH → T",
    ("dʒ", "dz"): "J → DZ",
    ("dʒ", "d"): "J → D",
    ("dʒ", "z"): "J → Z",
    # final devoicing
    ("z", "s"): "Z → S devoicing",
    ("s", "z"): "S → Z voicing",
    ("d", "t"): "D → T devoicing",
    ("b", "p"): "B → P devoicing",
    ("g", "k"): "G → K devoicing",
    ("v", "f"): "V → F devoicing",
    # vowel confusions common for Mandarin speakers
    ("æ", "ɛ"): "A → E (mouth too closed)",
    ("ɛ", "æ"): "E → A (mouth too open)",
    ("ɪ", "i"): "I → EE (tense)",
    ("i", "ɪ"): "EE → I (lax)",
    ("ʊ", "u"): "U → OO (rounded)",
    ("u", "ʊ"): "OO → U (unrounded)",
    ("ɑ", "ʌ"): "AH → UH",
    ("ɔ", "ɑ"): "AW → AH",
}

# Native-speaker allophonic variation that should not be penalised.
VOWEL_REDUCTION_PAIRS: set[tuple[str, str]] = {
    ("ə", "ʌ"),
    ("ə", "ɑ"),
    ("ə", "ɝ"),
    ("ʌ", "ɝ"),
    ("i", "ɪ"),
    ("u", "ʊ"),
    ("ɛ", "e"),
    ("ɔ", "oʊ"),
    ("æ", "ɛ"),  # only for weak forms; kept here for leniency
}


def _in_pair_set(a: str, b: str, pairs: set[tuple[str, str]]) -> bool:
    return (a, b) in pairs or (b, a) in pairs


def sub_cost(a: str, b: str) -> int:
    """Cost of substituting phone *a* with phone *b*."""
    if a == b:
        return MATCH_COST
    if _in_pair_set(a, b, VOWEL_REDUCTION_PAIRS):
        return PATTERN_COST
    if (a, b) in L1_PATTERN_LABELS:
        return PATTERN_COST
    return DIFF_COST


# ---------------------------------------------------------------------------
# Needleman-Wunsch alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentOp:
    kind: str  # "match" | "sub" | "ins" | "del"
    ref: str | None
    learner: str | None
    ref_index: int | None
    learner_index: int | None


def align(ref: list[str], learner: list[str]) -> list[AlignmentOp]:
    """Align two phone sequences with Needleman-Wunsch."""
    n, m = len(ref), len(learner)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * INSDEL_COST
    for j in range(1, m + 1):
        dp[0][j] = j * INSDEL_COST

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost(ref[i - 1], learner[j - 1]),
                dp[i - 1][j] + INSDEL_COST,
                dp[i][j - 1] + INSDEL_COST,
            )

    ops: list[AlignmentOp] = []
    i, j = n, m
    while i > 0 and j > 0:
        cost = sub_cost(ref[i - 1], learner[j - 1])
        if dp[i][j] == dp[i - 1][j - 1] + cost:
            kind = "match" if cost == MATCH_COST else "sub"
            ops.append(
                AlignmentOp(kind, ref[i - 1], learner[j - 1], i - 1, j - 1)
            )
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + INSDEL_COST:
            ops.append(AlignmentOp("del", ref[i - 1], None, i - 1, None))
            i -= 1
        else:
            ops.append(AlignmentOp("ins", None, learner[j - 1], None, j - 1))
            j -= 1

    while i > 0:
        ops.append(AlignmentOp("del", ref[i - 1], None, i - 1, None))
        i -= 1
    while j > 0:
        ops.append(AlignmentOp("ins", None, learner[j - 1], None, j - 1))
        j -= 1

    ops.reverse()
    return ops


# ---------------------------------------------------------------------------
# Per-word analysis
# ---------------------------------------------------------------------------


@dataclass
class WordError:
    position: int
    expected: str | None
    actual: str | None
    label: str
    l1_pattern: bool
    confidence: str  # "high" | "medium" | "low"
    ref_index: int = 0  # index into the word's target_phones (for highlighting)
    tips: dict | None = None


@dataclass
class WordResult:
    word: str
    target_ipa: str
    learner_ipa: str
    errors: list[WordError]
    score: float
    target_phones: list[str] = field(default_factory=list)
    coverage: float = 0.0  # reading completeness for this word (produced or not)
    learner_start: int = 0
    learner_end: int = 0


def _label_for_sub(ref: str, learner: str) -> str:
    return L1_PATTERN_LABELS.get((ref, learner), f"sound mismatch ({ref} → {learner})")


# GOP-based scoring (when acoustic posteriors are available).
GOP_OK = 0.5        # model this confident a phone was said → suppress the error
COVERAGE_OK = 0.25  # a phone with >= this GOP counts as "produced" (read)
# Fallback penalties (no GOP): substitution uses articulatory distance.
DEL_PENALTY = 0.6
INS_PENALTY = 0.4
SUB_SCALE = 2.2


@functools.lru_cache(maxsize=1)
def _feature_table():
    import panphon
    return panphon.FeatureTable()


@functools.lru_cache(maxsize=4096)
def _sub_penalty(ref: str, learner: str) -> float:
    """Substitution penalty = panphon articulatory-feature distance × scale.

    Close (accent-like) substitutions cost little; far ones cost more.
    """
    try:
        return float(_feature_table().feature_edit_distance(ref, learner) * SUB_SCALE)
    except Exception:
        return 0.6


def analyze_words(
    ref_per_word: list[tuple[str, list[str]]],
    learner: list[str],
    gop: list[float] | None = None,
) -> list[WordResult]:
    """Classify learner phones against the reference, grouped by word.

    When *gop* (per-target-phone acoustic confidence) is given, scoring is
    GOP-based: 发音 = mean GOP, 完成度 = soft saturating coverage, and errors the
    model is confident about are suppressed. Otherwise falls back to an
    articulatory-distance penalty.
    """
    flat_ref: list[str] = []
    word_idx_per_ref: list[int] = []
    for wi, (_, phones) in enumerate(ref_per_word):
        for p in phones:
            flat_ref.append(p)
            word_idx_per_ref.append(wi)

    ops = align(flat_ref, learner)

    # Build arrays that let us map any op index to nearest reference word.
    # Leading/trailing insertions (before first ref or after last ref) are
    # treated as noise and dropped from scoring.
    ref_word_for_op: list[int | None] = []
    for op in ops:
        if op.ref_index is not None:
            ref_word_for_op.append(word_idx_per_ref[op.ref_index])
        else:
            ref_word_for_op.append(None)

    first_ref_idx = next((i for i, w in enumerate(ref_word_for_op) if w is not None), None)
    last_ref_idx = next((i for i in range(len(ref_word_for_op) - 1, -1, -1) if ref_word_for_op[i] is not None), None)

    def nearest_word(op_index: int) -> int | None:
        if first_ref_idx is None:
            return None
        if op_index < first_ref_idx or op_index > last_ref_idx:
            return None
        # Prefer the next reference word (insertions act like leading noise
        # for the upcoming word), using learner_index distance when available.
        op = ops[op_index]
        ins_idx = op.learner_index

        lo = op_index
        while lo >= 0 and ref_word_for_op[lo] is None:
            lo -= 1
        hi = op_index
        while hi < len(ops) and ref_word_for_op[hi] is None:
            hi += 1

        left_w = ref_word_for_op[lo] if lo >= 0 else None
        right_w = ref_word_for_op[hi] if hi < len(ops) else None

        if left_w is None:
            return right_w
        if right_w is None:
            return left_w

        # if we know the insertion's learner index, pick by distance
        if ins_idx is not None:
            left_learner = ops[lo].learner_index
            right_learner = ops[hi].learner_index
            left_dist = abs(ins_idx - left_learner) if left_learner is not None else float("inf")
            right_dist = abs(ins_idx - right_learner) if right_learner is not None else float("inf")
            if left_dist < right_dist:
                return left_w
            if right_dist < left_dist:
                return right_w
        # tie / no learner index: prefer the next word
        return right_w

    word_ops: dict[int, list[AlignmentOp]] = {wi: [] for wi in range(len(ref_per_word))}
    word_learner: dict[int, list[str]] = {wi: [] for wi in range(len(ref_per_word))}
    word_learner_indices: dict[int, list[int]] = {wi: [] for wi in range(len(ref_per_word))}
    for idx, op in enumerate(ops):
        wi = ref_word_for_op[idx]
        if wi is None:
            wi = nearest_word(idx)
        if wi is None:
            continue
        word_ops[wi].append(op)
        if op.learner is not None and op.learner_index is not None:
            word_learner[wi].append(op.learner)
            word_learner_indices[wi].append(op.learner_index)

    results: list[WordResult] = []
    flat_offset = 0  # this word's start index into the flat gop / target phones
    for wi, (word, target_phones) in enumerate(ref_per_word):
        ops_for_word = word_ops.get(wi, [])
        indices = word_learner_indices.get(wi, [])
        learner_start = min(indices) if indices else 0
        learner_end = max(indices) + 1 if indices else 0
        target_count = max(len(target_phones), 1)

        def _gop_ok(ri: int) -> bool:
            gi = flat_offset + ri
            return gop is not None and 0 <= gi < len(gop) and gop[gi] >= GOP_OK

        errors: list[WordError] = []
        ref_pos = 0  # index into this word's target_phones
        for op in ops_for_word:
            if op.kind == "match":
                ref_pos += 1
                continue
            if op.kind == "sub":
                ref_phone = op.ref or ""
                learner_phone = op.learner or ""
                # Accent-like reductions, or the model being confident it was
                # said, are not errors.
                if _in_pair_set(ref_phone, learner_phone, VOWEL_REDUCTION_PAIRS) or _gop_ok(ref_pos):
                    ref_pos += 1
                    continue
                label = _label_for_sub(ref_phone, learner_phone)
                l1_pattern = (ref_phone, learner_phone) in L1_PATTERN_LABELS
                errors.append(
                    WordError(
                        position=ref_pos, expected=ref_phone, actual=learner_phone,
                        label=label, l1_pattern=l1_pattern,
                        confidence="high" if l1_pattern else "medium", ref_index=ref_pos,
                    )
                )
                ref_pos += 1
            elif op.kind == "ins":
                errors.append(
                    WordError(
                        position=ref_pos, expected=None, actual=op.learner,
                        label=f"extra sound ({op.learner})", l1_pattern=False,
                        confidence="medium", ref_index=min(ref_pos, target_count - 1),
                    )
                )
                # insertion does not consume a reference phone
            elif op.kind == "del":
                if not _gop_ok(ref_pos):  # the model didn't hear it → real omission
                    errors.append(
                        WordError(
                            position=ref_pos, expected=op.ref, actual=None,
                            label=f"missing sound ({op.ref})", l1_pattern=False,
                            confidence="medium", ref_index=ref_pos,
                        )
                    )
                ref_pos += 1

        # Score (发音) + coverage (完成度).
        gop_seg = (
            gop[flat_offset : flat_offset + len(target_phones)]
            if gop is not None and flat_offset + len(target_phones) <= len(gop)
            else None
        )
        if gop_seg:
            score = sum(gop_seg) / len(gop_seg)
            # Soft, saturating coverage: full credit once GOP reaches COVERAGE_OK,
            # partial below. min(g/OK,1) >= g, so 完成度 is never below 发音.
            word_coverage = sum(min(g / COVERAGE_OK, 1.0) for g in gop_seg) / len(gop_seg)
        else:
            penalty = 0.0
            for e in errors:
                if e.expected and e.actual:
                    penalty += _sub_penalty(e.expected, e.actual)
                elif e.expected:
                    penalty += DEL_PENALTY
                else:
                    penalty += INS_PENALTY
            penalty = min(penalty, float(target_count))
            score = max(0.0, 1.0 - penalty / target_count) ** 0.7
            word_coverage = score

        flat_offset += len(target_phones)
        results.append(
            WordResult(
                word=word,
                target_ipa="".join(target_phones),
                learner_ipa="".join(word_learner.get(wi, [])),
                errors=errors[:3],  # show at most 3 errors per word in UI
                score=round(score, 2),
                target_phones=list(target_phones),
                coverage=round(word_coverage, 2),
                learner_start=learner_start,
                learner_end=learner_end,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Sentence handling
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split on sentence terminators while keeping punctuation attached."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _alignment_cut_points(
    flat_ref: list[str],
    sentence_idx_per_ref: list[int],
    learner: list[str],
    n_sentences: int,
) -> list[int]:
    """Map reference sentence boundaries to learner indices.

    Returns n_sentences + 1 cut points (starts with 0, ends with len(learner)).
    """
    cuts = [0] + [-1] * (n_sentences - 1) + [len(learner)]
    if not flat_ref:
        return cuts

    ops = align(flat_ref, learner)
    last_learner_after_ref: dict[int, int] = {}
    learner_cursor = 0
    for op in ops:
        if op.learner_index is not None:
            learner_cursor = op.learner_index + 1
        if op.ref_index is not None:
            last_learner_after_ref[op.ref_index] = learner_cursor

    for si in range(n_sentences - 1):
        last_ref_in_sentence = max(
            (i for i, s in enumerate(sentence_idx_per_ref) if s == si),
            default=None,
        )
        if last_ref_in_sentence is not None:
            cuts[si + 1] = last_learner_after_ref.get(last_ref_in_sentence, cuts[si])

    # enforce monotonic
    for i in range(1, len(cuts)):
        if cuts[i] < cuts[i - 1]:
            cuts[i] = cuts[i - 1]
    return cuts


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def _normalize_learner_tokens(tokens: Sequence[str]) -> list[str]:
    """Strip stress markers and syllable separators from model tokens.

    The wav2vec2 espeak model may emit tokens like ``o5`` (stress) or
    ``s.`` (syllable boundary) which the unstressed reference does not contain.
    """
    out: list[str] = []
    for t in tokens:
        t = t.rstrip(".0123456789")
        if t:
            out.append(t)
    return out


def analyze(
    target_text: str,
    learner_tokens: Sequence[str],
    language: str = "en-us",
    gop: list[float] | None = None,
) -> dict:
    """Analyse learner pronunciation against *target_text*.

    Parameters
    ----------
    target_text : str
        The text the learner was supposed to say.
    learner_tokens : Sequence[str]
        IPA phone tokens recognised from the learner's audio.
    language : str
        espeak language code, e.g. ``en-us``, ``fr-fr``, ``es``.

    Returns
    -------
    dict
        Overall score, per-sentence breakdown, per-word results.
    """
    learner = _normalize_learner_tokens(learner_tokens)
    sentences = split_sentences(target_text)

    if len(sentences) <= 1:
        words = analyze_words(reference_ipa_per_word(target_text, language=language), learner, gop=gop)
    else:
        per_sentence_refs = [reference_ipa_per_word(s, language=language) for s in sentences]
        flat_ref: list[str] = []
        sentence_idx_per_ref: list[int] = []
        for si, ref in enumerate(per_sentence_refs):
            for _, phones in ref:
                for p in phones:
                    flat_ref.append(p)
                    sentence_idx_per_ref.append(si)

        cuts = _alignment_cut_points(
            flat_ref, sentence_idx_per_ref, learner, len(sentences)
        )
        words: list[WordResult] = []
        for si, ref in enumerate(per_sentence_refs):
            chunk = learner[cuts[si] : cuts[si + 1]]
            words.extend(analyze_words(ref, chunk))

    total_target = sum(len(w.target_ipa) for w in words) or 1
    weighted = sum(w.score * len(w.target_ipa) for w in words)
    overall = round(weighted / total_target, 2)

    # 完成度 (reading completeness): soft, saturating coverage from GOP; kept >=
    # 发音 so the two never invert.
    if gop:
        soft = sum(min(g / COVERAGE_OK, 1.0) for g in gop) / max(len(gop), 1)
        coverage = round(max(soft, overall), 2)
    else:
        coverage = round(sum(w.coverage * len(w.target_ipa) for w in words) / total_target, 2)

    return {
        "overall_score": overall,
        "coverage": coverage,
        "sentences": sentences,
        "words": [
            {
                "word": w.word,
                "target_ipa": w.target_ipa,
                "target_phones": w.target_phones,
                "learner_ipa": w.learner_ipa,
                "score": w.score,
                "coverage": w.coverage,
                "learner_start": w.learner_start,
                "learner_end": w.learner_end,
                "errors": [asdict(e) for e in w.errors],
            }
            for w in words
        ],
    }

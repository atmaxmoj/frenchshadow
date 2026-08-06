// Group a word's IPA phones into syllables so the UI can play a *syllable*
// (e.g. "mɑ̃") instead of a bare, unnatural phone (e.g. the nasal "ɑ̃").

// French vowels (oral + nasal) plus a few fallbacks; glides j/w/ɥ act as onsets.
const VOWELS = new Set([
  "a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "ø", "œ", "ə", "u", "y",
  "ɑ̃", "ɛ̃", "ɔ̃", "œ̃", "ɐ", "æ", "ɪ", "ʊ",
]);

export function isVowel(phone: string): boolean {
  return VOWELS.has(phone);
}

// Maximal-onset heuristic: every vowel is a nucleus; preceding consonants attach
// as its onset; word-final consonants join the last syllable as coda. Good enough
// for playback (commencer /k ɔ m ɑ̃ s e/ → kɔ · mɑ̃ · se).
export function syllabify(phones: string[]): number[][] {
  const sylls: number[][] = [];
  let onset: number[] = [];
  phones.forEach((p, i) => {
    if (isVowel(p)) {
      sylls.push([...onset, i]);
      onset = [];
    } else {
      onset.push(i);
    }
  });
  if (onset.length) {
    if (sylls.length) sylls[sylls.length - 1].push(...onset);
    else sylls.push(onset);
  }
  return sylls;
}

// Indices of the syllable that contains `refIndex` (fallback: just that phone).
export function syllableIndices(phones: string[], refIndex: number): number[] {
  for (const s of syllabify(phones)) if (s.includes(refIndex)) return s;
  return refIndex >= 0 && refIndex < phones.length ? [refIndex] : [];
}

// The phones to speak for a sound in context: the syllable containing it, with
// the phone at `refIndex` optionally overridden (to voice the learner's slip).
export function syllablePhones(
  phones: string[],
  refIndex: number,
  override?: string,
): string[] {
  const idx = syllableIndices(phones, refIndex);
  if (!idx.length) return override ? [override] : [];
  return idx.map((i) => (i === refIndex && override ? override : phones[i]));
}

// The syllable containing `refIndex`, with `insert` added right after that
// phone — approximates an inserted ("extra") sound in its syllable context.
export function syllablePhonesInsert(
  phones: string[],
  refIndex: number,
  insert: string,
): string[] {
  const idx = syllableIndices(phones, refIndex);
  if (!idx.length) return [insert];
  const out: string[] = [];
  idx.forEach((i) => {
    out.push(phones[i]);
    if (i === refIndex) out.push(insert);
  });
  return out;
}

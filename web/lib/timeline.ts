// Pure timeline math — the single source of truth for "where is the playhead
// and what should happen there". No React, no DOM: unit-testable in isolation.

import type { Sentence, WordToken } from "./types";

// Effective end of a sentence: never overrun into the next one, even if the
// raw caption end times overlap.
export function sentenceEnd(sentences: Sentence[], idx: number): number {
  const s = sentences[idx];
  if (!s) return 0;
  const next = sentences[idx + 1];
  return next ? Math.min(s.end, next.start) : s.end;
}

// The sentence currently playing at `time` = the first one that has not ended
// yet. Returns -1 past the last sentence or for an empty list.
export function sentenceAt(sentences: Sentence[], time: number): number {
  for (let i = 0; i < sentences.length; i++) {
    if (time < sentences[i].end) return i;
  }
  return -1;
}

// The sentence whose span contains the playhead (start <= time < end); -1 if
// the playhead sits in a gap or past the end.
export function sentenceContaining(sentences: Sentence[], time: number): number {
  for (let i = 0; i < sentences.length; i++) {
    if (time >= sentences[i].start && time < sentences[i].end) return i;
  }
  return -1;
}

// Which sentence to visually focus (big subtitle + list highlight): follow the
// playhead while playing, otherwise stay on the sentence being practiced.
export function focusIndex(
  sentences: Sentence[],
  time: number,
  playing: boolean,
  currentIdx: number,
): number {
  if (playing) {
    const i = sentenceContaining(sentences, time);
    if (i >= 0) return i;
  }
  return currentIdx;
}

// The word being spoken at `time` within one sentence; -1 if none.
export function wordAt(words: WordToken[], time: number): number {
  for (let i = 0; i < words.length; i++) {
    if (time >= words[i].start && time < words[i].end) return i;
  }
  return -1;
}

// Should playback pause now? Returns the sentence index to stop at once the
// playhead reaches its effective end (within `eps` seconds), else -1. The caller
// dedupes so a given boundary only pauses once.
export function pauseTarget(sentences: Sentence[], time: number, eps = 0.08): number {
  const idx = sentenceAt(sentences, time);
  if (idx === -1) return -1;
  return time >= sentenceEnd(sentences, idx) - eps ? idx : -1;
}

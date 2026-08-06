import { test } from "node:test";
import assert from "node:assert/strict";
import {
  pauseTarget,
  sentenceAt,
  sentenceContaining,
  sentenceEnd,
} from "./timeline";

// Three contiguous sentences: [0,2) [2,4) [4,6).
const SENTENCES = [
  { text: "a", start: 0, end: 2, words: [] },
  { text: "b", start: 2, end: 4, words: [] },
  { text: "c", start: 4, end: 6, words: [] },
];

test("sentenceEnd never overruns into the next sentence", () => {
  const overlapping = [
    { text: "a", start: 0, end: 2.5, words: [] },
    { text: "b", start: 2, end: 4, words: [] },
  ];
  assert.equal(sentenceEnd(overlapping, 0), 2);
  assert.equal(sentenceEnd(SENTENCES, 2), 6); // last one uses its own end
});

test("pauseTarget fires only near a sentence's effective end", () => {
  assert.equal(pauseTarget(SENTENCES, 1.0), -1); // mid-sentence
  assert.equal(pauseTarget(SENTENCES, 1.95), 0); // within eps of the end
  // Exactly AT the boundary the playhead is already inside sentence 1,
  // far from ITS end — so no pause target. This is what makes resume safe.
  assert.equal(pauseTarget(SENTENCES, 2.0), -1);
});

// The semantics that make 继续 = plain resume (no seek) safe: the boundary
// watcher pauses up to ~80ms late (poll interval), so the playhead can sit
// slightly PAST the boundary it paused for. Resume without seek must not
// re-fire that same boundary, and must still fire the next one.
test("after an overshot pause, resume does not re-fire the same boundary", () => {
  const pausedFor = 0; // watcher paused for sentence 0's boundary…
  const t = 2.05; // …but the playhead overshot into sentence 1
  const idx = pauseTarget(SENTENCES, t);
  // It is NOT sentence 0's boundary anymore; nothing to dedupe against.
  assert.notEqual(idx, pausedFor);
  assert.equal(idx, -1); // mid sentence 1: keep playing
});

test("if the pause landed slightly BEFORE the boundary, dedupe blocks re-fire", () => {
  const pausedFor = 0;
  const t = 1.94; // pauseTarget still reports sentence 0 here
  const idx = pauseTarget(SENTENCES, t);
  assert.equal(idx, 0);
  // The watcher dedupes on pausedBoundaryRef === idx, so resume won't instantly
  // re-pause. Playback continues into sentence 1, which does NOT pause yet.
  assert.equal(pauseTarget(SENTENCES, 2.04), -1);
});

test("the NEXT boundary still fires after a resume", () => {
  assert.equal(pauseTarget(SENTENCES, 3.96), 1);
  assert.equal(pauseTarget(SENTENCES, 5.98), 2);
  assert.equal(sentenceAt(SENTENCES, 6.0), -1); // past the end: no more pauses
});

test("sentenceContaining distinguishes mid-sentence from gaps", () => {
  assert.equal(sentenceContaining(SENTENCES, 1.0), 0);
  const gappy = [
    { text: "a", start: 0, end: 2, words: [] },
    { text: "b", start: 3, end: 5, words: [] },
  ];
  assert.equal(sentenceContaining(gappy, 2.5), -1); // silence gap
  assert.equal(sentenceAt(gappy, 2.5), 1); // but sentenceAt points at the next one
});

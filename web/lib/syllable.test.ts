import { test } from "node:test";
import assert from "node:assert/strict";
import {
  isVowel,
  syllabify,
  syllableIndices,
  syllablePhones,
  syllablePhonesInsert,
} from "./syllable";

// commencer /k ɔ m ɑ̃ s e/ → kɔ · mɑ̃ · se
const COMMENCER = ["k", "ɔ", "m", "ɑ̃", "s", "e"];
// épisode /e p i z ɔ d/ → e · pi · zɔd
const EPISODE = ["e", "p", "i", "z", "ɔ", "d"];

test("isVowel covers French oral and nasal vowels", () => {
  assert.equal(isVowel("ɑ̃"), true);
  assert.equal(isVowel("e"), true);
  assert.equal(isVowel("z"), false);
  assert.equal(isVowel("m"), false);
});

test("syllabify groups with maximal onset and final coda", () => {
  assert.deepEqual(syllabify(COMMENCER), [[0, 1], [2, 3], [4, 5]]);
  assert.deepEqual(syllabify(EPISODE), [[0], [1, 2], [3, 4, 5]]);
});

test("syllabify handles a word with no vowels", () => {
  assert.deepEqual(syllabify(["s", "t"]), [[0, 1]]);
});

test("syllableIndices finds the containing syllable", () => {
  assert.deepEqual(syllableIndices(COMMENCER, 3), [2, 3]); // ɑ̃ → mɑ̃
  assert.deepEqual(syllableIndices(EPISODE, 3), [3, 4, 5]); // z → zɔd
  assert.deepEqual(syllableIndices(EPISODE, 0), [0]); // lone vowel syllable
});

test("syllableIndices falls back for out-of-range index", () => {
  assert.deepEqual(syllableIndices(EPISODE, 99), []);
});

test("syllablePhones returns the syllable for the target phone", () => {
  assert.deepEqual(syllablePhones(EPISODE, 3), ["z", "ɔ", "d"]);
});

test("syllablePhones with override voices the learner's slip in context", () => {
  // 应读 z，读成了 s → hear "sɔd" instead of the bare "s".
  assert.deepEqual(syllablePhones(EPISODE, 3, "s"), ["s", "ɔ", "d"]);
});

test("syllablePhones falls back to the bare phone when index is invalid", () => {
  assert.deepEqual(syllablePhones(EPISODE, 99, "s"), ["s"]);
  assert.deepEqual(syllablePhones(EPISODE, 99), []);
});

test("syllablePhonesInsert adds the extra phone after the anchor", () => {
  // Extra "s" after z in épisode → "zsɔd" context.
  assert.deepEqual(syllablePhonesInsert(EPISODE, 3, "s"), ["z", "s", "ɔ", "d"]);
});

test("syllablePhonesInsert falls back to the bare insert when index invalid", () => {
  assert.deepEqual(syllablePhonesInsert(EPISODE, 99, "s"), ["s"]);
});

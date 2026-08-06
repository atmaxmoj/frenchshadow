import { test } from "node:test";
import assert from "node:assert/strict";
import { attemptAudioUrl } from "./api";

test("attemptAudioUrl defaults to the raw recording", () => {
  assert.equal(attemptAudioUrl("abc123"), "/api/attempts/abc123/audio");
});

test("attemptAudioUrl with playback=true asks for the processed version", () => {
  assert.equal(attemptAudioUrl("abc123", true), "/api/attempts/abc123/audio?playback=1");
});

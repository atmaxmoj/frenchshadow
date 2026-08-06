import { test } from "node:test";
import assert from "node:assert/strict";
import { wordCloneButtonState } from "./PracticeView";

test("disabled with label 克隆音 before any recording", () => {
  const s = wordCloneButtonState(false, "idle");
  assert.equal(s.disabled, true);
  assert.equal(s.label, "克隆音");
});

test("enabled once a recording exists", () => {
  const s = wordCloneButtonState(true, "idle");
  assert.equal(s.disabled, false);
  assert.equal(s.label, "克隆音");
});

test("busy: disabled and shows 烤制中…", () => {
  const s = wordCloneButtonState(true, "busy");
  assert.equal(s.disabled, true);
  assert.equal(s.label, "烤制中…");
});

test("ready: enabled, replayable, label stays 克隆音", () => {
  const s = wordCloneButtonState(true, "ready");
  assert.equal(s.disabled, false);
  assert.equal(s.label, "克隆音");
});

test("error: re-clickable (retry) and shows 克隆失败", () => {
  const s = wordCloneButtonState(true, "error");
  assert.equal(s.disabled, false);
  assert.equal(s.label, "克隆失败");
});

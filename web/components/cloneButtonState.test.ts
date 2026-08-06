import { test } from "node:test";
import assert from "node:assert/strict";
import { cloneButtonState } from "./PracticeView";

test("disabled and shows '克隆标准音' before any recording", () => {
  const state = cloneButtonState(false, false, false, null);
  assert.equal(state.disabled, true);
  assert.equal(state.label, "克隆标准音");
  assert.equal(state.showKbd, false);
});

test("disabled and shows '烤制中…' while cloning", () => {
  const state = cloneButtonState(false, true, false, null);
  assert.equal(state.disabled, true);
  assert.equal(state.label, "烤制中…");
  assert.equal(state.showKbd, false);
});

test("disabled and shows '烤制中…' even if an old clone URL exists", () => {
  const state = cloneButtonState(false, true, false, "/cosy_clone_audio/abc.wav");
  assert.equal(state.disabled, true);
  assert.equal(state.label, "烤制中…");
  assert.equal(state.showKbd, false);
});

test("enabled, shows '克隆音' and the C hotkey when cloning is done", () => {
  const state = cloneButtonState(false, false, false, "/cosy_clone_audio/abc.wav");
  assert.equal(state.disabled, false);
  assert.equal(state.label, "克隆音");
  assert.equal(state.showKbd, true);
});

test("stays disabled and shows '克隆失败' on error", () => {
  const state = cloneButtonState(false, false, true, null);
  assert.equal(state.disabled, true);
  assert.equal(state.label, "克隆失败");
  assert.equal(state.showKbd, false);
});

test("stays disabled while analysis is still running", () => {
  const state = cloneButtonState(true, false, false, null);
  assert.equal(state.disabled, true);
  assert.equal(state.label, "克隆标准音");
  assert.equal(state.showKbd, false);
});

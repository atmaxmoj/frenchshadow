import { test } from "node:test";
import assert from "node:assert/strict";
import { playSequence, type SequenceAudio } from "./audio";

interface FakeAudio extends SequenceAudio {
  url: string;
  ended: () => void;
  errors: () => void;
}

function makeFactory(instances: FakeAudio[]) {
  return (url: string): FakeAudio => {
    const a: FakeAudio = {
      url,
      onended: null,
      onerror: null,
      play: () => Promise.resolve(),
      pause: () => {},
      ended() {
        this.onended?.();
      },
      errors() {
        this.onerror?.();
      },
    };
    instances.push(a);
    return a;
  };
}

test("plays every clip in order and reports progress + done", () => {
  const instances: FakeAudio[] = [];
  const progress: Array<[number, number]> = [];
  let done = false;
  playSequence(
    ["u1", "u2", "u3"],
    (i, total) => progress.push([i, total]),
    () => {
      done = true;
    },
    makeFactory(instances),
  );
  assert.deepEqual(instances.map((a) => a.url), ["u1"]);
  instances[0].ended();
  instances[1].ended();
  instances[2].ended();
  assert.deepEqual(instances.map((a) => a.url), ["u1", "u2", "u3"]);
  assert.deepEqual(progress, [
    [0, 3],
    [1, 3],
    [2, 3],
  ]);
  assert.equal(done, true);
});

test("a broken clip is skipped, not fatal", () => {
  const instances: FakeAudio[] = [];
  let done = false;
  playSequence(["u1", "u2"], undefined, () => (done = true), makeFactory(instances));
  instances[0].errors(); // u1 fails → skip to u2
  instances[1].ended();
  assert.deepEqual(instances.map((a) => a.url), ["u1", "u2"]);
  assert.equal(done, true);
});

test("a rejected play() promise is skipped too", async () => {
  const instances: FakeAudio[] = [];
  const factory = makeFactory(instances);
  let done = false;
  const stop = playSequence(
    ["u1", "u2"],
    undefined,
    () => (done = true),
    (url) => {
      const a = factory(url);
      if (url === "u1") a.play = () => Promise.reject(new Error("autoplay blocked"));
      return a;
    },
  );
  await new Promise((r) => setTimeout(r, 0)); // let the rejection propagate
  instances[1]?.ended();
  assert.deepEqual(instances.map((a) => a.url), ["u1", "u2"]);
  assert.equal(done, true);
  stop();
});

test("stop() halts the sequence and pauses the current clip", () => {
  const instances: FakeAudio[] = [];
  let paused = "";
  const factory = makeFactory(instances);
  let done = false;
  const stop = playSequence(["u1", "u2"], undefined, () => (done = true), (url) => {
    const a = factory(url);
    a.pause = () => {
      paused = url;
    };
    return a;
  });
  stop();
  instances[0].ended(); // should be ignored after stop
  assert.equal(paused, "u1");
  assert.equal(instances.length, 1);
  assert.equal(done, false);
});

import { test } from "node:test";
import assert from "node:assert/strict";

// Real import: prebakeClones must post the voice sample + upcoming sentences
// straight to the backend (the queue is consumed one cosy3 generation at a
// time server-side, so the request itself returns instantly).
import { prebakeClones } from "./api";

test("prebakeClones posts sample + items directly to the backend", async () => {
  const original = globalThis.fetch;
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return Response.json({ queued: 2 });
  };
  try {
    const blob = new Blob(["voice-sample"], { type: "audio/webm" });
    await prebakeClones(
      blob,
      [
        { target_text: "Bonjour", prompt_text: "Bonjour" },
        { target_text: "Comment ça va", prompt_text: "Comment ça va" },
      ],
      "fr-fr",
    );
    const prebakeCalls = calls.filter((c) => c.url.includes("/cosy_prebake"));
    assert.equal(prebakeCalls.length, 1);
    assert.equal(prebakeCalls[0].url, "http://localhost:8767/cosy_prebake");

    const form = prebakeCalls[0].init?.body as FormData;
    assert.ok(form.get("audio"));
    assert.equal(form.get("language"), "fr-fr");
    const items = JSON.parse(form.get("items") as string);
    assert.equal(items.length, 2);
    assert.equal(items[0].target_text, "Bonjour");
    assert.equal(items[1].target_text, "Comment ça va");
  } finally {
    globalThis.fetch = original;
  }
});

test("prebakeClones swallows network failures (fire-and-forget)", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("fetch failed");
  };
  try {
    const blob = new Blob(["x"], { type: "audio/webm" });
    await prebakeClones(blob, [{ target_text: "t", prompt_text: "t" }], "fr-fr");
  } finally {
    globalThis.fetch = original;
  }
});

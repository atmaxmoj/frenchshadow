import { test } from "node:test";
import assert from "node:assert/strict";

// Imports the REAL cloneVoice from ./api (not a copy) so the tests fail if the
// implementation drifts — e.g. if the language field stops being sent.
import { cloneVoice } from "./api";

interface CapturedCall {
  url: string;
  init?: RequestInit;
}

function stubFetch(respond: (url: string) => Response): {
  calls: CapturedCall[];
  restore: () => void;
} {
  const original = globalThis.fetch;
  const calls: CapturedCall[] = [];
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    return respond(url);
  };
  return {
    calls,
    restore: () => {
      globalThis.fetch = original;
    },
  };
}

function okResponse(url: string): Response {
  if (url.endsWith("/cosy_clone")) {
    return Response.json({ url: "/cosy_clone_audio/abc.wav" });
  }
  return Response.json({ ok: true }); // clientLogger flush etc.
}

test("cloneVoice posts directly to the backend and returns an absolute URL", async () => {
  const { calls, restore } = stubFetch(okResponse);
  try {
    const blob = new Blob(["fake-audio"], { type: "audio/webm" });
    const url = await cloneVoice(blob, "target text", "prompt text", "fr-fr");
    assert.equal(url, "http://localhost:8767/cosy_clone_audio/abc.wav");

    const cloneCalls = calls.filter((c) => c.url.includes("/cosy_clone"));
    assert.equal(cloneCalls.length, 1);
    // Must bypass the Next.js proxy (its 30s timeout kills long generations).
    assert.equal(cloneCalls[0].url, "http://localhost:8767/cosy_clone");
    assert.equal(cloneCalls[0].init?.method, "POST");

    const form = cloneCalls[0].init?.body as FormData;
    assert.ok(form instanceof FormData);
    assert.equal(form.get("target_text"), "target text");
    assert.equal(form.get("prompt_text"), "prompt text");
    assert.equal(form.get("language"), "fr-fr");
    assert.ok(form.get("audio"));
  } finally {
    restore();
  }
});

test("cloneVoice respects NEXT_PUBLIC_BACKEND_URL", async () => {
  process.env.NEXT_PUBLIC_BACKEND_URL = "http://backend.test:9999";
  const { calls, restore } = stubFetch(okResponse);
  try {
    const blob = new Blob(["x"], { type: "audio/wav" });
    const url = await cloneVoice(blob, "target", "prompt", "en-us");
    assert.equal(url, "http://backend.test:9999/cosy_clone_audio/abc.wav");
    const cloneCalls = calls.filter((c) => c.url.includes("/cosy_clone"));
    assert.equal(cloneCalls[0].url, "http://backend.test:9999/cosy_clone");
  } finally {
    restore();
    delete process.env.NEXT_PUBLIC_BACKEND_URL;
  }
});

test("cloneVoice returns null on HTTP error", async () => {
  const { restore } = stubFetch((url) =>
    url.endsWith("/cosy_clone")
      ? Response.json({ detail: "boom" }, { status: 500 })
      : Response.json({ ok: true }),
  );
  try {
    const blob = new Blob(["x"], { type: "audio/webm" });
    const url = await cloneVoice(blob, "target", "prompt", "fr-fr");
    assert.equal(url, null);
  } finally {
    restore();
  }
});

test("cloneVoice returns null on network failure", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("fetch failed");
  };
  try {
    const blob = new Blob(["x"], { type: "audio/webm" });
    const url = await cloneVoice(blob, "target", "prompt", "fr-fr");
    assert.equal(url, null);
  } finally {
    globalThis.fetch = original;
  }
});

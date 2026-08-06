import { test } from "node:test";
import assert from "node:assert/strict";

// These tests import the REAL transcribe() from ./api (not a copy) so they
// fail if someone reroutes the request back through the Next.js proxy.
//
// Why direct-to-backend matters: analysis (wav2vec2 + GOP + Whisper
// intelligibility) regularly exceeds the Next.js dev-server proxy's ~30s
// timeout. The proxy then kills the connection and fabricates a 500 while the
// backend is still working — the learner sees "分析失败" on a good recording.
import { transcribe } from "./api";

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
  if (url.endsWith("/transcribe")) {
    return Response.json({ duration_s: 1.5, model: "m", tokens: [] });
  }
  // clientLogger flush or anything else: swallow.
  return Response.json({ ok: true });
}

test("transcribe posts directly to the backend, bypassing the Next.js proxy", async () => {
  const { calls, restore } = stubFetch(okResponse);
  try {
    const blob = new Blob(["fake-audio"], { type: "audio/webm" });
    const result = await transcribe(blob, "Bonjour les amis", "fr-fr");
    assert.equal(result.duration_s, 1.5);

    const transcribeCalls = calls.filter((c) => c.url.includes("/transcribe"));
    assert.equal(transcribeCalls.length, 1);
    // Must NOT go through "/api/..." (the Next.js rewrite proxy).
    assert.equal(transcribeCalls[0].url, "http://localhost:8767/transcribe");
    assert.equal(transcribeCalls[0].init?.method, "POST");

    const form = transcribeCalls[0].init?.body as FormData;
    assert.ok(form instanceof FormData);
    assert.equal(form.get("target_text"), "Bonjour les amis");
    assert.equal(form.get("language"), "fr-fr");
    assert.ok(form.get("audio"));
  } finally {
    restore();
  }
});

test("transcribe respects NEXT_PUBLIC_BACKEND_URL", async () => {
  process.env.NEXT_PUBLIC_BACKEND_URL = "http://backend.test:9999";
  const { calls, restore } = stubFetch(okResponse);
  try {
    const blob = new Blob(["x"], { type: "audio/webm" });
    await transcribe(blob, "t", "en-us");
    const transcribeCalls = calls.filter((c) => c.url.includes("/transcribe"));
    assert.equal(transcribeCalls[0].url, "http://backend.test:9999/transcribe");
  } finally {
    restore();
    delete process.env.NEXT_PUBLIC_BACKEND_URL;
  }
});

test("transcribe throws the server detail on HTTP error", async () => {
  const { restore } = stubFetch((url) =>
    url.endsWith("/transcribe")
      ? Response.json({ detail: "录音内容为空或太短" }, { status: 400 })
      : Response.json({ ok: true }),
  );
  try {
    const blob = new Blob(["x"], { type: "audio/webm" });
    await assert.rejects(transcribe(blob, "t", "fr-fr"), /录音内容为空或太短/);
  } finally {
    restore();
  }
});

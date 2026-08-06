import type { Analysis, Transcript, VideoInfo, TranscribeResult } from "./types";
import { logClient } from "./clientLogger";

// Long-running inference endpoints (transcribe, cosy_clone) must bypass the
// Next.js rewrite proxy: the dev-server proxy times out at ~30s and fabricates
// a 500 while the backend is still working. The FastAPI backend allows CORS
// from any origin, so hitting it directly is safe.
function backendUrl(): string {
  return process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8767";
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function fetchVideoInfo(url: string, language: string): Promise<VideoInfo> {
  return getJson<VideoInfo>(
    `/api/youtube/info?url=${encodeURIComponent(url)}&language=${encodeURIComponent(language)}`,
  );
}

export function fetchTranscript(videoId: string, language: string): Promise<Transcript> {
  return getJson<Transcript>(
    `/api/youtube/transcript?video_id=${encodeURIComponent(videoId)}&language=${encodeURIComponent(language)}`,
  );
}

function audioExtension(blob: Blob): string {
  const t = blob.type || "";
  if (t.includes("ogg")) return "ogg";
  if (t.includes("webm")) return "webm";
  if (t.includes("mp4") || t.includes("m4a")) return "m4a";
  if (t.includes("wav")) return "wav";
  return "webm";
}

// Analysis (wav2vec2 + GOP + Whisper intelligibility) regularly exceeds the
// dev-server proxy's 30s timeout, so this goes straight to the backend.
export async function transcribe(
  blob: Blob,
  targetText: string,
  language: string,
): Promise<TranscribeResult> {
  const form = new FormData();
  const filename = `recording.${audioExtension(blob)}`;
  form.append("audio", blob, filename);
  form.append("target_text", targetText);
  form.append("language", language);
  logClient("info", "sending transcribe request", { filename, size: blob.size, type: blob.type });
  const res = await fetch(`${backendUrl()}/transcribe`, { method: "POST", body: form });
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
    logClient("error", "transcribe request failed", { status: res.status, detail: body.detail });
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<TranscribeResult>;
}

export function phonemeAudioUrl(ipa: string, language = "fr-fr"): string {
  return `/api/phoneme_audio?${new URLSearchParams({ ipa, language }).toString()}`;
}

export function referenceAudioUrl(text: string, language: string, sentence?: string): string {
  const p = new URLSearchParams({ text, language });
  if (sentence) p.set("sentence", sentence);
  return `/api/reference_audio?${p.toString()}`;
}

// Ask the backend to bake the whole transcript's reference audio in the
// background (one word at a time) into its shared cache. Fire-and-forget.
export function prebakeTranscript(
  items: { text: string; sentence: string }[],
  language: string,
): Promise<void> {
  return fetch("/api/prebake", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ language, items }),
  })
    .then(() => undefined)
    .catch(() => undefined);
}

// Translate sentences (default fr→zh) via the backend GLM proxy. Returns one
// translation per input (or "" on failure), order-aligned.
export async function translateSentences(
  sentences: string[],
  target = "zh",
): Promise<string[]> {
  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentences, target }),
    });
    if (!res.ok) return [];
    const data = (await res.json()) as { translations?: string[] };
    return data.translations ?? [];
  } catch {
    return [];
  }
}

// For each (word, phone), get the word with that sound's letters wrapped in 【 】.
export async function fetchGraphemeMarks(
  pairs: { word: string; phone: string }[],
  language = "fr-fr",
): Promise<string[]> {
  try {
    const res = await fetch("/api/grapheme", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pairs, language }),
    });
    if (!res.ok) return [];
    const data = (await res.json()) as { marks?: string[] };
    return data.marks ?? [];
  } catch {
    return [];
  }
}

// ---------- Persistence (SQLite-backed on the server) ----------

export interface VideoProgress {
  video_id: string;
  total_sentences: number;
  last_sentence_idx: number;
  last_practiced_at: string | null;
  attempt_count: number;
  sentence_attempt_count: number;
}

export interface AttemptSummary {
  id: string;
  sentence_idx: number;
  sentence_text: string;
  overall_score: number;
  analysis: Analysis;
  created_at: string;
}

// Persist one shadow-read attempt (recording + analysis). Advances the video's
// saved progress. Returns the new attempt id, or null on failure.
export async function saveAttempt(p: {
  blob: Blob;
  videoId: string;
  sentenceIdx: number;
  sentenceText: string;
  language: string;
  analysis: Analysis;
  durationS: number;
  title: string;
  thumbnail: string;
  totalSentences: number;
}): Promise<string | null> {
  const form = new FormData();
  form.append("audio", p.blob, `attempt.${audioExtension(p.blob)}`);
  form.append("video_id", p.videoId);
  form.append("sentence_idx", String(p.sentenceIdx));
  form.append("sentence_text", p.sentenceText);
  form.append("language", p.language);
  form.append("analysis", JSON.stringify(p.analysis));
  form.append("duration_s", String(p.durationS));
  form.append("title", p.title);
  form.append("thumbnail", p.thumbnail);
  form.append("total_sentences", String(p.totalSentences));
  try {
    const res = await fetch("/api/attempts", { method: "POST", body: form });
    if (!res.ok) return null;
    const data = (await res.json()) as { id?: string };
    return data.id ?? null;
  } catch {
    return null;
  }
}

export async function fetchProgress(videoId: string): Promise<VideoProgress | null> {
  try {
    const res = await fetch(`/api/videos/${encodeURIComponent(videoId)}/progress`);
    if (!res.ok) return null;
    return (await res.json()) as VideoProgress;
  } catch {
    return null;
  }
}

export async function fetchAttempts(
  videoId: string,
  sentenceIdx?: number,
): Promise<AttemptSummary[]> {
  try {
    const q = new URLSearchParams({ video_id: videoId });
    if (sentenceIdx != null) q.set("sentence_idx", String(sentenceIdx));
    const res = await fetch(`/api/attempts?${q.toString()}`);
    if (!res.ok) return [];
    const data = (await res.json()) as { attempts?: AttemptSummary[] };
    return data.attempts ?? [];
  } catch {
    return [];
  }
}

export function attemptAudioUrl(id: string, playback = false): string {
  // playback=true → denoised + loudness-normalized version for the learner's
  // ears (history replay); raw stays the default for anything else.
  return `/api/attempts/${encodeURIComponent(id)}/audio${playback ? "?playback=1" : ""}`;
}

// ---------- History overview (URL-as-unit) + whole-video replay ----------

export interface RecentVideo {
  video_id: string;
  title: string;
  thumbnail: string;
  language: string;
  total_sentences: number;
  last_sentence_idx: number;
  last_practiced_at: string | null;
  attempt_count: number;
  sentence_attempt_count: number;
}

export async function fetchRecentVideos(limit = 50): Promise<RecentVideo[]> {
  try {
    const res = await fetch(`/api/recent_videos?limit=${limit}`);
    if (!res.ok) return [];
    const data = (await res.json()) as { videos?: RecentVideo[] };
    return data.videos ?? [];
  } catch {
    return [];
  }
}

export interface PlaylistItem {
  sentence_idx: number;
  text: string;
  attempt_id: string;
  overall_score: number;
  created_at: string;
  clone_key: string | null;
}

// One row per practiced sentence (latest take): the learner's own recording
// plus the baked clone cache key when it exists.
export async function fetchPracticePlaylist(
  videoId: string,
  language: string,
): Promise<PlaylistItem[]> {
  try {
    const q = new URLSearchParams({ language });
    const res = await fetch(`/api/videos/${encodeURIComponent(videoId)}/practice_playlist?${q.toString()}`);
    if (!res.ok) return [];
    const data = (await res.json()) as { items?: PlaylistItem[] };
    return data.items ?? [];
  } catch {
    return [];
  }
}

export function cloneAudioUrl(cloneKey: string): string {
  return `/api/cosy_clone_audio/${encodeURIComponent(cloneKey)}.wav`;
}

// Bake the sentence-level clone for a persisted attempt (voice sample = that
// attempt's own recording). Long-running → direct to the backend.
export async function cloneAttempt(
  attemptId: string,
  targetText: string,
  language: string,
): Promise<string | null> {
  const form = new FormData();
  form.append("attempt_id", attemptId);
  form.append("target_text", targetText);
  form.append("prompt_text", targetText);
  form.append("language", language);
  try {
    const res = await fetch(`${backendUrl()}/cosy_clone`, { method: "POST", body: form });
    if (!res.ok) return null;
    const data = (await res.json()) as { url?: string };
    if (!data.url) return null;
    // Normalize to the proxied path so the history page can play it anywhere.
    return data.url.startsWith("/") ? `/api${data.url}` : data.url;
  } catch {
    return null;
  }
}

// Ask the local CosyVoice3 service to synthesize the target sentence in the
// user's voice (cloned from their attempt recording). Returns the generated WAV
// URL on success, or null if the service is unavailable / generation failed.
//
// We bypass the Next.js rewrite proxy and hit the backend directly because the
// dev-server proxy has a 30s timeout, while CosyVoice3 generation can take 30-90s.
//
// Can be called in parallel with transcribe(): it accepts the raw recording blob
// so it does not have to wait for analysis/persistence to finish.
export async function cloneVoice(
  blob: Blob,
  targetText: string,
  promptText: string,
  language: string,
): Promise<string | null> {
  const form = new FormData();
  form.append("audio", blob, `recording.${audioExtension(blob)}`);
  form.append("target_text", targetText);
  form.append("prompt_text", promptText);
  form.append("language", language);
  const base = backendUrl();
  try {
    const res = await fetch(`${base}/cosy_clone`, { method: "POST", body: form });
    if (!res.ok) {
      const body = (await res.json().catch(() => ({}))) as { detail?: string };
      logClient("error", "cloneVoice failed", { status: res.status, detail: body.detail });
      return null;
    }
    const data = (await res.json()) as { url?: string };
    const url = data.url ?? null;
    // Return an absolute URL so playback also bypasses the dev-server proxy.
    if (url && url.startsWith("/")) return `${base}${url}`;
    return url;
  } catch {
    return null;
  }
}

// Fire-and-forget: after the FIRST recording we have a voice sample, so clones
// for all upcoming sentences can be baked in the background (one at a time on
// the backend) instead of waiting for each sentence's own recording. Hits the
// backend directly; cached items are skipped server-side.
export function prebakeClones(
  blob: Blob,
  items: { target_text: string; prompt_text: string }[],
  language: string,
): Promise<void> {
  const form = new FormData();
  form.append("audio", blob, `recording.${audioExtension(blob)}`);
  form.append("items", JSON.stringify(items));
  form.append("language", language);
  return fetch(`${backendUrl()}/cosy_prebake`, { method: "POST", body: form })
    .then(() => undefined)
    .catch(() => undefined);
}

export function extractVideoId(input: string): string | null {
  const m = input.match(/(?:v=|youtu\.be\/|embed\/|shorts\/)([A-Za-z0-9_-]{11})/);
  if (m) return m[1];
  return /^[A-Za-z0-9_-]{11}$/.test(input.trim()) ? input.trim() : null;
}

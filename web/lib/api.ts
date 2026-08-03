import type { Analysis, Transcript, VideoInfo, TranscribeResult } from "./types";

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

export async function transcribe(
  blob: Blob,
  targetText: string,
  language: string,
): Promise<TranscribeResult> {
  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  form.append("target_text", targetText);
  form.append("language", language);
  const res = await fetch("/api/transcribe", { method: "POST", body: form });
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
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
  form.append("audio", p.blob, "attempt.webm");
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

export function attemptAudioUrl(id: string): string {
  return `/api/attempts/${encodeURIComponent(id)}/audio`;
}

export function extractVideoId(input: string): string | null {
  const m = input.match(/(?:v=|youtu\.be\/|embed\/|shorts\/)([A-Za-z0-9_-]{11})/);
  if (m) return m[1];
  return /^[A-Za-z0-9_-]{11}$/.test(input.trim()) ? input.trim() : null;
}

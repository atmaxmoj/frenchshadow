// Loudness-normalized playback. Mic distance makes recordings wildly uneven, so
// we decode, measure RMS, and apply a gain toward a target loudness — with a
// peak ceiling so it never clips. Used for 我的 (recording) and 标准音 (TTS) so
// they play back at a consistent, audible level.

const TARGET_RMS = 0.14; // target loudness
const PEAK_CEIL = 0.97; // never let a sample exceed this after gain
const MAX_GAIN = 12; // don't blow up near-silent clips
const MIN_RMS = 1e-4;

let sharedCtx: AudioContext | null = null;

function ctx(): AudioContext {
  if (!sharedCtx) sharedCtx = new AudioContext();
  return sharedCtx;
}

function normGain(buf: AudioBuffer, start: number, end: number): number {
  const data = buf.getChannelData(0);
  const i0 = Math.max(0, Math.floor(start * buf.sampleRate));
  const i1 = Math.min(data.length, Math.floor(end * buf.sampleRate));
  let sum = 0;
  let peak = 0;
  let n = 0;
  for (let i = i0; i < i1; i++) {
    const v = data[i];
    sum += v * v;
    const a = v < 0 ? -v : v;
    if (a > peak) peak = a;
    n++;
  }
  const r = Math.sqrt(sum / Math.max(n, 1));
  if (r < MIN_RMS) return 1; // essentially silent — leave it
  const byRms = TARGET_RMS / r;
  const byPeak = peak > 0 ? PEAK_CEIL / peak : MAX_GAIN;
  return Math.min(byRms, byPeak, MAX_GAIN);
}

async function playBuffer(buf: AudioBuffer, start: number, end: number): Promise<void> {
  const ac = ctx();
  if (ac.state === "suspended") await ac.resume().catch(() => {});
  const gain = ac.createGain();
  gain.gain.value = normGain(buf, start, end);
  const src = ac.createBufferSource();
  src.buffer = buf;
  src.connect(gain);
  gain.connect(ac.destination);
  src.start(0, Math.max(0, start), Math.max(0.03, end - start));
}

// Play a (slice of a) recorded Blob at normalized loudness.
export async function playBlobNormalized(blob: Blob, start = 0, end?: number): Promise<void> {
  const ac = ctx();
  if (ac.state === "suspended") await ac.resume().catch(() => {});
  const buf = await ac.decodeAudioData(await blob.arrayBuffer());
  await playBuffer(buf, start, end ?? buf.duration);
}

// Fetch a URL (TTS / stored attempt) and play at normalized loudness.
export async function playUrlNormalized(url: string): Promise<void> {
  const res = await fetch(url);
  if (!res.ok) return;
  await playBlobNormalized(await res.blob());
}

"use client";

/**
 * Browser-side audio utilities.
 *
 * - Loudness-normalized playback for recordings and TTS audio.
 * - WAV conversion for recorded blobs so the backend receives plain PCM instead
 *   of webm/opus or mp4 containers that some ffmpeg builds cannot decode.
 */

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

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

function floatTo16BitPCM(input: Float32Array): ArrayBuffer {
  const output = new ArrayBuffer(input.length * 2);
  const view = new DataView(output);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return output;
}

function encodeWav(pcm: ArrayBuffer, sampleRate: number): Blob {
  const wavBuffer = new ArrayBuffer(44 + pcm.byteLength);
  const view = new DataView(wavBuffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + pcm.byteLength, true);
  writeString(view, 8, "WAVE");

  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size
  view.setUint16(20, 1, true); // AudioFormat = PCM
  view.setUint16(22, 1, true); // NumChannels = mono
  view.setUint32(24, sampleRate, true); // SampleRate
  view.setUint32(28, sampleRate * 2, true); // ByteRate
  view.setUint16(32, 2, true); // BlockAlign
  view.setUint16(34, 16, true); // BitsPerSample

  writeString(view, 36, "data");
  view.setUint32(40, pcm.byteLength, true);

  new Uint8Array(wavBuffer).set(new Uint8Array(pcm), 44);
  return new Blob([wavBuffer], { type: "audio/wav" });
}

export async function blobToWav(blob: Blob, targetSampleRate = 16000): Promise<Blob> {
  const arrayBuffer = await blob.arrayBuffer();

  // Decode the original container (webm/opus, mp4, etc.) using the browser's
  // built-in decoder. This will fail only if the container itself is corrupt.
  const decodeCtx = new (window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
  try {
    const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);

    // Resample to targetSampleRate and mix down to mono via OfflineAudioContext.
    const offline = new OfflineAudioContext(
      1,
      Math.ceil(audioBuffer.duration * targetSampleRate),
      targetSampleRate,
    );
    const source = offline.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();

    const pcm = floatTo16BitPCM(rendered.getChannelData(0));
    return encodeWav(pcm, targetSampleRate);
  } finally {
    await decodeCtx.close().catch(() => {});
  }
}

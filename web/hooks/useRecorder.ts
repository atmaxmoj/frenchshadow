"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// Voice-activity auto-stop: stop after a stretch of silence once past a minimum.
const SILENCE_THRESHOLD = 0.012;
const SILENCE_MS = 1800;
const MIN_RECORDING_MS = 1000;
const MAX_RECORDING_MS = 25000;

// Not every browser supports the same container. Safari rejects audio/webm
// and only records audio/mp4. Chrome records opus inside a webm container that
// some ffmpeg builds struggle with; ogg/opus is more reliably decoded. Pick
// the first supported type and let the backend (ffmpeg) sniff whatever we send.
const MIME_CANDIDATES = [
  "audio/ogg;codecs=opus",
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
];

function pickMimeType(): string {
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return "";
  for (const t of MIME_CANDIDATES) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

export interface Recorder {
  recording: boolean;
  prime: () => Promise<void>; // warm the mic ahead of time so start() is instant
  cue: () => Promise<void>; // play a short "go" beep and resolve when it ends
  start: () => Promise<void>;
  stop: () => void;
  release: () => void; // fully release the mic (turns off the tab indicator)
  getAnalyser: () => AnalyserNode | null; // live frequency data for the visualizer
}

export function useRecorder(onComplete: (blob: Blob) => void): Recorder {
  const [recording, setRecording] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const rafRef = useRef(0);

  // Acquire the mic + audio graph once and keep them warm. Called during video
  // playback (a user gesture chain) so the context is allowed to make sound.
  const prime = useCallback(async () => {
    if (streamRef.current) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { noiseSuppression: true, echoCancellation: false, autoGainControl: false },
    });
    streamRef.current = stream;
    const ctx = new AudioContext();
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    ctxRef.current = ctx;
    analyserRef.current = analyser;
  }, []);

  // A short "speak now" beep played BEFORE recording starts (so it never lands
  // inside the recording). Resolves when the tone ends.
  const cue = useCallback(async () => {
    await prime();
    const ctx = ctxRef.current;
    if (!ctx) return;
    if (ctx.state === "suspended") {
      await ctx.resume().catch(() => {});
    }
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.frequency.value = 880;
    const t0 = ctx.currentTime;
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.exponentialRampToValueAtTime(0.12, t0 + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.14);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t0);
    osc.stop(t0 + 0.15);
    await new Promise<void>((resolve) => {
      osc.onended = () => resolve();
    });
  }, [prime]);

  const stop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    const mr = mediaRef.current;
    if (mr && mr.state !== "inactive") mr.stop();
    setRecording(false);
    // Keep the stream/context warm for the next take.
  }, []);

  const start = useCallback(async () => {
    await prime(); // instant if already primed
    const stream = streamRef.current;
    if (!stream) return;
    if (ctxRef.current?.state === "suspended") {
      await ctxRef.current.resume().catch(() => {});
    }

    const mimeType = pickMimeType();
    const mr = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    const chosenType = mr.mimeType;
    chunksRef.current = [];
    const startedAt = Date.now();
    mr.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    mr.onstop = () => {
      // MediaRecorder fires `stop` immediately after `stop()` is called, but the
      // final `dataavailable` event is queued separately. Wait one tick so the
      // last chunk (which contains the container footer) is appended before we
      // build the Blob, otherwise ffmpeg sees an incomplete file and fails with
      // "Invalid data found when processing input".
      setTimeout(() => {
        const durationMs = Date.now() - startedAt;
        if (chunksRef.current.length === 0 || durationMs < MIN_RECORDING_MS) {
          console.warn("[recorder] recording too short or empty, discarding", {
            chunks: chunksRef.current.length,
            durationMs,
          });
          setRecording(false);
          return;
        }
        onComplete(new Blob(chunksRef.current, { type: chosenType || "audio/webm" }));
      }, 0);
    };
    mr.start(100);
    mediaRef.current = mr;
    setRecording(true);

    const analyser = analyserRef.current;
    if (!analyser) return;
    const buf = new Uint8Array(analyser.fftSize);
    let silenceStart: number | null = null;
    const tick = () => {
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = (buf[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / buf.length);
      const now = Date.now();
      if (rms < SILENCE_THRESHOLD) {
        if (silenceStart === null) silenceStart = now;
        else if (now - silenceStart > SILENCE_MS && now - startedAt > MIN_RECORDING_MS) {
          stop();
          return;
        }
      } else {
        silenceStart = null;
      }
      // Hard ceiling so a noisy room never records forever.
      if (now - startedAt > MAX_RECORDING_MS) {
        stop();
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [onComplete, prime, stop]);

  const release = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    const mr = mediaRef.current;
    if (mr && mr.state !== "inactive") mr.stop();
    mediaRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    ctxRef.current?.close().catch(() => {});
    ctxRef.current = null;
    analyserRef.current = null;
    setRecording(false);
  }, []);

  const getAnalyser = useCallback(() => analyserRef.current, []);

  useEffect(() => release, [release]); // release the mic on unmount

  return { recording, prime, cue, start, stop, release, getAnalyser };
}

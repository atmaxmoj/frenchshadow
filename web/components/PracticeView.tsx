"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Play, Pause, Mic, RotateCcw, Volume2, Headphones } from "lucide-react";
import { YouTubePlayer, useVideoTime, type YTPlayerInstance } from "./YouTubePlayer";
import { MicStatus } from "./MicStatus";
import { Waveform } from "./Waveform";
import { useRecorder } from "@/hooks/useRecorder";
import {
  transcribe,
  referenceAudioUrl,
  phonemeAudioUrl,
  prebakeTranscript,
  translateSentences,
  fetchGraphemeMarks,
  saveAttempt,
  fetchAttempts,
  attemptAudioUrl,
  cloneVoice,
  prebakeClones,
  fetchPracticePlaylist,
  cloneAudioUrl,
  cloneAttempt,
  fetchRecentVideos,
  type AttemptSummary,
  type PlaylistItem,
  type RecentVideo,
} from "@/lib/api";
import { playBlobNormalized, playUrlNormalized, playSequence } from "@/lib/audio";
import { syllablePhones, syllablePhonesInsert } from "@/lib/syllable";
import { focusIndex, pauseTarget, wordAt } from "@/lib/timeline";
import { recorderErrorMessage } from "@/lib/mic";
import { errorKind, apiAsset } from "@/lib/feedback";
import type { Analysis, Sentence, Transcript, VideoInfo, WordResult, WordToken } from "@/lib/types";

function scoreClass(score: number): "good" | "warn" | "bad" {
  if (score >= 0.85) return "good";
  if (score >= 0.6) return "warn";
  return "bad";
}

// Pure helper so the clone-button state is unit-testable without rendering.
export function cloneButtonState(
  busy: boolean,
  cloneBusy: boolean,
  cloneError: boolean,
  cloneUrl: string | null,
): { disabled: boolean; label: string; showKbd: boolean } {
  return {
    disabled: busy || cloneBusy || cloneError || !cloneUrl,
    label: cloneBusy ? "烤制中…" : cloneError ? "克隆失败" : cloneUrl ? "克隆音" : "克隆标准音",
    showKbd: !cloneBusy && !busy && !cloneError && !!cloneUrl,
  };
}

export type WordCloneStatus = "idle" | "busy" | "ready" | "error";

// Pure helper so the per-word clone-button state is unit-testable without rendering.
export function wordCloneButtonState(
  hasRecording: boolean,
  status: WordCloneStatus,
): { disabled: boolean; label: string } {
  return {
    disabled: !hasRecording || status === "busy",
    label:
      status === "busy" ? "烤制中…" : status === "error" ? "克隆失败" : "克隆音",
  };
}

// Two decorrelated signals so a low number is diagnosable: 达意 = would a listener
// understand you (Whisper, accent-forgiving); 发音 = how native-like (GOP, strict).
function ScoreHeader({ analysis }: { analysis: Analysis }) {
  const intel = analysis.intelligibility ?? analysis.coverage;
  return (
    <>
      <div className="scores">
        <div className="score-metric">
          <div className={`big-score ${scoreClass(intel)}`}>{Math.round(intel * 100)}</div>
          <div className="score-label">达意<span>听得懂没</span></div>
        </div>
        <div className="score-metric">
          <div className={`big-score sub ${scoreClass(analysis.overall_score)}`}>
            {Math.round(analysis.overall_score * 100)}
          </div>
          <div className="score-label">发音<span>像不像</span></div>
        </div>
      </div>
      {analysis.heard && (
        <p className="heard-line">
          <span className="heard-label">听成</span>「{analysis.heard}」
        </p>
      )}
    </>
  );
}

function shortTime(iso: string): string {
  // sqlite "YYYY-MM-DD HH:MM:SS" (or ISO) → "MM-DD HH:MM"
  const m = iso.match(/(\d{2})-(\d{2})[ T](\d{2}:\d{2})/);
  return m ? `${m[1]}-${m[2]} ${m[3]}` : iso;
}

export function PracticeView({
  info,
  transcript,
  language,
  startIdx = 0,
  onExit,
  onLoadUrl,
}: {
  info: VideoInfo;
  transcript: Transcript;
  language: string;
  startIdx?: number;
  onExit?: () => void;
  onLoadUrl?: (url: string, idx?: number) => void;
}) {
  const [sourceUrl, setSourceUrl] = useState(`https://www.youtube.com/watch?v=${info.video_id}`);
  const sentences = transcript.sentences;
  const [player, setPlayer] = useState<YTPlayerInstance | null>(null);
  const [currentIdx, setCurrentIdx] = useState(Math.min(startIdx, sentences.length - 1));
  const [playing, setPlaying] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [status, setStatus] = useState("▶原句 或直接播放，句末自动停下让你跟读");
  const [busy, setBusy] = useState(false);
  const recordedUrlRef = useRef<string | null>(null);
  const recordedBlobRef = useRef<Blob | null>(null);
  const [hasRecording, setHasRecording] = useState(false);
  const lastAttemptIdRef = useRef<string | null>(null);
  const [cloneUrl, setCloneUrl] = useState<string | null>(null);
  const [cloneBusy, setCloneBusy] = useState(false);
  const cloneBusyRef = useRef(cloneBusy);
  useEffect(() => {
    cloneBusyRef.current = cloneBusy;
  }, [cloneBusy]);
  const [cloneError, setCloneError] = useState(false);
  const cloneControllerRef = useRef<AbortController | null>(null);

  // Word-level clone ("用我的声音读这个词的标准音"): one bake at a time,
  // cached per recording+word. The nonce invalidates the cache on every new
  // recording so a stale clone is never played against a fresh attempt.
  const recordingNonceRef = useRef(0);
  const wordCloneCacheRef = useRef(new Map<string, string>());
  const wordCloneBusyRef = useRef(false);
  const [wordClone, setWordClone] = useState<{ key: string; status: WordCloneStatus } | null>(null);
  // The first recording of the session, pinned as THE voice sample so the
  // backend clone cache stays valid across retakes.
  const firstSampleRef = useRef<Blob | null>(null);

  // The sentence being recorded, captured in a ref so the async analyze→save
  // flow attributes the attempt to the right sentence even if focus drifts.
  const recordIdxRef = useRef(currentIdx);
  useEffect(() => {
    recordIdxRef.current = currentIdx;
  }, [currentIdx]);

  // A separate history-review overlay (kept out of the practice screen).
  const [showHistory, setShowHistory] = useState(false);
  const [showKeys, setShowKeys] = useState(false);

  // Hovered word → drives the full-height 细节 detail column on the right.
  const [hoveredWordIdx, setHoveredWordIdx] = useState(-1); // -1 = none selected

  // Playback speed for shadowing (slow the native audio down).
  const [speed, setSpeed] = useState(1);
  useEffect(() => {
    player?.setPlaybackRate(speed);
  }, [player, speed]);

  const time = useVideoTime(player);
  const onReady = useCallback((p: YTPlayerInstance) => setPlayer(p), []);

  const current = sentences[currentIdx];

  // Target text for the recorder/transcribe, kept in a ref so the async
  // record→analyze flow always scores the sentence we actually paused on.
  const targetRef = useRef(current?.text ?? "");
  useEffect(() => {
    targetRef.current = sentences[currentIdx]?.text ?? "";
  }, [sentences, currentIdx]);

  // Keep the address bar in sync so a refresh restores the video + position.
  useEffect(() => {
    const p = new URLSearchParams({ v: info.video_id, lang: language, i: String(currentIdx) });
    window.history.replaceState(null, "", `?${p.toString()}`);
  }, [info.video_id, language, currentIdx]);

  // Timeline is the single axis: on restore, move the video PLAYHEAD to the
  // restored sentence's start (once, when the player is ready) so highlight and
  // playback both begin there — not at 0 while the UI points elsewhere.
  // NOTE: we deliberately do NOT seek-and-pause on load. A not-yet-played
  // YouTube iframe that's seeked+paused renders a BLACK frame (and muted
  // autoplay is often blocked before a user gesture), so we leave the thumbnail
  // showing. The timeline binds to the restored sentence on the first ▶ 原句 /
  // sentence click, which plays (and renders) from there.

  const handleRecording = useCallback(
    async (blob: Blob) => {
      if (recordedUrlRef.current) URL.revokeObjectURL(recordedUrlRef.current);
      recordedUrlRef.current = URL.createObjectURL(blob);
      recordedBlobRef.current = blob;
      setHasRecording(true);
      setCloneUrl(null);
      setCloneError(false);
      lastAttemptIdRef.current = null;
      recordingNonceRef.current += 1; // invalidate word-clone cache
      setWordClone(null);
      // Rolling pre-bake: pin the FIRST recording as the voice sample (re-using
      // it keeps the backend cache valid across takes) and, on every recording,
      // re-queue the remaining sentences from HERE. The backend caps the queue
      // (SHADOW_READER_COSY_PREBAKE_MAX) and skips already-cached items, so by
      // the time the learner reaches sentence N its clone is usually baked.
      if (!firstSampleRef.current) firstSampleRef.current = blob;
      const upcoming = sentences
        .slice(recordIdxRef.current + 1)
        .map((s) => ({ target_text: s.text, prompt_text: s.text }));
      if (upcoming.length) void prebakeClones(firstSampleRef.current, upcoming, language);
      setBusy(true);
      setStatus("分析中…");
      // Start voice cloning immediately in parallel with analysis.
      // We use the target sentence as the prompt so we don't have to wait for
      // the Whisper transcript; analysis and cloning share the same trigger.
      void cloneMine(blob, targetRef.current);
      let analysisData: Analysis | null = null;
      try {
        const res = await transcribe(blob, targetRef.current, language);
        analysisData = res.analysis ?? null;
        setAnalysis(analysisData);
        setHoveredWordIdx(-1);
        if (analysisData) {
          // Persist the attempt (recording + analysis); advances saved progress.
          const id = await saveAttempt({
            blob,
            videoId: info.video_id,
            sentenceIdx: recordIdxRef.current,
            sentenceText: targetRef.current,
            language,
            analysis: analysisData,
            durationS: res.duration_s ?? 0,
            title: info.title,
            thumbnail: info.thumbnail,
            totalSentences: sentences.length,
          });
          lastAttemptIdRef.current = id;
        }
        setStatus(cloneBusyRef.current ? "音色克隆中…" : "分析完成 · r 重读 · 空格 继续");
      } catch (err) {
        setStatus(`分析失败：${(err as Error).message}`);
      } finally {
        setBusy(false);
      }
    },
    [language, info, sentences],
  );

  const recorder = useRecorder(handleRecording);
  const recorderRef = useRef(recorder);
  useEffect(() => {
    recorderRef.current = recorder;
  }, [recorder]);

  // Beep first (a "go" cue that never lands in the recording), then start the
  // pre-warmed mic so the very first phoneme is captured cleanly.
  const startRecording = useCallback(async () => {
    try {
      setStatus("准备…");
      await recorderRef.current.cue();
      await recorderRef.current.start();
      setStatus("🔴 录音中 · 请跟读");
    } catch (e: unknown) {
      console.error("[shadow-reader] mic start failed:", e);
      const name = (e as { name?: string })?.name ?? "";
      const message = (e as { message?: string })?.message ?? "";
      setStatus(recorderErrorMessage(name, message));
    }
  }, []);

  // Continuous timeline watcher — stops at every sentence end (however playback
  // started) and hands the turn to the learner. Warms the mic as soon as any
  // playback begins so recording can start instantly at the boundary.
  const pausedBoundaryRef = useRef(-1);
  const segmentPlayingRef = useRef(false);
  const wasPlayingRef = useRef(false);
  useEffect(() => {
    if (!player) return;
    const id = setInterval(() => {
      let t: number;
      let st: number;
      try {
        t = player.getCurrentTime();
        st = player.getPlayerState();
      } catch {
        return;
      }
      const isPlaying = st === 1;
      if (isPlaying && !wasPlayingRef.current) {
        void recorderRef.current.prime().catch(() => {});
      }
      wasPlayingRef.current = isPlaying;
      setPlaying((prev) => (prev === isPlaying ? prev : isPlaying));
      if (!isPlaying || segmentPlayingRef.current) return;
      const idx = pauseTarget(sentences, t);
      if (idx >= 0 && pausedBoundaryRef.current !== idx) {
        pausedBoundaryRef.current = idx;
        player.pauseVideo();
        targetRef.current = sentences[idx].text;
        setCurrentIdx(idx);
        void startRecording();
      }
    }, 80);
    return () => clearInterval(id);
  }, [player, sentences, startRecording]);

  const playSentence = useCallback(
    (idx: number) => {
      const s = sentences[idx];
      if (!s || !player) return;
      setCurrentIdx(idx);
      setAnalysis(null);
      setHoveredWordIdx(-1);
      pausedBoundaryRef.current = -1; // re-arm: stop again at this sentence's end
      segmentPlayingRef.current = false;
      targetRef.current = s.text;
      player.seekTo(s.start, true);
      player.playVideo();
      void recorderRef.current.prime().catch(() => {}); // warm the mic during playback
      setStatus("播放中…到句末会自动停");
    },
    [player, sentences],
  );

  // Play one word's slice of the original video, pausing right at its end
  // (without letting the sentence-boundary watcher fire a recording).
  const segTimerRef = useRef<number>(0);
  const playVideoSegment = useCallback(
    (start: number, end: number) => {
      if (!player || !(end > start)) return;
      clearInterval(segTimerRef.current);
      segmentPlayingRef.current = true;
      player.seekTo(start, true);
      player.playVideo();
      segTimerRef.current = window.setInterval(() => {
        if (player.getCurrentTime() >= end - 0.03) {
          player.pauseVideo();
          clearInterval(segTimerRef.current);
          segmentPlayingRef.current = false;
        }
      }, 40);
    },
    [player],
  );

  // 原句: just replay the original sentence audio — no mic, no re-analysis.
  const playOriginal = useCallback(() => {
    const s = sentences[currentIdx];
    if (s) playVideoSegment(s.start, s.end);
  }, [sentences, currentIdx, playVideoSegment]);

  const repeat = useCallback(() => playSentence(currentIdx), [playSentence, currentIdx]);

  // 继续 = pause/play toggle: while playing it pauses; while paused it resumes
  // from EXACTLY where the playhead stopped (no seek). The boundary watcher
  // stays armed, so playback still stops at the next sentence end. Resuming
  // without a seek is safe even when the boundary pause overshot slightly —
  // see timeline.test.ts for the dedupe semantics.
  const togglePlay = useCallback(() => {
    if (!player) return;
    if (recorderRef.current.recording) return; // mic owns the turn
    let st = -1;
    try {
      st = player.getPlayerState();
    } catch {
      return;
    }
    if (st === 1) {
      player.pauseVideo();
      setStatus("已暂停 · 空格 继续");
    } else {
      segmentPlayingRef.current = false; // make sure the boundary watcher is armed
      player.playVideo();
      void recorderRef.current.prime().catch(() => {});
      setStatus("播放中…到句末会自动停");
    }
  }, [player]);

  // Reference-audio cache. Prefetch a sentence's words in the background so the
  // 标准 button plays instantly instead of baking on click ("流式烤制 + 缓存").
  const refCacheRef = useRef<Map<string, string>>(new Map());
  const refInflightRef = useRef<Set<string>>(new Set());
  const refKey = useCallback(
    (text: string, sentence: string) => `${language}|${sentence}|${text}`,
    [language],
  );
  const prefetchRef = useCallback(
    async (text: string, sentence: string): Promise<string | undefined> => {
      const key = refKey(text, sentence);
      const cached = refCacheRef.current.get(key);
      if (cached) return cached;
      if (refInflightRef.current.has(key)) return undefined;
      refInflightRef.current.add(key);
      try {
        const res = await fetch(referenceAudioUrl(text, language, sentence));
        if (!res.ok) return undefined;
        const url = URL.createObjectURL(await res.blob());
        refCacheRef.current.set(key, url);
        return url;
      } catch {
        return undefined;
      } finally {
        refInflightRef.current.delete(key);
      }
    },
    [language, refKey],
  );
  const playRef = useCallback(
    async (text: string, sentence: string) => {
      const url = refCacheRef.current.get(refKey(text, sentence)) ?? (await prefetchRef(text, sentence));
      if (url) void playUrlNormalized(url);
    },
    [refKey, prefetchRef],
  );

  // Kick the backend worker to bake the whole transcript's reference audio, in
  // order, one word at a time, into its shared cache. The UI just reads on
  // click — no client-side bake storm competing with the CPU-bound model.
  useEffect(() => {
    const items: { text: string; sentence: string }[] = [];
    sentences.forEach((s) => {
      items.push({ text: s.text, sentence: s.text }); // full sentence (natural liaison)
      s.words.forEach((w) => items.push({ text: w.text, sentence: "" })); // bare word, no liaison
    });
    if (items.length) void prebakeTranscript(items, language);
  }, [sentences, language]);

  // Fetch Chinese translations for the whole transcript in the background.
  const [translations, setTranslations] = useState<string[]>([]);
  useEffect(() => {
    let cancelled = false;
    void translateSentences(sentences.map((s) => s.text), "en").then((t) => {
      if (!cancelled) setTranslations(t);
    });
    return () => {
      cancelled = true;
    };
  }, [sentences]);

  // For each error phone, fetch the source word with the offending letter(s)
  // highlighted (via GLM, cached). Keyed "word|phone"; accumulates across takes.
  const [graphemeMarks, setGraphemeMarks] = useState<Map<string, string>>(new Map());
  useEffect(() => {
    if (!analysis) return;
    const pairs: { word: string; phone: string }[] = [];
    const seen = new Set<string>();
    analysis.words.forEach((w) =>
      w.errors.forEach((e) => {
        if (e.expected) {
          const key = `${w.word}|${e.expected}`;
          if (!seen.has(key)) {
            seen.add(key);
            pairs.push({ word: w.word, phone: e.expected });
          }
        }
      }),
    );
    if (!pairs.length) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;

    // Fetch marks; a plain (no 【】) result means GLM failed for that pair (rate
    // limit / not cached yet). Retry the still-missing ones with backoff so the
    // highlight recovers WITHOUT needing a new recording.
    const run = (todo: { word: string; phone: string }[]) => {
      void fetchGraphemeMarks(todo).then((marks) => {
        if (cancelled) return;
        const missing: { word: string; phone: string }[] = [];
        setGraphemeMarks((prev) => {
          const m = new Map(prev);
          todo.forEach((p, i) => {
            if (marks[i]?.includes("【")) m.set(`${p.word}|${p.phone}`, marks[i]);
            else missing.push(p);
          });
          return m;
        });
        if (missing.length && attempt < 5) {
          attempt += 1;
          timer = setTimeout(() => run(missing), 2000 * attempt);
        }
      });
    };
    run(pairs);

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [analysis]);

  // Revoke cached object URLs on unmount.
  useEffect(() => {
    const cache = refCacheRef.current;
    return () => cache.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  const playReference = useCallback(() => {
    void playRef(current.text, current.text);
  }, [current, playRef]);

  const playMine = useCallback(() => {
    if (recordedBlobRef.current) void playBlobNormalized(recordedBlobRef.current);
  }, []);

  const cloneMine = useCallback(
    async (blob: Blob, targetText: string) => {
      if (cloneControllerRef.current) cloneControllerRef.current.abort();
      cloneControllerRef.current = new AbortController();
      setCloneBusy(true);
      setCloneError(false);
      setStatus("音色克隆中…");
      try {
        // Use the target sentence itself as the prompt so cloning can start
        // immediately, in parallel with transcription/analysis.
        const url = await cloneVoice(blob, targetText, targetText, language);
        if (url) {
          setCloneUrl(url);
          void playUrlNormalized(url);
          setStatus("克隆完成 · 正在播放");
        } else {
          setCloneError(true);
          setStatus("克隆失败：CosyVoice3 不可用");
        }
      } catch (err) {
        setCloneError(true);
        setStatus(`克隆失败：${(err as Error).message}`);
      } finally {
        setCloneBusy(false);
      }
    },
    [language],
  );

  const playClone = useCallback(() => {
    if (cloneUrl) void playUrlNormalized(cloneUrl);
  }, [cloneUrl]);

  // Map each word's text to its position in the original video for per-word playback.
  const wordTimes = useMemo(() => {
    const m = new Map<string, WordToken>();
    current?.words.forEach((w) => {
      const k = w.text.toLowerCase();
      if (!m.has(k)) m.set(k, w);
    });
    return m;
  }, [current]);

  const onPlayWordVideo = useCallback(
    (w: WordResult) => {
      const wt = wordTimes.get(w.word.toLowerCase());
      if (wt) playVideoSegment(wt.start, wt.end);
    },
    [wordTimes, playVideoSegment],
  );
  const onPlayWordRef = useCallback(
    (w: WordResult) => {
      if (current) void playRef(w.word, ""); // bare word, no liaison expansion
    },
    [current, playRef],
  );
  const onPlayWordMine = useCallback((w: WordResult) => {
    const blob = recordedBlobRef.current;
    if (blob && w.start_time != null && w.end_time != null) {
      void playBlobNormalized(blob, w.start_time, w.end_time);
    }
  }, []);

  // Word-level clone: synthesize just this word in the user's voice. The
  // prompt is the sentence they actually read (it describes the reference
  // audio); the backend caches per (recording, target, prompt).
  const onCloneWord = useCallback(
    (w: WordResult) => {
      const blob = recordedBlobRef.current;
      const promptText = sentences[recordIdxRef.current]?.text;
      if (!blob || !promptText) return;
      const key = `${recordingNonceRef.current}|${w.word.toLowerCase()}`;
      const cached = wordCloneCacheRef.current.get(key);
      if (cached) {
        void playUrlNormalized(cached);
        return;
      }
      if (wordCloneBusyRef.current) return; // one bake at a time
      wordCloneBusyRef.current = true;
      setWordClone({ key, status: "busy" });
      cloneVoice(blob, w.word, promptText, language)
        .then((url) => {
          if (url) {
            wordCloneCacheRef.current.set(key, url);
            setWordClone({ key, status: "ready" });
            void playUrlNormalized(url);
          } else {
            setWordClone({ key, status: "error" });
          }
        })
        .catch(() => setWordClone({ key, status: "error" }))
        .finally(() => {
          wordCloneBusyRef.current = false;
        });
    },
    [sentences, language],
  );

  const wordCloneStatus = useCallback(
    (w: WordResult): WordCloneStatus => {
      const key = `${recordingNonceRef.current}|${w.word.toLowerCase()}`;
      if (wordClone?.key === key) return wordClone.status;
      if (wordCloneCacheRef.current.has(key)) return "ready";
      return "idle";
    },
    [wordClone],
  );

  // Hotkeys. Sentence: A 原句 · E 跟读 · R 重读 · Space 继续 · S 标准音 · D 我的 · C 克隆音.
  // Word nav: ← / → move the selected word (none → last / first).
  // Selected word tracks: O 原声 · B 标准 · M 我的.
  // Selected word's Nth error (top-down): Ctrl+N 应读 · Ctrl+N while holding X 我读的.
  useEffect(() => {
    const held = new Set<string>();
    const wordAt = (i: number) =>
      analysis && i >= 0 && i < analysis.words.length ? analysis.words[i] : undefined;

    const onKey = (e: KeyboardEvent) => {
      const tag = (document.activeElement as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;

      // Ctrl+digit → play the SYLLABLE of the selected word's Nth error (from top).
      if (e.ctrlKey && !e.metaKey && !e.altKey && /^[1-9]$/.test(e.key)) {
        const w = wordAt(hoveredWordIdx);
        const err = w?.errors[parseInt(e.key, 10) - 1];
        if (!w || !err) return;
        // X → 你读的 (slip voiced in the syllable), else 应读 (target syllable).
        const play = held.has("x")
          ? err.actual
            ? err.expected
              ? syllablePhones(w.target_phones, err.ref_index, err.actual)
              : syllablePhonesInsert(w.target_phones, err.ref_index, err.actual)
            : []
          : err.expected
            ? syllablePhones(w.target_phones, err.ref_index)
            : [];
        if (play.length) {
          e.preventDefault();
          new Audio(phonemeAudioUrl(play.join(" "))).play().catch(() => {});
        }
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      // ← / → : move the selected word (none → last / first).
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        if (!analysis || !analysis.words.length) return;
        e.preventDefault();
        const dir = e.key === "ArrowRight" ? 1 : -1;
        const n = analysis.words.length;
        setHoveredWordIdx((cur) =>
          cur < 0 ? (dir > 0 ? 0 : n - 1) : Math.min(n - 1, Math.max(0, cur + dir)),
        );
        return;
      }

      const k = e.key.toLowerCase();
      if (k === "a") playOriginal();
      else if (k === "e") { if (!recorder.recording) void startRecording(); }
      else if (k === "s") playReference();
      else if (k === "d") playMine();
      else if (k === "c") { if (cloneUrl && !cloneBusy) playClone(); }
      else if (k === "r") repeat();
      else if (k === "o") { const w = wordAt(hoveredWordIdx); if (w) onPlayWordVideo(w); }
      else if (k === "b") { const w = wordAt(hoveredWordIdx); if (w) onPlayWordRef(w); }
      else if (k === "m") { const w = wordAt(hoveredWordIdx); if (w) onPlayWordMine(w); }
      else if (e.key === " " || e.code === "Space") {
        e.preventDefault(); // stop the page from scrolling
        togglePlay();
      }
    };

    const onDown = (e: KeyboardEvent) => {
      const kk = e.key.toLowerCase();
      if (kk === "z" || kk === "x") held.add(kk);
      onKey(e);
    };
    const onUp = (e: KeyboardEvent) => held.delete(e.key.toLowerCase());
    const onBlur = () => held.clear();
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
      window.removeEventListener("blur", onBlur);
    };
  }, [repeat, togglePlay, playOriginal, playSentence, currentIdx, startRecording, playReference, playMine, playClone,
      recorder.recording, analysis, hoveredWordIdx, onPlayWordVideo, onPlayWordRef, onPlayWordMine, cloneUrl, cloneBusy]);

  const focus = focusIndex(sentences, time, playing, currentIdx);
  const focusSentence = sentences[focus] ?? current;

  const wordResults = new Map<string, WordResult>();
  analysis?.words.forEach((w) => wordResults.set(w.word.toLowerCase(), w));
  const showScores = focus === currentIdx && !playing;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <span className="brand-mark">Lucerne</span>
            <span className="brand-sub">shadow reader</span>
          </div>
          <nav className="nav">
            <button className={`nav-item${!showHistory ? " active" : ""}`} onClick={() => setShowHistory(false)}>练习</button>
            <button className={`nav-item${showHistory ? " active" : ""}`} onClick={() => setShowHistory(true)}>历史</button>
          </nav>
          <div className="topbar-right">
            <div className="keys-menu">
              <button
                className={`nav-item${showKeys ? " active" : ""}`}
                onClick={() => setShowKeys((v) => !v)}
              >
                ⌨ 热键
              </button>
              {showKeys && <HotkeyLegend onClose={() => setShowKeys(false)} />}
            </div>
            {onExit && <button className="nav-item nav-exit" onClick={onExit}>换视频</button>}
          </div>
        </div>
      </header>
      <div className="app-body">
        <div className="practice-layout">
          <aside className="panel transcript-panel">
            <h3>全文</h3>
            <div className="transcript-list">
              {sentences.map((s, i) => (
                <span
                  key={i}
                  className={`sentence-item${i === focus ? " current" : ""}`}
                  onClick={() => playSentence(i)}
                >
                  <span className="idx">{i + 1}</span>
                  {s.text}
                </span>
              ))}
            </div>
          </aside>

          <div className="middle-stack">
            <div className="top-area">
              <div className="ctl-top">
                <input
                  className="source-input"
                  value={sourceUrl}
                  onChange={(e) => setSourceUrl(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && sourceUrl.trim()) onLoadUrl?.(sourceUrl.trim());
                  }}
                  placeholder="粘贴 YouTube 链接…"
                  spellCheck={false}
                />
                <button
                  className="btn-primary source-load"
                  onClick={() => sourceUrl.trim() && onLoadUrl?.(sourceUrl.trim())}
                >
                  load
                </button>
                <span className="ctl-divider" />
                <span className="speed-label">速度</span>
                {[0.5, 0.75, 1].map((r) => (
                  <button
                    key={r}
                    className={`speed-btn${speed === r ? " active" : ""}`}
                    onClick={() => setSpeed(r)}
                  >
                    {r}×
                  </button>
                ))}
              </div>

              <div className="video-row">
                <div className="video-box">
                  <YouTubePlayer videoId={info.video_id} onReady={onReady} />
                </div>
                <div className="controls-vert">
                  <button className="ctl-btn" onClick={playOriginal}>
                    <Play size={16} /><span>原句</span><kbd>A</kbd>
                  </button>
                  <button className="ctl-btn primary" onClick={() => void startRecording()} disabled={recorder.recording}>
                    <Mic size={16} /><span>跟读</span><kbd>E</kbd>
                  </button>
                  <button className="ctl-btn" onClick={repeat}>
                    <RotateCcw size={16} /><span>重读</span><kbd>R</kbd>
                  </button>
                  <button className="ctl-btn" onClick={togglePlay} disabled={recorder.recording}>
                    {playing ? <Pause size={16} /> : <Play size={16} />}
                    <span>{playing ? "暂停" : "继续"}</span><kbd>空格</kbd>
                  </button>
                  <div className="ctl-gap" />
                  <button className="ctl-btn" onClick={playReference}>
                    <Volume2 size={16} /><span>标准音</span><kbd>S</kbd>
                  </button>
                  <button className="ctl-btn" onClick={playMine} disabled={!hasRecording}>
                    <Headphones size={16} /><span>我的</span><kbd>D</kbd>
                  </button>
                  {(() => {
                    const state = cloneButtonState(busy, cloneBusy, cloneError, cloneUrl);
                    return (
                      <button className="ctl-btn" onClick={playClone} disabled={state.disabled}>
                        <Play size={16} />
                        <span>{state.label}</span>
                        {state.showKbd && <kbd>C</kbd>}
                      </button>
                    );
                  })()}
                </div>
              </div>

              <div className="mid-row">
                <MicStatus recording={recorder.recording} />
                <div className="status">{busy ? "分析中…" : status}</div>
                <Waveform recording={recorder.recording} getAnalyser={recorder.getAnalyser} />
              </div>

              <Subtitle
                sentence={focusSentence}
                time={time}
                playing={playing}
                recording={recorder.recording}
                wordResults={showScores ? wordResults : new Map()}
              />
              {translations.length === sentences.length && translations[focus] && (
                <div className="translation">{translations[focus]}</div>
              )}
            </div>

            <AnalysisCards
              analysis={analysis}
              busy={busy}
              recording={recorder.recording}
              hoveredIdx={hoveredWordIdx}
              onHover={setHoveredWordIdx}
            />
          </div>

          <DetailColumn
            analysis={analysis}
            hoveredIdx={hoveredWordIdx}
            graphemeMarks={graphemeMarks}
            hasRecording={hasRecording}
            onPlayVideo={onPlayWordVideo}
            onPlayRef={onPlayWordRef}
            onPlayMine={onPlayWordMine}
            onCloneWord={onCloneWord}
            wordCloneStatus={wordCloneStatus}
          />
        </div>
        {showHistory && (
          <HistoryOverlay
            currentVideoId={info.video_id}
            sentences={sentences}
            language={language}
            onClose={() => setShowHistory(false)}
            onJump={(idx) => {
              setShowHistory(false);
              playSentence(idx);
            }}
            onOpenVideo={
              onLoadUrl
                ? (vid, idx) => {
                    setShowHistory(false);
                    onLoadUrl(`https://www.youtube.com/watch?v=${vid}`, idx);
                  }
                : undefined
            }
          />
        )}
      </div>
    </div>
  );
}

function Subtitle({
  sentence,
  time,
  playing,
  recording,
  wordResults,
}: {
  sentence: Sentence;
  time: number;
  playing: boolean;
  recording: boolean;
  wordResults: Map<string, WordResult>;
}) {
  const activeWord = playing ? wordAt(sentence.words, time) : -1;
  return (
    <div className={`subtitle-box${recording ? " recording" : ""}`}>
      {sentence.words.map((w, i) => {
        const res = wordResults.get(w.text.toLowerCase());
        const scoreCls = res ? scoreClass(res.score) : "";
        return (
          <span key={i} className={`subtitle-word ${scoreCls}${i === activeWord ? " active" : ""}`}>
            {w.text}{" "}
          </span>
        );
      })}
    </div>
  );
}

// Middle-column 分析 panel: overall scores + a grid of compact word cards.
// Hovering a card sets the shared hovered word (drives the 细节 column).
function AnalysisCards({
  analysis,
  busy,
  recording,
  hoveredIdx,
  onHover,
}: {
  analysis: Analysis | null;
  busy: boolean;
  recording: boolean;
  hoveredIdx: number;
  onHover: (i: number) => void;
}) {
  let body: ReactNode;
  if (recording && !analysis) {
    body = <div className="analysis-status"><span className="rec-dot" />正在听你读…读完自动分析</div>;
  } else if (busy && !analysis) {
    body = <div className="analysis-status"><span className="spinner" />分析中…</div>;
  } else if (!analysis) {
    body = <p className="analysis-empty">跟读后会显示逐词分析</p>;
  } else {
    const hi = Math.min(hoveredIdx, analysis.words.length - 1);
    body = (
      <div className="panel-scroll">
        <ScoreHeader analysis={analysis} />
        <div className="word-cards">
          {analysis.words.map((w, i) => (
            <div
              key={i}
              className={`word-card${i === hi ? " active" : ""}`}
              onMouseEnter={() => onHover(i)}
            >
              <div className="word-result-header">
                <span className="word-result-word">{w.word}</span>
                <span className="word-result-ipa">/{w.target_ipa}/</span>
                <span className="word-scores" title="达意 · 发音">
                  <span className={`ws ${scoreClass(w.intelligibility ?? w.coverage)}`}>{Math.round((w.intelligibility ?? w.coverage) * 100)}</span>
                  <span className={`ws sub ${scoreClass(w.score)}`}>{Math.round(w.score * 100)}</span>
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return (
    <aside className="panel analysis-left">
      <h3>分析</h3>
      {body}
    </aside>
  );
}

// Full-height right column: the hovered word's complete breakdown, seen at a glance.
function DetailColumn({
  analysis,
  hoveredIdx,
  graphemeMarks,
  hasRecording,
  onPlayVideo,
  onPlayRef,
  onPlayMine,
  onCloneWord,
  wordCloneStatus,
}: {
  analysis: Analysis | null;
  hoveredIdx: number;
  graphemeMarks: Map<string, string>;
  hasRecording: boolean;
  onPlayVideo: (w: WordResult) => void;
  onPlayRef: (w: WordResult) => void;
  onPlayMine: (w: WordResult) => void;
  onCloneWord: (w: WordResult) => void;
  wordCloneStatus: (w: WordResult) => WordCloneStatus;
}) {
  const w = analysis ? analysis.words[Math.min(hoveredIdx, analysis.words.length - 1)] : undefined;
  return (
    <aside className="panel detail-panel">
      <div className="detail-title-row">
        <h3>细节{w ? ` · ${w.word}` : ""}</h3>
        {w && (
          <span className="detail-scores">
            <span>达意：<b className={scoreClass(w.intelligibility ?? w.coverage)}>{Math.round((w.intelligibility ?? w.coverage) * 100)}</b></span>
            <span>发音：<b className={scoreClass(w.score)}>{Math.round(w.score * 100)}</b></span>
          </span>
        )}
      </div>
      <div className="panel-scroll">
        {w ? (
          <WordDetail
            w={w}
            graphemeMarks={graphemeMarks}
            hasRecording={hasRecording}
            onPlayVideo={onPlayVideo}
            onPlayRef={onPlayRef}
            onPlayMine={onPlayMine}
            onCloneWord={onCloneWord}
            cloneStatus={wordCloneStatus(w)}
          />
        ) : (
          <p className="analysis-empty">跟读后，把鼠标放到左边的词上看细节</p>
        )}
      </div>
    </aside>
  );
}

// Two-level practice history, URL-as-unit: level 1 lists every practiced
// video; level 2 drills into one video's per-sentence takes. Opened from the
// landing page (no current video) or from practice (current video badged).
export function HistoryOverlay({
  currentVideoId,
  sentences = [],
  language,
  initialVideoId,
  closeLabel = "← 返回练习",
  onClose,
  onJump,
  onOpenVideo,
}: {
  currentVideoId?: string; // the video being practiced (badge + 重录 jump)
  sentences?: Sentence[]; // transcript of the CURRENT practice video (may be [])
  language: string;
  initialVideoId?: string; // open straight in this video's detail
  closeLabel?: string;
  onClose: () => void;
  onJump?: (idx: number) => void; // jump within the CURRENT practice video
  onOpenVideo?: (videoId: string, idx: number) => void; // open another video at idx
}) {
  const [videos, setVideos] = useState<RecentVideo[] | null>(null);
  const [detailId, setDetailId] = useState<string | null>(initialVideoId ?? null);

  useEffect(() => {
    let cancelled = false;
    void fetchRecentVideos().then((v) => {
      if (!cancelled) setVideos(v);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const detailVideo = detailId
    ? {
        video_id: detailId,
        language: videos?.find((v) => v.video_id === detailId)?.language ?? language,
      }
    : null;

  return (
    <div className="history-overlay">
      <div className="history-topbar">
        <h2>练习历史</h2>
        {detailVideo && (
          <button className="btn-secondary" onClick={() => setDetailId(null)}>← 所有视频</button>
        )}
        <button className="btn-secondary" onClick={onClose}>{closeLabel}</button>
      </div>
      {detailVideo ? (
        <VideoHistoryDetail
          key={detailVideo.video_id}
          videoId={detailVideo.video_id}
          language={detailVideo.language}
          sentences={detailVideo.video_id === currentVideoId ? sentences : []}
          jumpLabel={detailVideo.video_id === currentVideoId ? "重录" : "去这个视频练"}
          onJump={(idx) => {
            if (detailVideo.video_id === currentVideoId) onJump?.(idx);
            else onOpenVideo?.(detailVideo.video_id, idx);
          }}
        />
      ) : (
        <div className="panel-scroll">
          {(videos ?? []).length === 0 ? (
            <p className="analysis-empty">
              {videos === null ? "加载中…" : "还没有练习记录。回去跟读几句，这里就会有历史。"}
            </p>
          ) : (
            <div className="recent-list">
              {(videos ?? []).map((v) => (
                <div key={v.video_id} className="recent-row">
                  {v.thumbnail && (
                    // eslint-disable-next-line @next/next/no-img-element -- YouTube thumbnail, next/image is unsuitable
                    <img className="recent-thumb" src={v.thumbnail} alt="" />
                  )}
                  <div className="recent-meta">
                    <div className="recent-title">
                      {v.title || v.video_id}
                      {v.video_id === currentVideoId && <span className="recent-current">（当前）</span>}
                    </div>
                    <div className="recent-sub">
                      {v.sentence_attempt_count}/{v.total_sentences} 句 · {v.attempt_count} 次跟读
                    </div>
                  </div>
                  <button className="track-btn" onClick={() => setDetailId(v.video_id)}>查看</button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// One video's history: takes grouped by sentence, whole-video replay of the
// learner's takes or their baked clones (missing ones can be baked on demand),
// and a read-only per-word breakdown of any selected take.
function VideoHistoryDetail({
  videoId,
  sentences,
  language,
  jumpLabel,
  onJump,
}: {
  videoId: string;
  sentences: Sentence[];
  language: string;
  jumpLabel: string;
  onJump: (idx: number) => void;
}) {
  const [attempts, setAttempts] = useState<AttemptSummary[]>([]);
  const [selected, setSelected] = useState<AttemptSummary | null>(null);
  const [playlist, setPlaylist] = useState<PlaylistItem[] | null>(null);
  const [seq, setSeq] = useState<{ kind: "mine" | "clone"; i: number; total: number } | null>(null);
  const [bakeBusy, setBakeBusy] = useState(false);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetchAttempts(videoId).then((a) => {
      if (!cancelled) setAttempts(a);
    });
    void fetchPracticePlaylist(videoId, language).then((items) => {
      if (!cancelled) setPlaylist(items);
    });
    return () => {
      cancelled = true;
    };
  }, [videoId, language]);

  // Never leave a sequence playing after the detail unmounts.
  useEffect(() => () => stopRef.current?.(), []);

  const stopSeq = useCallback(() => {
    stopRef.current?.();
    stopRef.current = null;
    setSeq(null);
  }, []);

  const startSeq = useCallback(
    (kind: "mine" | "clone") => {
      if (!playlist) return;
      stopSeq();
      const urls = playlist
        .map((it) =>
          kind === "mine"
            ? attemptAudioUrl(it.attempt_id, true) // processed for pleasant replay
            : it.clone_key
              ? cloneAudioUrl(it.clone_key)
              : null,
        )
        .filter((u): u is string => !!u);
      if (!urls.length) return;
      setSeq({ kind, i: 0, total: urls.length });
      stopRef.current = playSequence(
        urls,
        (i, total) => setSeq({ kind, i, total }),
        () => setSeq(null),
      );
    },
    [playlist, stopSeq],
  );

  const cloneCount = playlist ? playlist.filter((it) => it.clone_key).length : 0;

  // Bake the sentence clones that are missing, one at a time, then refresh.
  const bakeMissing = useCallback(async () => {
    if (!playlist) return;
    setBakeBusy(true);
    try {
      for (const it of playlist) {
        if (!it.clone_key) await cloneAttempt(it.attempt_id, it.text, language);
      }
      setPlaylist(await fetchPracticePlaylist(videoId, language));
    } finally {
      setBakeBusy(false);
    }
  }, [playlist, videoId, language]);

  const bySentence = new Map<number, AttemptSummary[]>();
  attempts.forEach((a) => {
    const arr = bySentence.get(a.sentence_idx) ?? [];
    arr.push(a);
    bySentence.set(a.sentence_idx, arr);
  });
  const idxs = Array.from(bySentence.keys()).sort((a, b) => a - b);

  if (attempts.length === 0) {
    return <p className="analysis-empty">这个视频还没有练习记录。</p>;
  }

  return (
    <>
      {playlist && playlist.length > 0 && (
        <div className="history-replay">
          <button className="track-btn" onClick={() => (seq?.kind === "mine" ? stopSeq() : startSeq("mine"))}>
            {seq?.kind === "mine" ? `⏹ 停止 ${seq.i + 1}/${seq.total}` : "🎧 连播我的朗读"}
          </button>
          <button
            className="track-btn"
            onClick={() => (seq?.kind === "clone" ? stopSeq() : startSeq("clone"))}
            disabled={cloneCount === 0 && seq?.kind !== "clone"}
          >
            {seq?.kind === "clone" ? `⏹ 停止 ${seq.i + 1}/${seq.total}` : `🪄 连播克隆音 ${cloneCount}/${playlist.length}`}
          </button>
          {cloneCount < playlist.length && (
            <button className="track-btn" onClick={() => void bakeMissing()} disabled={bakeBusy}>
              {bakeBusy ? "补烤中…" : `补烤缺失克隆 (${playlist.length - cloneCount})`}
            </button>
          )}
        </div>
      )}
      <div className="history-body">
        <div className="history-list">
          {idxs.map((idx) => {
            const takes = bySentence.get(idx) as AttemptSummary[];
            const best = Math.max(...takes.map((t) => t.overall_score));
            return (
              <div key={idx} className="history-sentence">
                <div className="history-sentence-head">
                  <span className="idx">{idx + 1}</span>
                  <span className="hs-text">{sentences[idx]?.text ?? takes[0].sentence_text}</span>
                  <span className={`sent-score ${scoreClass(best)}`}>{Math.round(best * 100)}</span>
                  <span className="sent-count">·{takes.length}次</span>
                  <button className="track-btn" onClick={() => onJump(idx)}>{jumpLabel}</button>
                </div>
                {takes.map((t) => (
                  <div key={t.id} className={`history-row${selected?.id === t.id ? " sel" : ""}`}>
                    <span className={`sent-score ${scoreClass(t.overall_score)}`}>
                      {Math.round(t.overall_score * 100)}
                    </span>
                    <span className="history-time">{shortTime(t.created_at)}</span>
                    <button className="track-btn" onClick={() => setSelected(t)}>查看逐词</button>
                    <button
                      className="track-btn"
                      onClick={() => void playUrlNormalized(attemptAudioUrl(t.id, true))}
                    >
                      🎧 重听
                    </button>
                  </div>
                ))}
              </div>
            );
          })}
        </div>
        <div className="history-detail panel-scroll">
          {selected ? (
            <AnalysisReadonly analysis={selected.analysis} />
          ) : (
            <p className="analysis-empty">点某次的「查看逐词」，看当时每个词读成什么样。</p>
          )}
        </div>
      </div>
    </>
  );
}

// Read-only per-word breakdown for a saved take (no audio/grapheme fetching).
function AnalysisReadonly({ analysis }: { analysis: Analysis }) {
  return (
    <div>
      <ScoreHeader analysis={analysis} />
      <div className="analysis-words">
        {analysis.words.map((w, i) => (
        <div key={i} className="word-result">
          <div className="word-result-header">
            <span className="word-result-word">{w.word}</span>
            <span className="word-result-ipa">/{w.target_ipa}/</span>
            <span className="word-scores" title="达意 · 发音">
              <span className={`ws ${scoreClass(w.intelligibility ?? w.coverage)}`}>{Math.round((w.intelligibility ?? w.coverage) * 100)}</span>
              <span className={`ws sub ${scoreClass(w.score)}`}>{Math.round(w.score * 100)}</span>
            </span>
          </div>
          {w.errors.slice(0, 3).map((e, j) => (
            <div key={j} className="tip-box">
              <div className="error-headline">
                <ErrorHeadline expected={e.expected} actual={e.actual} label={e.label} row={j} phones={w.target_phones} refIndex={e.ref_index} />
              </div>
              {(e.tips?.diagram_expected || e.tips?.diagram_actual) && (
                <div className="mouth-diagrams">
                  <DiagramImg src={apiAsset(e.tips?.diagram_expected ?? null)} label="正确" kind="good" />
                  <DiagramImg src={apiAsset(e.tips?.diagram_actual ?? null)} label="你的" kind="bad" />
                </div>
              )}
              {e.tips?.tongue && <p><span className="label">舌头：</span>{e.tips.tongue}</p>}
              {e.tips?.lips && <p><span className="label">嘴唇：</span>{e.tips.lips}</p>}
              {e.tips?.jaw && <p><span className="label">下巴：</span>{e.tips.jaw}</p>}
            </div>
          ))}
        </div>
        ))}
      </div>
    </div>
  );
}

// The source word with the erroring sound's letter(s) — GLM wraps them in
// 【 】 — rendered highlighted so you see exactly which part to fix.
function MarkedWord({ marked }: { marked: string }) {
  const parts = marked.split(/[【】]/);
  return (
    <div className="marked-word">
      {parts.map((p, i) => (i % 2 === 1 ? <mark key={i}>{p}</mark> : <span key={i}>{p}</span>))}
    </div>
  );
}

// Detail for the hovered word (right pane). Every word shows something: a
// per-error breakdown (headline + spelling highlight + mouth diagrams + tips),
// or a "read well" note when there's nothing to fix.
function WordDetail({
  w,
  graphemeMarks,
  hasRecording,
  onPlayVideo,
  onPlayRef,
  onPlayMine,
  onCloneWord,
  cloneStatus,
}: {
  w: WordResult;
  graphemeMarks: Map<string, string>;
  hasRecording: boolean;
  onPlayVideo: (w: WordResult) => void;
  onPlayRef: (w: WordResult) => void;
  onPlayMine: (w: WordResult) => void;
  onCloneWord: (w: WordResult) => void;
  cloneStatus: WordCloneStatus;
}) {
  const cloneBtn = wordCloneButtonState(hasRecording, cloneStatus);
  return (
    <div className="word-detail">
      <div className="word-result-header detail-head">
        <span className="word-result-word">{w.word}</span>
        <span className="word-result-ipa">/{w.target_ipa}/</span>
        <span className="detail-tracks">
          <button className="track-btn" onClick={() => onPlayVideo(w)}>▶ 原声 <kbd>O</kbd></button>
          <button className="track-btn" onClick={() => onPlayRef(w)}>🔊 标准 <kbd>B</kbd></button>
          <button className="track-btn" onClick={() => onPlayMine(w)} disabled={!hasRecording || w.start_time == null}>🎧 我的 <kbd>M</kbd></button>
          <button
            className="track-btn"
            onClick={() => onCloneWord(w)}
            disabled={cloneBtn.disabled}
            title="用我的声音读这个词的标准音（CosyVoice3）"
          >
            🪄 {cloneBtn.label}
          </button>
        </span>
      </div>
      {w.errors.length === 0 ? (
        <p className="detail-ok">✓ 这个词读得挺标准，没有明显问题。</p>
      ) : (
        w.errors.slice(0, 3).map((e, j) => (
          <div key={j} className="tip-box">
            <div className="error-headline">
              <ErrorHeadline expected={e.expected} actual={e.actual} label={e.label} phones={w.target_phones} refIndex={e.ref_index} />
            </div>
            {e.expected && (
              <MarkedWord marked={graphemeMarks.get(`${w.word}|${e.expected}`) ?? w.word} />
            )}
            {(e.tips?.diagram_expected || e.tips?.diagram_actual) && (
              <div className="mouth-diagrams">
                <DiagramImg src={apiAsset(e.tips?.diagram_expected ?? null)} label="正确" kind="good" />
                <DiagramImg src={apiAsset(e.tips?.diagram_actual ?? null)} label="你的" kind="bad" />
              </div>
            )}
            {e.tips?.tongue && <p><span className="label">舌头：</span>{e.tips.tongue}</p>}
            {e.tips?.lips && <p><span className="label">嘴唇：</span>{e.tips.lips}</p>}
            {e.tips?.jaw && <p><span className="label">下巴：</span>{e.tips.jaw}</p>}
          </div>
        ))
      )}
    </div>
  );
}

// A phone chip with a 🔊. It plays the SYLLABLE containing the phone (natural,
// practicable), never the bare phone in isolation, so the learner can A/B the
// target sound vs. what they produced in context.
function PhoneChip({
  phone,
  kind,
  play,
}: {
  phone: string;
  kind: "good" | "bad";
  play?: string[]; // syllable phones to speak (natural); falls back to the bare phone
}) {
  const audio = play && play.length ? play.join(" ") : phone;
  return (
    <span className={`phone-chip ${kind}`}>
      {phone}
      <button
        className="chip-speak"
        title={play && play.length > 1 ? `听音节 /${play.join("")}/` : "听这个音"}
        onClick={() => new Audio(phonemeAudioUrl(audio)).play().catch(() => {})}
      >
        🔊
      </button>
    </span>
  );
}

// "应读 [t]🔊，你读成了 [d]🔊" with colorblind-safe chips (blue = correct target,
// rose = what you actually said). Chips play the containing SYLLABLE: 应读 plays
// the target syllable, 你读成了 plays it with the slip voiced in place.
function ErrorHeadline({
  expected,
  actual,
  label,
  row,
  phones,
  refIndex,
}: {
  expected: string | null;
  actual: string | null;
  label: string;
  row?: number; // 0-based error index → hotkey Ctrl+(row+1); omit to hide hints
  phones: string[]; // the word's target phones (syllable context for playback)
  refIndex: number; // index of the error phone within `phones`
}) {
  const kind = errorKind(expected, actual);
  const n = (row ?? 0) + 1;
  const showHints = row != null && n <= 9;
  const expHint = showHints ? <kbd className="chip-key" title="听应读音节">⌃{n}</kbd> : null;
  const actHint = showHints ? <kbd className="chip-key" title="听你读的音节">⌃{n}·X</kbd> : null;
  const expPlay = expected ? syllablePhones(phones, refIndex) : undefined;
  const actPlay =
    kind === "sub" && actual
      ? syllablePhones(phones, refIndex, actual)
      : kind === "extra" && actual
        ? syllablePhonesInsert(phones, refIndex, actual)
        : undefined;
  if (kind === "sub") {
    return (
      <>
        应读 <PhoneChip phone={expected as string} kind="good" play={expPlay} />{expHint}，你读成了{" "}
        <PhoneChip phone={actual as string} kind="bad" play={actPlay} />{actHint}
      </>
    );
  }
  if (kind === "missing") {
    return (
      <>
        漏了这个音 <PhoneChip phone={expected as string} kind="good" play={expPlay} />{expHint}
      </>
    );
  }
  if (kind === "extra") {
    return (
      <>
        多了这个音 <PhoneChip phone={actual as string} kind="bad" play={actPlay} />{actHint}
      </>
    );
  }
  return <>{label}</>;
}

// A discoverable cheat-sheet of every keyboard shortcut, opened from the topbar.
function HotkeyLegend({ onClose }: { onClose: () => void }) {
  const groups: { title: string; items: [ReactNode, string][] }[] = [
    {
      title: "句子",
      items: [
        [<kbd key="a">A</kbd>, "原句"],
        [<kbd key="e">E</kbd>, "跟读"],
        [<kbd key="r">R</kbd>, "重读"],
        [<kbd key="sp">空格</kbd>, "继续"],
        [<kbd key="s">S</kbd>, "标准音"],
        [<kbd key="d">D</kbd>, "我的"],
        [<kbd key="c">C</kbd>, "克隆音"],
      ],
    },
    {
      title: "选词",
      items: [
        [
          <span key="lr"><kbd>←</kbd> <kbd>→</kbd></span>,
          "上一个 / 下一个词（未选中时 → 选第一个，← 选最后一个）",
        ],
      ],
    },
    {
      title: "选中的词",
      items: [
        [<kbd key="o">O</kbd>, "原声"],
        [<kbd key="b">B</kbd>, "标准音"],
        [<kbd key="m">M</kbd>, "我的录音"],
      ],
    },
    {
      title: "音节（选中词的第 N 个错误，从上往下数）",
      items: [
        [<span key="cn"><kbd>Ctrl</kbd>+<kbd>N</kbd></span>, "听「应读」的音节"],
        [<span key="cnx"><kbd>Ctrl</kbd>+<kbd>N</kbd> 时按住 <kbd>X</kbd></span>, "听「你读的」音节"],
      ],
    },
  ];
  return (
    <>
      <div className="keys-backdrop" onClick={onClose} />
      <div className="keys-panel" role="dialog" aria-label="键盘快捷键">
        <div className="keys-head">
          键盘快捷键
          <button className="keys-x" onClick={onClose} aria-label="关闭">✕</button>
        </div>
        {groups.map((g) => (
          <div key={g.title} className="keys-group">
            <div className="keys-group-title">{g.title}</div>
            {g.items.map(([keys, desc], i) => (
              <div key={i} className="keys-row">
                <span className="keys-keys">{keys}</span>
                <span className="keys-desc">{desc}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </>
  );
}

// Articulatory diagram (SVG from the backend). Hides itself if the phone has no
// diagram (the endpoint 404s), so we never show a broken image.
function DiagramImg({
  src,
  label,
  kind,
}: {
  src: string | null;
  label: string;
  kind: "good" | "bad";
}) {
  const [ok, setOk] = useState(true);
  if (!src || !ok) return null;
  return (
    <figure className="mouth-diagram-wrap">
      <figcaption className={`mouth-diagram-label ${kind}`}>{label}</figcaption>
      {/* eslint-disable-next-line @next/next/no-img-element -- proxied SVG, next/image is unsuitable */}
      <img className="mouth-diagram" src={src} alt="" onError={() => setOk(false)} />
    </figure>
  );
}

"use client";

import { useCallback, useEffect, useState } from "react";
import { fetchVideoInfo, fetchTranscript, fetchProgress } from "@/lib/api";
import { PracticeView } from "@/components/PracticeView";
import type { Transcript, VideoInfo } from "@/lib/types";

export default function Home() {
  const [url, setUrl] = useState("");
  const [language, setLanguage] = useState("fr-fr");
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [info, setInfo] = useState<VideoInfo | null>(null);
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [startIdx, setStartIdx] = useState(0);

  // idx: explicit sentence to open at; undefined → resume from saved progress.
  const load = useCallback(async (rawUrl: string, lang: string, idx?: number) => {
    setLanguage(lang);
    setLoading(true);
    setStatus("解析视频与字幕中…");
    try {
      const vinfo = await fetchVideoInfo(rawUrl, lang);
      const trans = await fetchTranscript(vinfo.video_id, lang);
      if (!trans.sentences.length) {
        setStatus("这个视频没有可用的字幕。");
        return;
      }
      let resume = idx ?? 0;
      if (idx == null) {
        const prog = await fetchProgress(vinfo.video_id);
        if (prog && prog.last_sentence_idx > 0) resume = prog.last_sentence_idx;
      }
      setInfo(vinfo);
      setTranscript(trans);
      setStartIdx(Math.min(resume, trans.sentences.length - 1));
      setStatus("");
    } catch (e) {
      setStatus(`加载失败：${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  // Restore a session from ?v=&lang=&i= so a refresh keeps you in place.
  useEffect(() => {
    const p = new URLSearchParams(window.location.search);
    const v = p.get("v");
    if (!v) return;
    const lang = p.get("lang") ?? "fr-fr";
    // Explicit ?i= wins; otherwise resume from saved progress.
    const iParam = p.get("i");
    const idx = iParam != null ? parseInt(iParam, 10) || 0 : undefined;
    const t = setTimeout(() => void load(`https://www.youtube.com/watch?v=${v}`, lang, idx), 0);
    return () => clearTimeout(t);
  }, [load]);

  if (info && transcript) {
    return (
      <PracticeView
        key={info.video_id}
        info={info}
        transcript={transcript}
        language={language}
        startIdx={startIdx}
        onExit={() => {
          setInfo(null);
          setTranscript(null);
          window.history.replaceState(null, "", "/");
        }}
        onLoadUrl={(newUrl) => void load(newUrl, language)}
      />
    );
  }

  return (
    <main className="app landing">
      <div className="brand"><span className="brand-mark">Lucerne</span> <span className="brand-sub">shadow reader</span></div>
      <div className="card">
        <h1>YouTube 影子跟读</h1>
        <p className="subtitle">贴一个 YouTube 链接，逐句跟读，即时点评你的法语发音。</p>
        <div className="url-form">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=…"
            onKeyDown={(e) => {
              if (e.key === "Enter" && url.trim()) void load(url, language);
            }}
          />
          <select value={language} onChange={(e) => setLanguage(e.target.value)}>
            <option value="fr-fr">Français</option>
            <option value="en-us">English</option>
          </select>
          <button className="btn-primary" disabled={loading || !url.trim()} onClick={() => void load(url, language)}>
            {loading ? "加载中…" : "加载"}
          </button>
        </div>
        <div className="status">{status}</div>
      </div>
    </main>
  );
}

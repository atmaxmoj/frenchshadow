"use client";

import { useEffect, useRef, useState } from "react";

// Minimal typing for the YouTube IFrame Player API we use.
export interface YTPlayerInstance {
  getCurrentTime(): number;
  getDuration(): number;
  seekTo(seconds: number, allowSeekAhead: boolean): void;
  playVideo(): void;
  pauseVideo(): void;
  cueVideoById(id: string): void;
  getPlayerState(): number;
  mute(): void;
  unMute(): void;
  setPlaybackRate(rate: number): void;
  destroy(): void;
}

interface YTNamespace {
  Player: new (el: HTMLElement, opts: unknown) => YTPlayerInstance;
  PlayerState: Record<string, number>;
}

declare global {
  interface Window {
    YT?: YTNamespace;
    onYouTubeIframeAPIReady?: () => void;
  }
}

let apiPromise: Promise<void> | null = null;

function loadYouTubeApi(): Promise<void> {
  if (typeof window === "undefined") return Promise.resolve();
  if (window.YT?.Player) return Promise.resolve();
  if (apiPromise) return apiPromise;
  apiPromise = new Promise<void>((resolve) => {
    const prev = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      prev?.();
      resolve();
    };
    if (!document.getElementById("yt-iframe-api")) {
      const s = document.createElement("script");
      s.id = "yt-iframe-api";
      s.src = "https://www.youtube.com/iframe_api";
      document.head.appendChild(s);
    }
  });
  return apiPromise;
}

export function YouTubePlayer({
  videoId,
  onReady,
}: {
  videoId: string;
  onReady: (player: YTPlayerInstance) => void;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<YTPlayerInstance | null>(null);

  useEffect(() => {
    let cancelled = false;
    const host = hostRef.current;
    void loadYouTubeApi().then(() => {
      if (cancelled || playerRef.current || !host || !window.YT) return;
      const mount = document.createElement("div");
      host.appendChild(mount);
      const player = new window.YT.Player(mount, {
        width: "100%",
        height: "100%",
        videoId,
        playerVars: { rel: 0, modestbranding: 1, controls: 1, playsinline: 1 },
        events: { onReady: () => onReady(player) },
      });
      playerRef.current = player;
    });
    return () => {
      cancelled = true;
      playerRef.current?.destroy?.();
      playerRef.current = null;
      if (host) host.innerHTML = "";
    };
    // Recreate only when the video changes; onReady must be stable (useCallback).
  }, [videoId, onReady]);

  return (
    <div className="player-wrap">
      <div ref={hostRef} className="player-host" />
    </div>
  );
}

// Poll the player's current time on every animation frame → single source of
// truth that drives sentence/word highlighting downstream.
export function useVideoTime(player: YTPlayerInstance | null): number {
  const [time, setTime] = useState(0);
  useEffect(() => {
    if (!player) return;
    let raf = 0;
    const loop = () => {
      try {
        const t = player.getCurrentTime();
        if (typeof t === "number" && !Number.isNaN(t)) setTime(t);
      } catch {
        // player not ready yet
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [player]);
  return time;
}

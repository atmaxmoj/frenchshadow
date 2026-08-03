"use client";

import { useEffect, useRef } from "react";

// Live spectrum "bars" visualizer (the kind music players show) driven by the
// recorder's AnalyserNode. Doubles as the unmistakable "mic is live" signal.
export function Waveform({
  recording,
  getAnalyser,
}: {
  recording: boolean;
  getAnalyser: () => AnalyserNode | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const BARS = 48;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const g = canvas.getContext("2d");
    if (!g) return;

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(r.width * dpr));
      canvas.height = Math.max(1, Math.floor(r.height * dpr));
    };
    resize();

    // Idle: a flat, muted baseline row — present but "off", so no layout jump.
    const drawIdle = () => {
      const w = canvas.width;
      const h = canvas.height;
      g.clearRect(0, 0, w, h);
      const gap = Math.max(1, w * 0.004);
      const barW = (w - gap * (BARS - 1)) / BARS;
      const barH = Math.max(h * 0.04, 2);
      const y = (h - barH) / 2;
      g.fillStyle = "hsl(30, 14%, 66%)";
      for (let i = 0; i < BARS; i++) g.fillRect(i * (barW + gap), y, barW, barH);
    };

    const analyser = recording ? getAnalyser() : null;
    if (!analyser) {
      drawIdle();
      return;
    }

    const bins = analyser.frequencyBinCount;
    const data = new Uint8Array(bins);
    const step = Math.max(1, Math.floor(bins / BARS));
    let raf = 0;

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      analyser.getByteFrequencyData(data);
      g.clearRect(0, 0, w, h);
      const gap = Math.max(1, w * 0.004);
      const barW = (w - gap * (BARS - 1)) / BARS;
      for (let i = 0; i < BARS; i++) {
        let sum = 0;
        for (let j = 0; j < step; j++) sum += data[i * step + j] ?? 0;
        const v = sum / step / 255;
        const barH = Math.max(h * 0.03, v * h);
        const x = i * (barW + gap);
        const y = (h - barH) / 2;
        // Warm amber → lighter as it gets louder (colorblind-safe, single hue).
        g.fillStyle = `hsl(28, 72%, ${44 + v * 14}%)`;
        g.fillRect(x, y, barW, barH);
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [recording, getAnalyser]);

  return (
    <div className={`waveform-wrap${recording ? " live" : ""}`}>
      <span className="rec-dot" />
      <span className="waveform-label">{recording ? "录音中" : "待机"}</span>
      <canvas ref={canvasRef} className="waveform" />
    </div>
  );
}

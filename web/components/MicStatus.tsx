"use client";

import { useMicPermission } from "@/hooks/useMicPermission";
import { micView } from "@/lib/mic";

// Persistent microphone indicator: authorization state (granted / prompt /
// denied) plus the live capture state, with an up-front "授权" action.
export function MicStatus({ recording }: { recording: boolean }) {
  const { state, request } = useMicPermission();
  const view = micView(state, recording);
  return (
    <div className="mic-status">
      <span className={`mic-dot ${view.dotClass}`} aria-hidden />
      <span className="mic-label">{view.label}</span>
      {view.showRequest && (
        <button className="mic-req" onClick={() => void request()}>
          授权麦克风
        </button>
      )}
      {view.showHint && (
        <span className="mic-hint">在地址栏左侧的站点设置里重新开启麦克风</span>
      )}
    </div>
  );
}

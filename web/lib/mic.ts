// Pure mapping from mic permission state → what the indicator should show.
// Kept React-free so it can be unit-tested without mocking browser APIs.

export type MicState = "granted" | "denied" | "prompt" | "unsupported";

export interface MicView {
  dotClass: "good" | "warn" | "bad" | "recording";
  label: string;
  showRequest: boolean; // offer a "授权" button
  showHint: boolean; // show the "open browser settings" hint
}

// Turn a getUserMedia / MediaRecorder failure into an accurate message. The
// caller passes the DOMException `name`; a blank/unknown name falls through to a
// generic message so we never wrongly blame permissions for a format/device error.
export function recorderErrorMessage(name: string, fallback = ""): string {
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "麦克风打不开：请确认①系统设置›隐私与安全性›麦克风 里允许了该浏览器，②地址栏的站点麦克风权限也已允许";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "没有检测到麦克风设备";
    case "NotReadableError":
    case "TrackStartError":
      return "麦克风被其他程序占用了";
    case "NotSupportedError":
      return "此浏览器不支持所需的录音格式";
    default:
      return fallback ? `无法开始录音：${fallback}` : "无法开始录音";
  }
}

export function micView(state: MicState, recording: boolean): MicView {
  if (recording) {
    return { dotClass: "recording", label: "🎙️ 录音中", showRequest: false, showHint: false };
  }
  switch (state) {
    case "granted":
      return { dotClass: "good", label: "麦克风已授权", showRequest: false, showHint: false };
    case "denied":
      return { dotClass: "bad", label: "麦克风被拒绝", showRequest: false, showHint: true };
    case "unsupported":
      return { dotClass: "bad", label: "此浏览器不支持麦克风", showRequest: false, showHint: false };
    case "prompt":
    default:
      return { dotClass: "warn", label: "麦克风未授权", showRequest: true, showHint: false };
  }
}

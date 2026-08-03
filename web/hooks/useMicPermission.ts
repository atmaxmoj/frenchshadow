"use client";

import { useCallback, useEffect, useState } from "react";
import type { MicState } from "@/lib/mic";

// Track the microphone permission live: seed from the Permissions API, then
// follow its `change` events (so revoking in browser settings updates the UI),
// and expose an explicit request() that triggers the grant prompt.
export function useMicPermission(): { state: MicState; request: () => Promise<void> } {
  const [state, setState] = useState<MicState>("prompt");

  useEffect(() => {
    let cancelled = false;
    let status: PermissionStatus | null = null;
    if (typeof navigator === "undefined" || !navigator.permissions?.query) {
      const t = setTimeout(() => {
        if (!cancelled) setState("unsupported");
      }, 0);
      return () => {
        cancelled = true;
        clearTimeout(t);
      };
    }
    navigator.permissions
      .query({ name: "microphone" as PermissionName })
      .then((s) => {
        if (cancelled) return;
        status = s;
        setState(s.state as MicState);
        s.onchange = () => setState(s.state as MicState);
      })
      .catch(() => {
        if (!cancelled) setState("unsupported");
      });
    return () => {
      cancelled = true;
      if (status) status.onchange = null;
    };
  }, []);

  const request = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((t) => t.stop());
      setState("granted");
    } catch {
      // Dismissed or blocked — the Permissions `change` handler will correct
      // this if the real state differs.
      setState("denied");
    }
  }, []);

  return { state, request };
}

"use client";

/**
 * Forward important frontend logs and errors to the backend so they show up
 * in the server log file without requiring the user to open the browser console.
 */

export type LogLevel = "info" | "warning" | "error";

interface LogEntry {
  level: LogLevel;
  message: string;
  context?: Record<string, unknown>;
}

let queue: LogEntry[] = [];
let flushTimer: ReturnType<typeof setTimeout> | null = null;
let isBuffering = false;

function flush() {
  if (flushTimer) {
    clearTimeout(flushTimer);
    flushTimer = null;
  }
  if (queue.length === 0) return;
  const batch = queue;
  queue = [];

  fetch("/api/client_log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(batch[0]), // send one at a time to keep it simple
  }).catch(() => {
    // If the backend is unreachable, drop the log to avoid recursion.
  });
}

function scheduleFlush() {
  if (flushTimer) return;
  flushTimer = setTimeout(flush, 500);
}

export function logClient(level: LogLevel, message: string, context?: Record<string, unknown>) {
  // Always mirror to the browser console first.
  const consoleMethod = level === "error" ? console.error : level === "warning" ? console.warn : console.log;
  consoleMethod(`[client:${level}]`, message, context ?? "");

  queue.push({ level, message, context });
  scheduleFlush();
}

export function logRecorder(message: string, context?: Record<string, unknown>) {
  logClient("info", message, context);
}

export function logRecorderError(message: string, context?: Record<string, unknown>) {
  logClient("error", message, context);
}

export function installGlobalClientLogging() {
  if (typeof window === "undefined") return;
  if (isBuffering) return;
  isBuffering = true;

  window.addEventListener("error", (event) => {
    logClient("error", "unhandled window error", {
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      stack: event.error?.stack,
    });
  });

  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason;
    logClient("error", "unhandled promise rejection", {
      reason: reason instanceof Error ? reason.message : String(reason),
      stack: reason instanceof Error ? reason.stack : undefined,
    });
  });
}

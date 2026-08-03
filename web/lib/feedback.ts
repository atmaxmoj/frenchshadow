// Pure interpretation of a phoneme error. The backend gives `expected` (the
// correct/target phone) and `actual` (what the learner actually produced); the
// raw English label "sound mismatch (t → d)" doesn't say which side is which.

export type ErrorKind = "sub" | "missing" | "extra" | "unknown";

export function errorKind(expected: string | null, actual: string | null): ErrorKind {
  if (expected && actual) return "sub"; // substituted one sound for another
  if (expected && !actual) return "missing"; // dropped a sound that should be there
  if (!expected && actual) return "extra"; // added a sound that shouldn't be there
  return "unknown";
}

// Plain-text headline (also used as the aria description).
export function errorHeadline(expected: string | null, actual: string | null): string {
  switch (errorKind(expected, actual)) {
    case "sub":
      return `应读 ${expected}，你读成了 ${actual}`;
    case "missing":
      return `漏了这个音：${expected}`;
    case "extra":
      return `多了这个音：${actual}`;
    default:
      return "发音有出入";
  }
}

// Prefix a backend-relative asset URL (e.g. "/mouth_diagram?phone=t") so it
// goes through the Next `/api/*` proxy to the FastAPI backend.
export function apiAsset(url: string | null): string | null {
  if (!url) return null;
  return url.startsWith("/api/") ? url : `/api${url}`;
}

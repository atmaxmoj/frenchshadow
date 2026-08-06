#!/usr/bin/env bash
# Start the shadow-reader FastAPI backend, the local CosyVoice3 clone service,
# and the Next.js frontend, with prefixed logs so all ports are visible in one
# terminal.
#
# Backend: http://localhost:8767
# CosyVoice3: http://localhost:8769
# Frontend: http://localhost:8768

set -euo pipefail

cd "$(dirname "$0")"

BACKEND_PORT="${1:-8767}"
FRONTEND_PORT="${2:-8768}"
COSY3_PORT="${3:-8769}"

export PHONEMIZER_ESPEAK_PATH="${PHONEMIZER_ESPEAK_PATH:-/opt/homebrew/bin/espeak-ng}"
export PHONEMIZER_ESPEAK_LIBRARY="${PHONEMIZER_ESPEAK_LIBRARY:-/opt/homebrew/lib/libespeak-ng.dylib}"

LOCAL_MODEL="$(cd "$(dirname "$0")" && pwd)/models/wav2vec2-lv-60-espeak-cv-ft"
if [ -d "$LOCAL_MODEL" ]; then
  export SHADOW_READER_MODEL="$LOCAL_MODEL"
fi

# CosyVoice3 lives in the user's tts-bench checkout with its own venv.
COSY3_PYTHON="${COSY3_PYTHON:-/Users/wangsijie/Develop/tools/tts-bench/cosyenv/bin/python}"

# Make sure the frontend dependencies are installed.
if [ ! -d "web/node_modules" ]; then
  echo "[setup] web/node_modules not found; running npm install..."
  (cd web && npm install)
fi

pids=()
cleanup() {
  echo "[start] shutting down..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit
}
trap cleanup INT TERM EXIT

# CosyVoice3 service (slow to load, start it first).
if [ -x "$COSY3_PYTHON" ]; then
  "$COSY3_PYTHON" api/cosy3_service.py --port "$COSY3_PORT" 2>&1 | sed -u 's/^/[cosy3:'"$COSY3_PORT"'] /' &
  pids+=("$!")
  echo "[start] waiting for CosyVoice3 on http://localhost:$COSY3_PORT ..."
  for i in {1..60}; do
    if curl -sf "http://localhost:$COSY3_PORT/health" >/dev/null 2>&1; then
      echo "[start] CosyVoice3 ready"
      break
    fi
    sleep 2
  done
else
  echo "[warn] CosyVoice3 python not found at $COSY3_PYTHON; clone button will be unavailable"
fi

# Backend log prefix.
./run.sh "$BACKEND_PORT" 2>&1 | sed -u 's/^/[backend:'"$BACKEND_PORT"'] /' &
pids+=("$!")

# Give the backend a moment to bind before starting the frontend.
sleep 2

# Frontend log prefix.
(cd web && npm run dev -- -p "$FRONTEND_PORT") 2>&1 | sed -u 's/^/[frontend:'"$FRONTEND_PORT"'] /' &
pids+=("$!")

echo "[start] backend on http://localhost:$BACKEND_PORT, CosyVoice3 on http://localhost:$COSY3_PORT, frontend on http://localhost:$FRONTEND_PORT"
echo "[start] press Ctrl-C to stop all"

wait

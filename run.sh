#!/usr/bin/env bash
# Start the shadow-reader backend on port 8767.
set -euo pipefail

cd "$(dirname "$0")"
PORT="${1:-8767}"

export PHONEMIZER_ESPEAK_PATH="${PHONEMIZER_ESPEAK_PATH:-/opt/homebrew/bin/espeak-ng}"
export PHONEMIZER_ESPEAK_LIBRARY="${PHONEMIZER_ESPEAK_LIBRARY:-/opt/homebrew/lib/libespeak-ng.dylib}"

# Use a locally downloaded model if present.
LOCAL_MODEL="$(cd "$(dirname "$0")" && pwd)/models/wav2vec2-lv-60-espeak-cv-ft"
if [ -d "$LOCAL_MODEL" ]; then
  export SHADOW_READER_MODEL="$LOCAL_MODEL"
fi

exec ./.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port "$PORT"

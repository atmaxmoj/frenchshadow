#!/usr/bin/env bash
# Start the shadow-reader backend on port 8767.
set -euo pipefail

cd "$(dirname "$0")"
PORT="${1:-8767}"

export PHONEMIZER_ESPEAK_PATH="${PHONEMIZER_ESPEAK_PATH:-/opt/homebrew/bin/espeak-ng}"
export PHONEMIZER_ESPEAK_LIBRARY="${PHONEMIZER_ESPEAK_LIBRARY:-/opt/homebrew/lib/libespeak-ng.dylib}"

exec ./.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port "$PORT"

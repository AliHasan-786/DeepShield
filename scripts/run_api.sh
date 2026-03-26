#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${DEEPSHIELD_HOST:-0.0.0.0}"
PORT="${DEEPSHIELD_PORT:-8000}"
WORKERS="${DEEPSHIELD_WORKERS:-1}"

cd "$ROOT_DIR"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

exec uvicorn api_server:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS"

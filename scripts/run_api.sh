#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${HOST:-0.0.0.0}"
PORT="${1:-7002}"

network_ip="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") { print $(i + 1); exit }}')"
if [ -z "$network_ip" ]; then
  network_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi

echo "SAM3 API"
echo "Local URL:   http://127.0.0.1:${PORT}"
if [ -n "$network_ip" ]; then
  echo "Network URL: http://${network_ip}:${PORT}"
fi
echo "(listening on ${HOST}:${PORT})"
echo

exec uv run uvicorn app.fastapi_app:app --host "$HOST" --port "$PORT"

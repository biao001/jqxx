#!/usr/bin/env bash
# 一键在局域网启动 DMS：后端(0.0.0.0:8000) + 前端(0.0.0.0:3000)，并打印手机扫码二维码。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

cleanup() { kill "${BACK_PID:-}" "${FRONT_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo ">> 启动后端 (0.0.0.0:${BACKEND_PORT}) ..."
( cd "$ROOT/backend" && DMS_HOST=0.0.0.0 DMS_PORT="$BACKEND_PORT" "$PY" run.py ) &
BACK_PID=$!

echo ">> 启动前端 vite (0.0.0.0:${FRONTEND_PORT}) ..."
( cd "$ROOT/frontend" && npm run dev -- --port "$FRONTEND_PORT" --host 0.0.0.0 ) &
FRONT_PID=$!

sleep 2
"$PY" "$ROOT/scripts/lan_qr.py" "$FRONTEND_PORT" || true

wait

#!/usr/bin/env bash
# 一键启动 DMS：后端(FastAPI, 0.0.0.0:8000) + 前端(Vite HTTPS, 0.0.0.0:3000)。
# 前端走 HTTPS(浏览器只在 https/localhost 下才放开摄像头)，证书取仓库 .certs/dms-dev.*。
# 前端 dev server 会把 /api /ws /health 反向代理到后端，局域网只需访问前端 3000 端口。
#
# 用法:
#   ./start.sh                # 启动前后端(HTTPS)，Ctrl+C 一起退出
#   HTTPS=0 ./start.sh        # 退回 HTTP(不推荐，摄像头会被浏览器拦)
#   BACKEND_PORT=8001 FRONTEND_PORT=3001 ./start.sh   # 自定义端口
#   ./start.sh --qr           # 额外打印手机扫码二维码（需要装好依赖）
set -euo pipefail

HTTPS="${HTTPS:-1}"   # 默认 HTTPS

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
WANT_QR=0
[ "${1:-}" = "--qr" ] && WANT_QR=1

# --- 定位 Python(优先仓库 venv) ---
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"

# --- 定位 Node 并加入 PATH(npm 的 shebang 依赖 node 在 PATH 上) ---
for cand in "$HOME/.local/node/bin" "$HOME/.nvm/versions/node"/*/bin /usr/local/bin /usr/bin; do
  if [ -x "$cand/node" ]; then PATH="$cand:$PATH"; break; fi
done
export PATH
if ! command -v node >/dev/null 2>&1; then
  echo "✗ 未找到 node，请先安装 Node.js(>=18)" >&2; exit 1
fi

echo ">> Python : $("$PY" --version 2>&1)  ($PY)"
echo ">> Node   : $(node -v)  ($(command -v node))"
echo

cleanup() {
  echo; echo ">> 正在关闭服务 ..."
  kill "${BACK_PID:-}" "${FRONT_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ">> 启动后端  http://0.0.0.0:${BACKEND_PORT}  ..."
( cd "$ROOT/backend" && DMS_HOST=0.0.0.0 DMS_PORT="$BACKEND_PORT" "$PY" run.py ) &
BACK_PID=$!

SCHEME=http
[ "$HTTPS" = 1 ] && SCHEME=https
echo ">> 启动前端  ${SCHEME}://0.0.0.0:${FRONTEND_PORT}  ..."
( cd "$ROOT/frontend" && DMS_HTTPS="$HTTPS" npm run dev -- --port "$FRONTEND_PORT" --host 0.0.0.0 ) &
FRONT_PID=$!

# --- 等后端健康检查就绪 ---
echo -n ">> 等待后端就绪 "
for _ in $(seq 1 40); do
  if curl -fsS -m 2 "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
  echo -n "."; sleep 1
done
echo
LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "============================================================"
if [ "${ok:-}" = 1 ]; then echo "✓ 后端就绪"; else echo "⚠ 后端未在 40s 内就绪，看上方日志"; fi
echo "  本机访问 : ${SCHEME}://localhost:${FRONTEND_PORT}"
[ -n "$LAN_IP" ] && echo "  局域网   : ${SCHEME}://${LAN_IP}:${FRONTEND_PORT}"
echo "  (前端已反向代理后端，无需直连 ${BACKEND_PORT})"
echo "  Ctrl+C 退出"
echo "============================================================"

# --- 可选：打印局域网二维码 ---
if [ "$WANT_QR" = 1 ] && [ -f "$ROOT/scripts/lan_qr.py" ]; then
  "$PY" "$ROOT/scripts/lan_qr.py" "$FRONTEND_PORT" || true
fi

wait

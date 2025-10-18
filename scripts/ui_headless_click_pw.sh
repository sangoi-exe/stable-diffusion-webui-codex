#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-7860}
CKPT_DIR=${CKPT_DIR:-"/mnt/f/stable-diffusion-webui-forge/models/Stable-diffusion"}
KEEP_SERVER=${KEEP_SERVER:-0}
URL=${URL:-"http://127.0.0.1:${PORT}"}
LOG=${LOG:-.webui-headless.log}
PID=${PID:-.webui-headless.pid}
PY=${PY:-"$HOME/.venv/bin/python"}
SELECTOR=${SELECTOR:-"#txt2img_generate"}
TIMEOUT_MS=${TIMEOUT_MS:-120000}
CHROME_BIN=${CHROME_BIN:-}
BROWSER=${BROWSER:-chromium}

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$PWD/.pw-browsers}"
mkdir -p "$MPLCONFIGDIR" || true
: > "$LOG"

# Kill if running
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then kill -9 "$(cat "$PID")" || true; fi
rm -f "$PID"

nohup "$PY" launch.py \
  --skip-prepare-environment \
  --skip-install \
  --skip-python-version-check \
  --api --api-server-stop \
  --listen --port "$PORT" \
  --ckpt-dir "$CKPT_DIR" \
  --always-cpu --skip-torch-cuda-test \
  >> "$LOG" 2>&1 & echo $! > "$PID"

echo "[ui-headless-pw] Waiting WebUI on :$PORT ..." >&2
for i in $(seq 1 180); do
  if curl -sf "$URL" >/dev/null; then break; fi
  sleep 1
done

if [ ! -d node_modules/playwright ]; then
  npm i -D playwright --silent
fi
# Ensure browsers are present (chromium only to avoid install-deps sudo)
PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH" npx playwright install chromium || true

echo "[ui-headless-pw] Clicking Generate via Playwright ($BROWSER)..." >&2
URL="$URL" SELECTOR="$SELECTOR" TIMEOUT_MS="$TIMEOUT_MS" CHROME_BIN="$CHROME_BIN" BROWSER="$BROWSER" PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH" \
  node tools/ui-click-generate.playwright.mjs

STATUS=$?

if [ "$KEEP_SERVER" != "1" ]; then
  echo "[ui-headless-pw] Stopping server..." >&2
  curl -s -X POST "$URL/sdapi/v1/server-stop" >/dev/null || true
  sleep 1
  if kill -0 "$(cat "$PID")" 2>/dev/null; then kill -9 "$(cat "$PID")" || true; fi
  rm -f "$PID"
fi

exit $STATUS

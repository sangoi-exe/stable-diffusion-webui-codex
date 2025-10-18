#!/usr/bin/env bash
set -euo pipefail

# Launch WebUI server (UI + API) in background, then run headless Chrome (via Puppeteer)
# to open the UI and click the 'Generate' button. Finally, stop the server.

PORT=${PORT:-7860}
CKPT_DIR=${CKPT_DIR:-"/mnt/f/stable-diffusion-webui-forge/models/Stable-diffusion"}
KEEP_SERVER=${KEEP_SERVER:-0}
URL=${URL:-"http://127.0.0.1:${PORT}"}
LOG=${LOG:-.webui-headless.log}
PID=${PID:-.webui-headless.pid}
PY=${PY:-"$HOME/.venv/bin/python"}
SELECTOR=${SELECTOR:-"#txt2img_generate"}
TIMEOUT_MS=${TIMEOUT_MS:-120000}
NPM_CONFIG_CACHE=${NPM_CONFIG_CACHE:-$PWD/.npm-cache}
PUPPETEER_CACHE_DIR=${PUPPETEER_CACHE_DIR:-$PWD/.puppeteer-cache}

echo "[ui-headless] Starting server on :$PORT (ckpt-dir=$CKPT_DIR)" >&2
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
export NPM_CONFIG_CACHE
export PUPPETEER_CACHE_DIR
mkdir -p "$MPLCONFIGDIR" || true
: > "$LOG"

# Kill if running
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  kill -9 "$(cat "$PID")" || true
  rm -f "$PID"
fi

nohup "$PY" launch.py \
  --skip-prepare-environment \
  --skip-install \
  --skip-python-version-check \
  --api --api-server-stop \
  --listen --port "$PORT" \
  --ckpt-dir "$CKPT_DIR" \
  --always-cpu --skip-torch-cuda-test \
  >> "$LOG" 2>&1 & echo $! > "$PID"

echo "[ui-headless] Waiting up to 180s for UI..." >&2
for i in $(seq 1 180); do
  if curl -sf "$URL/" >/dev/null; then break; fi
  sleep 1
done

# Ensure puppeteer is installed and has a browser available
if [ ! -d node_modules/puppeteer ]; then
  echo "[ui-headless] Installing puppeteer (this downloads Chromium) ..." >&2
  export NPM_CONFIG_CACHE="${NPM_CONFIG_CACHE:-$PWD/.npm-cache}"
  export PUPPETEER_CACHE_DIR="${PUPPETEER_CACHE_DIR:-$PWD/.puppeteer-cache}"
  mkdir -p "$NPM_CONFIG_CACHE" "$PUPPETEER_CACHE_DIR" || true
  npm i -D puppeteer --silent
  # Ensure a chrome binary is downloaded into the cache
  npx puppeteer browsers install chrome --cache-dir "$PUPPETEER_CACHE_DIR" --quiet || npx puppeteer browsers install chrome --cache-dir "$PUPPETEER_CACHE_DIR"
fi

echo "[ui-headless] Running headless click test against $URL (chrome) ..." >&2
URL="$URL" SELECTOR="$SELECTOR" TIMEOUT_MS="$TIMEOUT_MS" PUPPETEER_CACHE_DIR="$PUPPETEER_CACHE_DIR" node tools/ui-click-generate.js || STATUS=$? || true

if [ "${STATUS:-0}" != "0" ]; then
  echo "[ui-headless] Chrome failed (status $STATUS). Trying Firefox fallback..." >&2
  export PUPPETEER_PRODUCT=firefox
  npm i -D puppeteer --silent || true
  npx puppeteer browsers install firefox --cache-dir "$PUPPETEER_CACHE_DIR" --quiet || true
  URL="$URL" SELECTOR="$SELECTOR" TIMEOUT_MS="$TIMEOUT_MS" node tools/ui-click-generate.js || STATUS=$? || true
fi

STATUS=${STATUS:-0}

if [ "$KEEP_SERVER" != "1" ]; then
  echo "[ui-headless] Stopping server..." >&2
  curl -s -X POST "$URL/sdapi/v1/server-stop" >/dev/null || true
  sleep 1
  if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then kill -9 "$(cat "$PID")" || true; fi
  rm -f "$PID"
fi

echo "[ui-headless] Done with status $STATUS" >&2
exit $STATUS

#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke test for API-only inference via CLI (no image return).
# Requirements: curl, jq, Python venv with WebUI deps installed.

PORT=${PORT:-7861}
CKPT_DIR=${CKPT_DIR:-"/mnt/f/stable-diffusion-webui-forge/models/Stable-diffusion"}
PROMPT=${PROMPT:-"smoke test"}
STEPS=${STEPS:-1}
SIZE=${SIZE:-64}
LOG=${LOG:-.webui-api.log}
PID=${PID:-.webui-api.pid}
PY=${PY:-"$HOME/.venv/bin/python"}

if ! command -v curl >/dev/null || ! command -v jq >/dev/null; then
  echo "ERROR: curl and jq are required." >&2
  exit 1
fi

# Kill previous run if any
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  kill -9 "$(cat "$PID")" || true
  rm -f "$PID"
fi
: > "$LOG"

nohup "$PY" launch.py \
  --skip-prepare-environment \
  --skip-install \
  --skip-python-version-check \
  --nowebui --api --api-server-stop \
  --listen --port "$PORT" \
  --ckpt-dir "$CKPT_DIR" \
  --always-cpu --skip-torch-cuda-test \
  >> "$LOG" 2>&1 & echo $! > "$PID"

echo "Waiting API on :$PORT ..." >&2
for i in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:$PORT/sdapi/v1/sd-models" >/dev/null; then break; fi
  sleep 1
done

MODEL_TITLE=${MODEL_TITLE:-$(curl -s "http://127.0.0.1:$PORT/sdapi/v1/sd-models" | jq -r '.[0].title // empty')}
if [ -z "${MODEL_TITLE:-}" ]; then
  echo "ERROR: No models found under CKPT_DIR=$CKPT_DIR" >&2
  exit 2
fi

curl -s -X POST "http://127.0.0.1:$PORT/sdapi/v1/options" \
  -H 'Content-Type: application/json' \
  -d "{\"sd_model_checkpoint\": \"$MODEL_TITLE\"}" >/dev/null

RESP=$(curl -s -m 180 -X POST "http://127.0.0.1:$PORT/sdapi/v1/txt2img" \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"$PROMPT\",\"steps\":$STEPS,\"width\":$SIZE,\"height\":$SIZE,\"batch_size\":1,\"n_iter\":1,\"send_images\":false,\"save_images\":false}")

echo "$RESP" | jq -r '.info'

# Stop server
curl -s -X POST "http://127.0.0.1:$PORT/sdapi/v1/server-stop" >/dev/null || true
sleep 1
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then kill -9 "$(cat "$PID")" || true; fi
rm -f "$PID"

exit 0


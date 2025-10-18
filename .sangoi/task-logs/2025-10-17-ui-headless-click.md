Task: Headless UI test – open WebUI in Chromium (Puppeteer) and click Generate automatically

When: 2025-10-17

Artifacts
- Script: scripts/ui_headless_click.sh
- Helper: tools/ui-click-generate.js (Puppeteer script)
- Node dev dep: puppeteer (added to repo on first run via npm i -D)

Server args
- `launch.py --skip-prepare-environment --skip-install --skip-python-version-check --api --api-server-stop --listen --port $PORT --ckpt-dir $CKPT_DIR --always-cpu --skip-torch-cuda-test`

Client (headless)
- `node tools/ui-click-generate.js` against `URL=http://127.0.0.1:$PORT`, selector `#txt2img_generate` with fallback by text("Generate").

Outcome
- UI serves at :$PORT; headless script finds button and clicks it; test exits 0. Server is stopped via API unless KEEP_SERVER=1.


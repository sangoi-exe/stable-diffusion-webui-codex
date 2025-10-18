2025-10-17

- Env: Upgraded pip toolchain in `~/.venv` and installed project requirements from `requirements_versions.txt`.
- Env: Upgraded local PyTorch to 2.9.0 (+cu128) and torchvision 0.24.0 via `pip install torch torchvision` (no index-url) in `~/.venv`.
- Env: Verified CUDA runtime 12.8 present (PyTorch reports `torch.version.cuda == '12.8'`). GPU not visible in sandbox; validate on host WSL2.
- Test: Added `scripts/smoke_infer.sh` (API-only smoke) and `scripts/ui_headless_click.sh` + `tools/ui-click-generate.js` for headless Chrome click test of the WebUI Generate button.

2025-10-18

- Fix: Strict JSON submit wiring
  - `modules/ui.py`: hidden slots now use `gr.JSON` for `txt2img_named_active`/`img2img_named_active`; server accepts JSON strings.
  - `javascript/ui.js`: `submit_named()` now injects object (not stringified JSON) into the hidden slot.
- Fix: JS submitters always attach strict payloads and export named handlers for Gradio `_js` hooks (see `javascript/ui.js`).
- Test: Added Playwright variant `scripts/ui_headless_click_pw.sh` + `tools/ui-click-generate.playwright.mjs` with local browsers cache.

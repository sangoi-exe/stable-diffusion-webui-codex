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
- Fix: Typed strict submit flow (`javascript/ui.js`) to satisfy `npm run typecheck` (added `StrictBuilder` typedef tightened to `IArguments`, error formatter, and UIWindow exports).
- UI: Restored VAE selector to checkpoint-style dropdown and moved text encoders to a separate multiselect (`modules_forge/main_entry.py`, `modules/ui_settings.py`, `javascript/ui.js`).
- Test: Added Playwright variant `scripts/ui_headless_click_pw.sh` + `tools/ui-click-generate.playwright.mjs` with local browsers cache.
- Fix: VAE/Text Encoder dropdown wouldn’t open due to global `.hidden { display: none !important; }` override. Removed the rule from `style.css`; dropdown overlays now render correctly under Gradio 5 while keeping overflow visible via `div.gradio-dropdown { overflow: visible !important; }`.
- UI: Removed legacy combined selector. Split Hires selector into `Hires VAE` (single) and `Hires Text Encoder(s)` (multi); kept hidden compatibility field `hr_vae_te` to preserve payload shape (`modules/ui.py`).
 - UI: Quicksettings VAE (`sd_vae`) and Text Encoders (`sd_text_encoders`) now render unconditionally inside Quicksettings row; removed `render=False` and explicit `.render()` calls to avoid duplicate/hidden components. Element ids updated to `sd_vae` and `sd_text_encoders`.
 - UX: Quicksettings VAE now espelha o seletor de Checkpoint (mesmo dropdown, refresh, eventos). Unificamos discovery (inclui `sd_vae.vae_dict`) e tornamos a resolução nome→caminho mais resiliente (aceita rótulos com sufixo `[hash]`).
 - Refactor: Eliminated legacy submit path. Server no longer scans args to locate strict JSON; it must be the last input. Removed pre-submit fallback that copied incoming positional values when strict JSON is absent (`modules/ui.py`).

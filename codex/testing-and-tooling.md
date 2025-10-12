Testing and Tooling
====================

Logging
- Default logger: Rich console with color and tqdm‑aware output.
- Change verbosity: set env `SD_WEBUI_LOG_LEVEL` to `DEBUG`, `INFO`, etc.
- Fallback: if Rich/Colorama are missing at bootstrap, code falls back to std logging.

Requirements Updater
- Script: `python tools/update_requirements.py --write`
  - Excludes: torch, torchvision, torchaudio (kept as‑is by default)
  - To drop excluded packages entirely (so pip won’t touch local builds): add `--drop-excluded`.
  - Force a specific pin: `--force pkg==X.Y.Z` (may repeat).

Baseline Capture (manual)
- Use a WebUI preset e seeds fixos para gerar PNGs e salvar infotext/metadata.
- Guarde os artefatos de referência em `tests/backend/fixtures/txt2img/` quando necessário.

Legacy Reference
- Quando um refactor quebrar a pipeline ou o comportamento divergir, consulte o código em `legacy/` (snapshot do Forge main) para entender o fluxo funcional esperado antes de corrigir.

Manual UI Validation (preferred)
- Cover these flows when validating refactors:
  - LoRA activation (with/without Disable extra networks)
  - Hires pass (scale, steps, upscaler; start from base res, not final)
  - Refiner hand‑off (if enabled)
  - Multi‑iteration batches (progress text, seeds)
- If behaviour drifts, capture a short log with `SD_WEBUI_LOG_LEVEL=DEBUG` and attach prompt + settings.

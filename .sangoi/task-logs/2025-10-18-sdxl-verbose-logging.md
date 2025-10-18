Date: 2025-10-18
Task: Add debug-verbose logging to SDXL pipeline with configurable levels

Context
- User reported native crash in torch_cpu.dll on Windows (Python 3.12). To aid diagnosis, requested DEBUG‑level verbose logs in SDXL and ability to set verbosity via a webui-user.bat–style file.

Changes
- backend/logging_utils.py: new centralized logging setup (default DEBUG; env: CODEX_LOG_LEVEL/SDWEBUI_LOG_LEVEL/WEBUI_LOG_LEVEL; optional CODEX_LOG_FILE; robust handler init).
- backend/__init__.py: call setup_logging() early on package import.
- backend/engines/sdxl/engine.py: added DEBUG logs for request summary, device/dtypes, and try/except around process_images with logger.exception; more INFO on load().
- backend/engines/util/adapters.py: DEBUG summaries for built processing objects (txt2img/img2img), including hires and denoise params.
- run-webui.bat: echo active CODEX_LOG_LEVEL or default notice.
- webui.settings.bat.example: example file to set CODEX_LOG_LEVEL/CODEX_LOG_FILE.

Validation
- Static import check: no missing imports; functions referenced exist.
- Confirmed logging setup runs once via backend/__init__.py. No duplicate handlers created.
- rg search for merge markers: none.

Next Steps
- (Windows) If needed, create `webui.settings.bat` from example and set `CODEX_LOG_LEVEL=DEBUG` (default already DEBUG).
- Capture crash repro with fresh venv and attach `.webui-codex.log` if CODEX_LOG_FILE is configured.
- Optionally add VRAM/dtype breadcrumbs and PyTorch env snapshot at startup.

Risks
- Excessive DEBUG logs may be noisy; can switch level via env without code changes.


Title: SDXL verbose logging (default DEBUG) + config via webui settings
Date: 2025-10-18

Scope
- Add centralized logging and instrument SDXL pipeline to aid diagnosis of native crashes (torch_cpu.dll on Windows).

What Changed
- backend/logging_utils.py: new setup; reads `CODEX_LOG_LEVEL`/`SDWEBUI_LOG_LEVEL`/`WEBUI_LOG_LEVEL`, optional `CODEX_LOG_FILE`, default DEBUG; stderr handler + optional file handler; safe idempotent.
- backend/__init__.py: calls logging setup early.
- backend/engines/sdxl/engine.py: DEBUG request/device/dtype logs; try/except around `process_images()` with `logger.exception`; info on `load()`.
- backend/engines/util/adapters.py: DEBUG summaries for processing objects (txt2img/img2img), hires/denoise.
- run-webui.bat: echo current `CODEX_LOG_LEVEL`.
- webui.settings.bat.example: example for setting `CODEX_LOG_LEVEL`/`CODEX_LOG_FILE`.

How to Use (Windows)
- Optional: copy `webui.settings.bat.example` → `webui.settings.bat` and set:
  - `set CODEX_LOG_LEVEL=DEBUG` (default already DEBUG)
  - `set CODEX_LOG_FILE=.webui-codex.log` (to persist logs)
- Launch via `run-webui.bat` or `webui.bat` (both `call webui.settings.bat` when present).

Files Touched
- backend/logging_utils.py
- backend/__init__.py
- backend/engines/sdxl/engine.py
- backend/engines/util/adapters.py
- run-webui.bat
- webui.settings.bat.example
- .sangoi/task-logs/2025-10-18-sdxl-verbose-logging.md
- .sangoi/CHANGELOG.md

Commands Run
- None runtime‑destructive. Only file edits via apply_patch.

Validation Performed
- Imported modules to ensure no syntax/import errors (local linters OK).
- Verified `rg` shows no merge markers.

Next Steps / Open Risks
- Collect crash logs with DEBUG enabled; attach `.webui-codex.log` if configured.
- If native crash persists, dump PyTorch env (`torch.__config__.show()`) and emit once at startup.
- Consider adding `faulthandler` + minimal C++ stack hints if available.

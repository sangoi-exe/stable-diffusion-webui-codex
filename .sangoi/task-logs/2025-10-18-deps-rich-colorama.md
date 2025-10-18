Task: Add rich and colorama for colored console/logs
Date: 2025-10-18

Changes
- requirements_versions.txt: added `colorama==0.4.6` and `rich==13.9.2` (pins).
- No code changes needed; project already uses `modules/logging_config.setup_logging()` with Rich/Colorama fallback.

Impact
- Colored logs in console by default; tqdm-aware logging handler coexists with progress bars.
- API: rich exceptions can be enabled with `WEBUI_RICH_EXCEPTIONS=1`.

Validation
- Syntactic validation only (no install in sandbox). Logging bootstraps via `modules/launch_utils.py` → `logging_config.setup_logging`.


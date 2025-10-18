Date: 2025-10-18
Task: Inline user settings in run-webui.bat (no separate settings file)

Changes
- run-webui.bat: added inline config block for CODEX_* env (log level, UNet/VAE dtypes, CPU toggle, all-fp32). Echoes active values at startup.
- Removed webui.settings.bat.example to avoid suggesting a new settings file.
- Kept optional `call webui.settings.bat` for compatibility if user wants an external file.

Reasoning
- User requested not to introduce a new .bat solely for settings; making run-webui self-contained is simpler.


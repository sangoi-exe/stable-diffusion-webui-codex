Session Log
===========

2025-10-11 – Runtime + Tooling
- Backend: advanced txt2img runtime to include RNG initialization, script batch hooks, extra network activation, post‑sample callbacks, and progress updates.
- LoRA: added diagnostics and safer activation path; continuing to harden registry initialization order across entrypoints.
- Logging: introduced Rich/Colorama pipeline with tqdm‑aware handler (fallback to std logging when unavailable).
- Deps: added updater `tools/update_requirements.py`; refreshed pins, excluded torch family for local builds; resolved open‑clip/protobuf pin conflict.
- Validation: agreed to prioritize UI manual tests; capture CLI remains available for deterministic artefacts.

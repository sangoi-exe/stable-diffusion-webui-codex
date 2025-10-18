Task: Settings toggle for video export (bind to engine options)
Date: 2025-10-18

Changes
- modules/shared_options.py: added `codex_export_video` (System ▸ Codex) boolean option.
- modules/txt2vid.py & modules/img2vid.py: pass `engine_options={"export_video": opts.codex_export_video}` to `InferenceOrchestrator.run()`.
- WAN TI2V‑5B engine: `_maybe_export_video()` now honors `self._load_options['export_video']` in addition to env `CODEX_EXPORT_VIDEO=1`.

Notes
- Default false (off). Keeps strict contracts: frames still returned, video export optional.


Task: Optional video export (mp4/webm) for WAN TI2V‑5B
Date: 2025-10-18

Changes
- backend/engines/video/wan/ti2v5b_engine.py: added `_maybe_export_video()` using ffmpeg when `CODEX_EXPORT_VIDEO=1`.
- Emits no errors if ffmpeg missing or frames are non-PIL; returns metadata in `info.video`.
- Exports to `artifacts/videos/<timestamp>/out.mp4|out.webm` and yields progress events (prepare/run already present).

Notes
- Default OFF; honors strict contracts (images list unchanged). UI not wired to auto-play; payload is forward compatible.


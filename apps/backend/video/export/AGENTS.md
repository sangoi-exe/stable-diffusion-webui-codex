# apps/backend/video/export Overview
Date: 2025-12-16
Last Review: 2026-08-26
Status: Active

## Purpose
- Encode frame sequences to a video container (mp4/webm/gif) using ffmpeg.

## Key files
- `apps/backend/video/export/ffmpeg_exporter.py` — `export_video()` writes frame PNGs to a temp dir then runs ffmpeg.

## Notes
- Output root is `CODEX_ROOT/output` (repo-local) and served via `/api/output/{rel_path}`.
- Backend must serve outputs via a root-scoped file route (see `/api/output/{rel_path}`) rather than exposing arbitrary paths.
- Export errors should be explicit (`VideoExportError`) so users can fix missing ffmpeg/codec issues; ffmpeg resolution now uses shared deterministic runtime resolver (`apps/backend/video/runtime_dependencies.py`).
- 2026-01-02: Added standardized file header docstrings to exporter modules (doc-only change; part of rollout).
- 2026-01-18: `export/__init__.py` is now a package marker (no re-exports); import `export_video` from `apps/backend/video/export/ffmpeg_exporter.py`.
- 2026-02-21: `ffmpeg_exporter.py` now parses boolean options (`save_output`, `pingpong`, `trim_to_audio`, `save_metadata`) via shared strict bool parsing and fails loud on invalid literals (no permissive truthy coercion drift from use-case contracts).
- 2026-03-11: `ffmpeg_exporter.py` now fails loud when `audio_source_path` is missing or when audio mux is requested for unsupported containers, and `VideoExportResult` now reports `has_audio` so shared video metadata can distinguish silent video from muxed output truthfully.
- 2026-03-13: `ffmpeg_exporter.py::resolve_video_export_container()` rejects unknown `video_options.format` values with `VideoExportError` instead of silently coercing them to mp4/h264.
- 2026-08-26: `ffmpeg_exporter.py` accepts finite fractional FPS values and passes them to ffmpeg without integer rounding. The dedicated SeedVR2 flow therefore retains the probed source FPS in export metadata and encoding.

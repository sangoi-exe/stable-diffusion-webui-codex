# apps/backend/video/export Overview
Date: 2025-12-16
Last Review: 2026-08-27
Status: Active

## Purpose
- Encode frame sequences to a video container (mp4/webm/gif) using ffmpeg.

## Key files
- `apps/backend/video/export/ffmpeg_exporter.py` — `export_video()` owns generic CFR frame export; `export_timestamped_video()` owns verified VFR MP4 staging and publication.
- `apps/backend/video/export/mp4_timing.py` — internal, fail-loud MP4 atom patch for the staged H.264 video track's terminal VFR sample duration. It is not a generic editing API.

## Notes
- Output root is `CODEX_ROOT/output` (repo-local) and served via `/api/output/{rel_path}`.
- Backend must serve outputs via a root-scoped file route (see `/api/output/{rel_path}`) rather than exposing arbitrary paths.
- Export errors should be explicit (`VideoExportError`) so users can fix missing ffmpeg/codec issues; ffmpeg resolution now uses shared deterministic runtime resolver (`apps/backend/video/runtime_dependencies.py`).
- 2026-01-02: Added standardized file header docstrings to exporter modules (doc-only change; part of rollout).
- 2026-01-18: `export/__init__.py` is now a package marker (no re-exports); import `export_video` from `apps/backend/video/export/ffmpeg_exporter.py`.
- 2026-02-21: `ffmpeg_exporter.py` now parses boolean options (`save_output`, `pingpong`, `trim_to_audio`, `save_metadata`) via shared strict bool parsing and fails loud on invalid literals (no permissive truthy coercion drift from use-case contracts).
- 2026-03-11: `ffmpeg_exporter.py` now fails loud when `audio_source_path` is missing or when audio mux is requested for unsupported containers, and `VideoExportResult` now reports `has_audio` so shared video metadata can distinguish silent video from muxed output truthfully.
- 2026-03-13: `ffmpeg_exporter.py::resolve_video_export_container()` rejects unknown `video_options.format` values with `VideoExportError` instead of silently coercing them to mp4/h264.
- 2026-08-27: `ffmpeg_exporter.py` accepts finite fractional FPS values for generic CFR export without integer rounding.
- 2026-08-27: `export_timestamped_video()` writes a concat manifest with per-frame durations, encodes from that manifest, and calls the bounded `mp4_timing.py` helper to retain the terminal VFR frame duration without adding a decoded frame. The helper validates every supported atom field and all fixed-offset writes before modifying staged bytes. The export translates MP4 structural and staging-file I/O failures to `VideoExportError`, verifies frame cardinality, decoded presentation-time offsets, video-track end duration, and source-audio codec/duration before atomically publishing the MP4 and JSON sidecar. It removes staged or partially published artifacts when cancellation or verification fails.

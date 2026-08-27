# apps/backend/video/io Overview
Date: 2025-12-16
Last Review: 2026-08-27
Status: Active

## Purpose
- Provide ffmpeg/ffprobe-backed video probing, decoded-frame timing, and frame-extraction utilities for backend video tasks.

## Key files
- `apps/backend/video/io/ffmpeg.py` — `probe_video()`, `probe_video_timing()`, and `extract_frames()` wrappers (fail-fast, explicit errors).

## Notes
- Keep imports minimal and use subprocess calls (no cv2 dependency).
- All output paths must be explicit and task-scoped. Callers select the task cache or other canonical work owner; this module does not invent an output root.
- Raise `FFmpegUnavailableError` when ffmpeg/ffprobe are missing instead of silently degrading; binary resolution now uses shared deterministic runtime resolver (`apps/backend/video/runtime_dependencies.py`).
- 2026-01-02: Added standardized file header docstrings to video IO modules (doc-only change; part of rollout).
- 2026-01-18: `io/__init__.py` is now a package marker (no re-exports); import `probe_video`/`extract_frames` from `apps/backend/video/io/ffmpeg.py`.
- 2026-08-27: `probe_video()` returns container duration plus stream-local video/audio durations. `probe_video_timing()` returns ordered decoded-frame presentation timestamps and fails loud on missing or non-increasing timing. The dedicated SeedVR2 flow uses this evidence instead of scalar FPS when it verifies VFR output and its terminal video duration independently from audio length.

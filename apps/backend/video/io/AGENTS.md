# apps/backend/video/io Overview
Date: 2025-12-16
Last Review: 2026-08-30
Status: Active

## Purpose
- Provide ffmpeg/ffprobe-backed video probing, decoded-frame timing, and frame-extraction utilities for backend video tasks.

## Key files
- `apps/backend/video/io/ffmpeg.py` — cancellable `probe_video()`, `probe_video_timing()`, and `extract_frames()` wrappers with exact decoded counts and stream-origin evidence.

## Notes
- Keep imports minimal and use subprocess calls (no cv2 dependency).
- All output paths must be explicit and task-scoped. Callers select the task cache or other canonical work owner; this module does not invent an output root.
- Raise `FFmpegUnavailableError` when ffmpeg/ffprobe are missing instead of silently degrading; binary resolution now uses shared deterministic runtime resolver (`apps/backend/video/runtime_dependencies.py`).
- 2026-01-02: Added standardized file header docstrings to video IO modules (doc-only change; part of rollout).
- 2026-01-18: `io/__init__.py` is now a package marker (no re-exports); import `probe_video`/`extract_frames` from `apps/backend/video/io/ffmpeg.py`.
- 2026-08-30: `probe_video(count_frames=True)` returns an exact decoded-frame count plus video/audio codecs, durations, and stream origins. Probe, timing, and extraction subprocesses honor task cancellation and terminate their complete process groups. The dedicated SeedVR2 route reuses the admitted probe when it reads decoded timing.

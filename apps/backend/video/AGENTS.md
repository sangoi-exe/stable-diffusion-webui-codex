# apps/backend/video Overview
Date: 2025-10-28
Last Review: 2026-08-30
Status: Active

## Purpose
- Houses shared video-specific helpers used across WAN22 and other video-capable pipelines.

## Subdirectories
- `interpolation/` — Video frame interpolation utilities (e.g., RIFE wrappers).
- `upscaling/` — SeedVR2 child-process runner (repo-local runtime bootstrap, concurrent source/target accelerator admission, numeric frame identity, full process-group cleanup, and fail-loud output validation).
- `io/` — Cancellable input video probing, exact decoded-frame counts, stream origins, decoded-frame timing, and extraction helpers (ffprobe/ffmpeg wrappers).
- `flow/` — Optical flow estimation + frame warping (torchvision RAFT).
- `export/` — Frame → video encoding and verified timestamp-aware MP4 publication with relative A/V-origin checks and sidecar-before-media visibility (ffmpeg exporter; writes under `CODEX_ROOT/output`).

## Notes
- Keep video utilities generic so multiple engines/use cases can reuse them.
- Video IO/export resolves ffmpeg binaries via deterministic repo-local runtime paths first (`.uv/xdg-data/ffmpeg-downloader/ffmpeg`), then explicit env overrides/PATH.
- Default RIFE checkpoint is provisioned under `.uv/xdg-data/rife/rife47.pth`; interpolation runtime now attempts one-shot auto-provision only for default-token requests (custom paths/env overrides still fail loud).
- Flow guidance requires `torch` + `torchvision`.
- 2026-01-02: Added standardized file header docstrings to video export modules (doc-only change; part of rollout).
- 2026-02-23: RIFE/RAFT runtime defaults now derive device identity from memory-manager mount-device authority; unsupported device/backend combinations fail loud instead of implicit CUDA→CPU fallback.
- 2026-08-31: `upscaling/seedvr2.py` admits direct and bounded streaming execution from measured CUDA/MPS memory with concurrent source/target batches and upstream uniform padding included, terminates the complete child process group, drains diagnostics to EOF, and validates contiguous numeric source-cardinality PNG output without a CPU fallback.
- 2026-08-30: The dedicated use case consumes router-admitted media/resource evidence. The timestamp-aware exporter verifies frame offsets, terminal video duration, relative source A/V origin, and stream-copied audio before it publishes the JSON sidecar and then the final MP4.

# apps/backend/video Overview
Date: 2025-10-28
Last Review: 2026-08-27
Status: Active

## Purpose
- Houses shared video-specific helpers used across WAN22 and other video-capable pipelines.

## Subdirectories
- `interpolation/` — Video frame interpolation utilities (e.g., RIFE wrappers).
- `upscaling/` — SeedVR2 child-process runner (repo-local runtime bootstrap, accelerator-memory admission, and fail-loud output validation).
- `io/` — Input video probing, decoded-frame timing, and decoding helpers (ffprobe/ffmpeg wrappers).
- `flow/` — Optical flow estimation + frame warping (torchvision RAFT).
- `export/` — Frame → video encoding and verified timestamp-aware MP4 publication (ffmpeg exporter; writes under `CODEX_ROOT/output`).

## Notes
- Keep video utilities generic so multiple engines/use cases can reuse them.
- Video IO/export resolves ffmpeg binaries via deterministic repo-local runtime paths first (`.uv/xdg-data/ffmpeg-downloader/ffmpeg`), then explicit env overrides/PATH.
- Default RIFE checkpoint is provisioned under `.uv/xdg-data/rife/rife47.pth`; interpolation runtime now attempts one-shot auto-provision only for default-token requests (custom paths/env overrides still fail loud).
- Flow guidance requires `torch` + `torchvision`.
- 2026-01-02: Added standardized file header docstrings to video export modules (doc-only change; part of rollout).
- 2026-02-23: RIFE/RAFT runtime defaults now derive device identity from memory-manager mount-device authority; unsupported device/backend combinations fail loud instead of implicit CUDA→CPU fallback.
- 2026-08-27: `upscaling/seedvr2.py` is the dedicated SeedVR2 child runner. It resolves the deterministic upstream checkout and model directory, admits direct or bounded streaming execution from measured CUDA/MPS memory, terminates the active child on immediate task cancellation, and validates source-cardinality PNG output without a CPU fallback. Generation pipelines no longer call it.
- 2026-08-27: The dedicated use case preserves decoded source-frame timing through the timestamp-aware exporter. It verifies encoded frame cardinality, presentation offsets, terminal video duration, and stream-copied source audio before it publishes the final MP4 and JSON sidecar.

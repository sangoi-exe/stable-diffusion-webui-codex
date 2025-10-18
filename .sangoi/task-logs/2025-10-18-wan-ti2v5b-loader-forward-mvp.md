Task: WAN 2.2 TI2V‑5B loader + forward skeleton with presets/progress
Date: 2025-10-18

Changes
- Added `backend/engines/video/wan/loader.py` with `WanLoader` and `WanComponents`.
- Implemented `backend/engines/video/wan/ti2v5b_engine.py` forward skeleton:
  - Validates request via `parse_params` and applies presets (fill-only) with logs.
  - Emits progress events (prepare/denoise) with step% and VRAM when CUDA visible.
  - Raises explicit `EngineExecutionError` until generation is wired.
- Presets: added `configs/presets/wan_ti2v_5b/Normal/img2vid.yaml`.

Notes
- No network downloads; loader requires local model directory, e.g. `models/Wan/<dir>` or absolute path via `model_ref`.
- Validation is UI-first; smoke script optional.

Next
- Wire actual WAN denoise/generate path (text encoder → transformer → VAE) and return frames.
- Extend presets for other modes; add scheduler mapping table.
- Add optional mp4/webm assembly (guarded by ffmpeg) after frame generation.


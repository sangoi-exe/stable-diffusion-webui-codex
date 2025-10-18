Task: Wire WAN 2.2 TI2V‑5B generation via Diffusers (local‑only)
Date: 2025-10-18

Changes
- backend/engines/video/wan/loader.py: load Diffusers `WanPipeline` and `AutoencoderKLWan` when available, `local_files_only` unless `allow_download=True`. Stores `pipeline` in `WanComponents`.
- backend/engines/video/wan/ti2v5b_engine.py: txt2vid/img2vid now call `pipe(...)` with prompt/negative/num_frames/num_inference_steps/height/width/guidance_scale; returns frames and `info` JSON; emits prepare/run progress events; measures VRAM.

Constraints
- No auto‑download. The model folder must exist locally (e.g., `models/Wan/Wan2.2-TI2V-5B-Diffusers` or equivalent HF cache dir). If `WanPipeline` import fails (diffusers too old), engine raises `EngineExecutionError` with an actionable hint.

Next
- Map sampler/scheduler presets more precisely; optional video assembly; extend to T2V/I2V‑14B and Hunyuan/SVD.


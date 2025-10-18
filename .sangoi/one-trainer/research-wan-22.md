# Research — Wan 2.2 (video)

Date: 2025-10-18

Findings (ComfyUI docs and community references)
- Variants: TI2V 5B (text+image→video), T2V 14B (text→video), I2V 14B (image→video). Precisions: fp8/fp16/bf16 depending on release.
- Components: WAN transformer (diffusion), VAE, text encoder (UMT5-XXL), optional CLIP-Vision for some variants; schedulers and samplers exposed via custom nodes in Comfy.
- Typical controls: width/height (e.g., 768×432 baseline), steps, fps, num_frames, seed, guidance/cfg; image→video requires init image and denoise/motion controls.
- VRAM: 14B variants require high VRAM; 5B is lighter; mixed-precision recommended.
- File naming: releases often ship as safetensors pairs for experts (high/low noise) in some distributions.

Planned mapping for this repo
- Engines: `wan_ti2v_5b` (txt2vid/img2vid), `wan_t2v_14b` (txt2vid), `wan_i2v_14b` (img2vid).
- Requests: reuse Txt2VidRequest / Img2VidRequest fields; add `motion_strength` and pass engine_options for precision.
- Loader: future work to integrate WAN transformer + VAE + UMT5 stack; evaluate diffusers or native loader parity.

Notes
- This file captures high-level facts to guide integration; see ComfyUI docs for parameter-specific defaults per variant.

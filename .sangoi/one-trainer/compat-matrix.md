 # Compatibility Matrix — Planned Engines vs. External UIs

 Date: 2025-10-18

 Families (Image)
 - SD15 — planned engine `sd15` (txt2img, img2img, inpaint, upscale)
 - SD2.x — planned under `sd15` engine with variant configs (or separate `sd2` if needed)
 - SDXL (+Turbo) — engine `sdxl`
 - SD3/SD3.5 — engine `sd3`
 - Flux 1.* (Dev/Schnell/etc.) — engine `flux`
 - PixArt Sigma — engine `pixart`
 - AuraFlow — engine `auraflow`
 - HunyuanDiT — engine `hunyuan`
 - Lumina Image 2.0 — engine `lumina`
 - Stable Cascade — engine `cascade`

Families (Video)
- Stable Video Diffusion (SVD) — engine `svd` (img2vid)
- Hunyuan Video — engine `hunyuan_video` (txt2vid/img2vid)
- Genmo Mochi 1 — engine `mochi`
- Lightricks LTX Video — engine `ltx`
- NVIDIA Cosmos (Text2World/Video) — engine `cosmos`
- Wan 2.1/2.2 — engine `wan`  
  - `wan_ti2v_5b` — TI2V 5B (txt2vid/img2vid)
  - `wan_t2v_14b` — T2V 14B (txt2vid)
  - `wan_i2v_14b` — I2V 14B (img2vid)

 External references
 - ComfyUI: README lists SD 1.x/2.x/XL, Stable Cascade, SD3/3.5, PixArt Sigma, AuraFlow, HunyuanDiT, Flux, Lumina 2.0.
 - SwarmUI: releases/docs list Qwen Image (image), OmniGen 2, Flux Kontext, Wan 2.x, Mochi 1, LTX Video, Cosmos, Hunyuan Video, SVD, etc.

 Notes
 - Each engine implements only relevant tasks (methods can raise `NotImplementedError`).
 - Some families (e.g., SD2.x) may share code with SD15; split if complexity grows.
 - Presets must declare `engine_id` + `task` and optional `engine_options`.

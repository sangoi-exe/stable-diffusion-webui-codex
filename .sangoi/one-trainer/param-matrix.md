# Parameter Matrix — Required vs Optional (per Engine/Task)

Conventions
- Requests (protocol): Codex dataclasses — Txt2ImgRequest, Img2ImgRequest, Txt2VidRequest, Img2VidRequest.
- Engine selection: `engine_key` from Quicksettings (`codex_engine` dropdown) + `sd_model_checkpoint` as `model_ref`.
- Mode selection: `codex_mode` dropdown (`Normal`, `LCM`, `Turbo`, `Lightning`). Modo é enviado em `request.metadata.mode`; engines que não suportam devem ignorar e logar aviso.
- UI strict JSON: Builders in `javascript/ui.js` send the fields below; handlers reject if `__strict_version != 1`.
- LoRAs/extra networks: A1111/Forge tokens in prompt (`<lora:name:weight>` etc.).

Shared (all image engines)
- Optional (global): `seed` (int), `guidance_scale` (float), `batch_size` (int≥1), `loras` (list[str]), `extra_networks` (list[str]), `clip_skip` (int), `metadata` (dict), `extras` (dict).

SD15 — tasks: txt2img, img2img, inpaint, upscale
- txt2img (required)
  - prompt, width, height, steps, sampler (name), scheduler (name), seed, cfg_scale
  - UI keys: `txt2img_prompt`, `txt2img_width`, `txt2img_height`, `txt2img_steps`, `txt2img_sampling`, `txt2img_scheduler`, `txt2img_seed`, `txt2img_cfg_scale`
- txt2img (optional)
  - styles[], batch_count/batch_size (UI), highres_fix: { enable, denoise, scale, upscaler, steps, resize_x, resize_y, hr_checkpoint_name, hr_additional_modules[], hr_sampler_name, hr_scheduler, hr_prompt, hr_negative_prompt, hr_cfg }
  - UI keys incluem `txt2img_hr_*`, `txt2img_styles`. Observação: “Distilled CFG” não se aplica ao SD15 padrão; é específico de modelos destilados (ex.: Flux) e deve ser ignorado para SD15.
- img2img (required)
  - prompt, init_image, width/height, steps, sampler, scheduler, seed, cfg_scale, denoise_strength
  - UI keys: `img2img_prompt`, `img2img_height`, `img2img_width`, `img2img_steps`, `img2img_sampling`, `img2img_scheduler`, `img2img_seed`, `img2img_cfg_scale`, `img2img_denoising_strength`; image from active subtab
- img2img (optional)
  - image_cfg_scale, resize_mode, scale_by, selected_scale_tab, styles[], inpaint params (when applicable): mask, mask_blur, mask_alpha, inpainting_fill, inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert
- inpaint (required)
  - same as img2img + mask (derived from subtab)
- upscale (required)
  - same as txt2img + highres_fix block (used for the second pass)

SDXL — tasks: txt2img, img2img, inpaint, upscale
- Same matrix as SD15; engines may clamp/ignore some hires fields. Nota: “Distilled CFG” não é parte do SDXL padrão; só aparece em variantes destiladas específicas (ex.: SDXL Turbo/LCM‑style). Para nosso escopo, trate como não aplicável.

Flux — tasks: txt2img
- txt2img (required)
  - prompt, width, height, steps, sampler, scheduler, seed
- txt2img (optional)
  - guidance_scale (comumente ~1.0), distilled_cfg_scale (aplicável), hires fields ignorados

Video — common fields
- Required (txt2vid): prompt, width, height, fps, num_frames; (steps often required by engine), sampler/scheduler may be constrained; seed optional.
- Required (img2vid): init_image, width, height, fps, num_frames; (steps often required), sampler/scheduler; seed optional.
- Optional: motion_strength, guidance_scale, styles[], extras.

SVD — tasks: img2vid
- img2vid (required): init_image, width, height, num_frames, fps; steps; sampler/scheduler as supported.
- Optional: motion_strength, guidance_scale, seed.

Hunyuan Video — tasks: txt2vid, img2vid
- txt2vid (required): prompt, width, height, fps, num_frames; steps; sampler/scheduler.
- img2vid (required): init_image, width, height, fps, num_frames; steps; sampler/scheduler.
- Optional: motion_strength, guidance_scale, seed.

Wan 2.2 — variants
- wan_ti2v_5b (txt2vid, img2vid)
  - txt2vid (required): prompt, width, height, fps, num_frames; steps
  - img2vid (required): init_image, width, height, fps, num_frames; steps
  - optional: motion_strength, guidance_scale, seed
- wan_t2v_14b (txt2vid)
  - required: prompt, width, height, fps, num_frames; steps
  - optional: guidance_scale, seed
- wan_i2v_14b (img2vid)
  - required: init_image, width, height, fps, num_frames; steps
  - optional: motion_strength, guidance_scale, seed

Validation / Errors
- Handlers e engines rejeitam payload sem `__strict_version == 1` com mensagem detalhada (diagnóstico completo).
- Campos fora do suporte do engine são ignorados com log; faltantes geram erro com nome do campo e contexto (task/engine/model).

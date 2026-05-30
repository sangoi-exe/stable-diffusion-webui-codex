<!-- tags: frontend, utils, xyz -->
# apps/interface/src/utils Overview
Date: 2025-12-03
Last Review: 2026-05-30
Status: Active

## Purpose
- Small utility helpers shared across frontend modules (parsers, formatters, pure functions).

## Notes
- Keep helpers pure and framework-agnostic so manual validation remains deterministic.
- For utility changes, run `cd apps/interface && npm run typecheck` and validate impacted UI flows manually.
- 2026-04-08: `guidance_advanced.ts` is the canonical tiny helper for frontend advanced-guidance capability probes plus the Basic Parameters `Advanced` toggle patch (`enabled` + `apgEnabled`/`cfgTruncEnabled` when supported); keep row rendering in `components/AdvancedGuidanceFields.vue` and keep nested `guidanceAdvanced` ownership in the parent cards/views.
- 2026-04-08: `image_io.ts` is the canonical frontend helper for `File -> data URL` reads and image-dimension probes reused by `ImageModelTab.vue`, `VideoModelTabWanRuntime.vue`, `VideoModelTabLtxRuntime.vue`, `Upscale.vue`, and `PngInfo.vue`; do not reintroduce per-view `FileReader` / `Image()` helper clones on those live surfaces.
- 2026-04-08: `inpaint_mask_preview.ts` is the canonical pure helper for inpaint preview geometry and blur-spill raster math (mask bbox, blur-support expansion, masked-padding crop expansion, outward blur-spill alpha generation, display/storage invert-mask resolution, and tint packing) shared by `InitialImageBlock.vue` and `InpaintMaskEditorOverlay.vue`; thumbnail-space containment still belongs to `InitialImageCard.vue`, not this helper, and crop math requires explicit processing dimensions instead of hidden image-dimension fallback.
- 2026-03-21: `img2img_resize.ts` now owns engine-scoped truthful resize-mode subsets; unmasked ZImage img2img reuses the same helper for UI filtering and payload normalization, exposing only the pixel-space modes that `image_init.py` actually implements.
- 2026-03-20: Added `image_request_contract.ts` as the canonical pure helper for frontend image request selector/extras resolution; `useGeneration.ts` and `xyz.ts` must both reuse it for checkpoint metadata, FLUX.2 guidance mode, asset-contract-backed `tenc_sha`/`vae_sha`, `vae_source`, and ZImage variant extras instead of duplicating that logic. Qwen Image text-encoder labels must be `qwen_image/<path>` selections resolved through `qwen_image_tenc`; raw paths, basenames, SHA-only labels, and other family prefixes must fail before request payloads are emitted.
- 2026-05-30: `image_request_contract.ts` emits `extras.vae_source` for every asset contract with `uses_vae=true` (including built-in VAE for monolithic SDXL `.safetensors`) and still emits no VAE selectors for no-VAE engines such as `zimage_l2p`.
- 2025-12-03: Added XYZ helpers (`xyz.ts`) for axis parsing/combo building used by the sweep view/store.
- 2026-01-03: Added standardized file header block to `xyz.ts` (doc-only change; part of rollout).
- 2026-01-29: Added PNG infotext parsing + sampler/scheduler mapping helpers (`pnginfo.ts`).
- 2026-02-18: `pnginfo.ts` parser now tokenizes KV blocks safely (quoted/bracketed comma support), captures additional A1111-style fields (`RNG`, `Eta`, `NGMS`, `Version`, `Hires Module 1`), and supports legacy JSON `parameters` fallback parsing.
- 2026-03-07: `pnginfo.ts` sampler/scheduler mapping no longer rewrites `normal -> simple`; import mapping now preserves individually recognized global sampler/scheduler values while still rejecting incompatible sampler->allowed_scheduler pairs.
- 2026-03-07: `pnginfo.ts` warning text now distinguishes sampler-only vs scheduler-only failures so partial import patching stays technically honest.
- 2026-02-03: XYZ axis ids for hires are now `hires_scale` / `hires_steps`.
- 2026-02-08: XYZ keeps axis id `refiner_steps` for sweep compatibility, but the UI label now reflects swap-pointer semantics (`Swap at step`).
- 2026-02-06: Added `engine_taxonomy.ts` as canonical frontend engine taxonomy mapping (tab-family aliases, request engine-id resolution, semantic-engine normalization, and backend-map semantic-engine resolution). It must not own runtime exact-engine id lists or executable sampler/scheduler defaults.
- 2026-02-08: Added `image_params.ts` for pure img2img/inpaint normalization helpers (`normalizeMaskEnforcement`, `normalizeInpaintingFill`, `normalizeNonNegativeInt`, `normalizeInpaintMaskToggleState`) used by `ImageModelTab.vue`.
- 2026-04-06: `image_params.parseInpaintMode(...)` is strict: it accepts only the live `inpaintMode` enum and returns `null` for stale/removed values so callers can reset or reject explicitly instead of laundering old modes.
- 2026-03-06: `image_params.ts` now also exposes hires policy/text helpers (`resolveTextOverride`, `resolveHiresModePolicy`) used by payload/view wiring to keep prompt fallback deterministic and hires visibility/reset mask-aware for img2img.
- 2026-04-05: `image_params.ts` now also owns `buildUseInitImagePatch(...)`, the canonical pure cleanup patch for disabling init-image mode; `QuickSettingsBar.vue` and `PngInfo.vue` must both reuse it so txt2img handoffs clear init+mask state together instead of drifting.
- 2026-02-18: Added `img2img_resize.ts` as the canonical img2img resize-mode contract (`just_resize`, `crop_and_resize`, `resize_and_fill`, `just_resize_latent_upscale`, `upscaler`) with a shared normalizer for store/view/composable wiring.
- 2026-02-22: Added `wan_img2vid_temporal.ts` as the canonical WAN img2vid temporal helper module (`solo|chunk|sliding|svi2|svi2_pro` mode normalization + window stride/commit normalizers with `stride % 4 == 0` and `commit - stride >= 4` guards) reused by store/view/payload layers.
- 2026-02-28: `wan_img2vid_temporal.ts` removed `chunk` from `WanImg2VidMode`; normalization is now fail-loud for `img2vid_mode='chunk'` and any unsupported value (`solo|sliding|svi2|svi2_pro` only).
- 2026-03-01: Added `wan_img2vid_frame_projection.ts` as the canonical WAN no-stretch frame-projection helper (free `imageScale > 0` + normalized crop offsets), including deterministic crop rect, slack bounds for drag clamping, and scaled-dimension metadata for init-image zoom overlays.
- 2026-03-05: `engine_taxonomy.ts` now includes first-class `flux2` semantic/request ids (no `flux1`/`flux1_kontext` alias), matching the backend FLUX.2 Klein 4B / base-4B slice, and `model_family_filters.ts` maps `flux2` to `flux2_ckpt` roots.
- 2026-03-12: `engine_taxonomy.ts` now includes first-class `ltx2` tab/semantic/request ids; `resolveImageRequestEngineId(...)` explicitly rejects `ltx2` because image routes are not valid for the LTX video lane, and `model_family_filters.ts` maps `ltx2` to `ltx2_ckpt`/`ltx2_vae` style roots plus `ltx` filename fallbacks.
- 2026-03-31: `engine_taxonomy.ts` stays scoped to tab-family aliases, request-engine resolution, semantic-engine normalization, and backend-map semantic-engine resolution; mask/inpaint support and executable sampler/scheduler defaults are owned by backend `/api/engines/capabilities`, not by frontend taxonomy helpers.
- 2026-05-17: `engine_taxonomy.ts` treats `qwen_image` as the single canonical Qwen Image tab, semantic, and image-request id. Do not add variant public ids for `Qwen-Image-2512`, `Qwen-Image-Edit-2511`, or `Qwen-Image-2.0`; those remain backend/runtime variant details.
- 2026-05-23: `engine_taxonomy.ts` treats `zimage_l2p` as a first-class image tab/semantic/request id, separate from latent `zimage`. `image_request_contract.ts` is the canonical no-VAE L2P selector owner: it accepts exactly one shared `zimage/<path>` Qwen3-4B label resolved through `zimage_tenc`, requires `checkpoint_core_only=true` with `checkpoint|gguf` model format, and must not emit VAE selectors for L2P.

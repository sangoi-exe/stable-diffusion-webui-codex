# Model Registry (Work in Progress)
Date: 2025-10-28
Last Review: 2026-05-24
Status: Draft

## Purpose
- Track structured metadata about supported checkpoints and pipelines.
- Provide detection heuristics and signatures for model loading without relying on `huggingface_guess`.

## Current Status
- Core dataclasses/enums (now `CodexCoreSignature`/`CodexCoreArchitecture`) in place with manifest-driven metadata harvesting.
- Detectors implemented for SD1.x, SDXL (base/refiner), Flux.1 (dev/schnell), FLUX.2 Klein 4B/base-4B core-only SafeTensors, LTX2 monolithic combined checkpoints, AuraFlow, SD3 / SD3.5 (medium & large families), Stable Cascade (B/C), Wan2.2 (T2V/I2V), Chroma, Qwen Image, and Anima (Cosmos Predict2 core `net.*` format).
- `capabilities.py` defines `SemanticEngine` and an `EngineParamSurface` describing which high-level UI parameter sections (txt2img/img2img/video/hires/refiner/LoRA/ControlNet/masked-img2img) are expected to be used for each semantic engine tag; exposed to the API for frontend gating.
- `capabilities.py` also owns `CAPABILITY_ONLY_EXACT_ENGINES`. Capability-only exact ids may appear only in capability/taxonomy maps and must not become registered engines, parked engines, aliases, facades, asset-contract ids, live-preview ids, runtime-family ids, or runtime dispatch ids.
- `capabilities.py` owns parked exact-engine truth for `lens` while Lens runtime is absent. Do not add `SemanticEngine.LENS`, runnable `ENGINE_SURFACES`, `ENGINE_ID_TO_SEMANTIC_ENGINE`, `_ENGINE_ID_PRIMARY_FAMILY`, or asset-contract rows until a runtime tranche explicitly un-parks Lens.
- 2025-12-14: `ModelSignature` gained a legacy `unet` alias for `core`, keeping older call sites working while the new contract stays `signature.core`.
- 2025-12-14: Qwen Image detector reintroduced (`detectors/qwen_image.py`) and enums extended (`ModelFamily.QWEN_IMAGE`, `LatentFormat.QWEN_IMAGE`).
- 2026-05-17: Qwen Image capability/runtime metadata now treats `qwen_image` as the single architecture family; `Qwen-Image-2512` and `Qwen-Image-Edit-2511` are internal variants, not separate engine or family ids.
- 2026-05-23: Z-Image L2P GGUF denoiser discovery requires both native L2P tensor-key evidence and dedicated Codex profile metadata (`codex.zimage_l2p.profile_id=zimage_l2p_denoiser`, component `denoiser`, family `zimage_l2p`, pixel-space/no-VAE facts). SafeTensors L2P discovery remains header-key based.
- 2025-12-12: Z Image runtime metadata was corrected (`context_dim=2560`, `flow_shift=3.0`) to match the canonical HF assets for Z-Image Turbo.
- 2025-12-13: Z Image Turbo default steps adjusted to 9 to match diffusers `ZImagePipeline` recommendation (≈8 effective updates; last `dt=0`).
- 2025-12-14: `ModelFamily.ZIMAGE.flow_shift` re-aligned to `3.0` in `family_runtime.py` (HF scheduler_config parity).
- 2026-01-28: Z Image semantic surface now declares `supports_img2img=true`; Z-Image Turbo/Base flow shift is treated as variant-specific (`shift=3.0` / `shift=6.0`) and resolved from vendored diffusers scheduler configs.
- 2026-01-06: Engine capability surfaces now default to model_index-derived sampler/scheduler pairs only where that default remains executable in the live contract. Current public examples: SD15 `ddim`/`ddim`, SDXL `euler`/`euler_discrete`, WAN22 `uni-pc`/`simple`; Hunyuan Video currently leaves sampler/scheduler defaults unset until its live execution contract is truthfully wired.
- 2026-01-08: Added `flow_shift.py` as the canonical flow-shift resolver from diffusers `scheduler_config.json` (fixed + dynamic) and removed hard-coded `flow_shift` values from family runtime specs where the value is not a true family invariant (Flux/WAN22).
- 2026-01-08: Refreshed file header blocks for `capabilities.py` and `flow_shift.py` to keep the Symbols lists in sync (doc-only change).
- 2026-01-18: Semantic engine surface for `chroma` now declares `supports_img2img=true` to match the registered `flux1_chroma` engine task surface.
- 2026-02-06: `SemanticEngine.ANIMA` capability surface now exposes `supports_txt2img=true` and `supports_img2img=true` after conditioning payload port (`crossattn` + `t5xxl_ids/t5xxl_weights/t5xxl_attention_mask`; no synthesized pooled `vector`) and compile/sampler pass-through validation.
- 2026-02-07: `SemanticEngine.ANIMA` sampler surface narrowed to native-supported entries (`euler`, `euler a`, `dpm++ 2m`) with default `euler`.
- 2026-02-08: `SemanticEngine.ANIMA` re-enabled native `er sde` rollout (Anima-only release scope) while keeping default sampler `euler`.
- 2026-02-16: WAN22 family bucket was split into explicit families (`WAN22_5B`, `WAN22_14B`, `WAN22_ANIMATE`); detector family assignment now follows model type (`ti2v` -> 5B, `animate` -> WAN22_ANIMATE, others -> 14B).
- 2026-02-18: `EngineParamSurface` now includes optional `guidance_advanced` per-engine support flags (`apg_*`, `guidance_rescale`, `cfg_trunc_ratio`, `renorm_cfg`) so frontend CFG-Advanced UI can be gated by backend contract (including partial per-control support).
- 2026-02-20: Removed semantic-engine map entry for `wan22_14b_native`; WAN22 14B semantic ownership is now single-key (`wan22_14b`).
- 2026-02-20: WAN22 animate semantic/loader key was renamed to `wan22_14b_animate` (old `wan22_animate_14b` removed).
- 2026-04-06: `capabilities.py` now also owns `EXACT_ENGINE_INPAINT_MODES`, the exact-engine img2img inpaint-mode map exposed through `/api/engines/capabilities`. Keep SDXL-only modes like `fooocus_inpaint` and `brushnet` off semantic-engine truth so they do not leak onto `sdxl_refiner` or unrelated engines.
- 2026-04-03: Added the metadata-only `netflix_void_execution.py` seam as the sole public owner for Netflix VOID checkpoint-kind classification (`pass1|pass2|unknown`) and literal sibling pair readiness. Runtime/engine assembly must consume that metadata instead of introducing a second detector/alias lane.
- 2026-04-03: Added `ModelFamily.NETFLIX_VOID` plus a family runtime spec so primary-family capability lookups stay fail-loud once the staged `netflix_void` engine is registered. The semantic engine surface remains discovery-only until `supports_vid2vid=true` and the native runtime cutover land together.
- 2026-02-21: `capabilities.engine_supports_cfg(engine_id)` now resolves family from explicit `engine_id -> ModelFamily` mapping (including WAN22 variants) instead of semantic-primary-family fallback, removing hidden `wan22 -> WAN22_5B` drift at capability checks.
- 2026-03-28: `capabilities.primary_family_for_engine_id(engine_id)` is the canonical runtime owner for exact same-family checks. `SemanticEngine` remains UI/workflow gating only and must not be reused as proof for strict family-equality runtime contracts such as exact top-level `swap_model` resume.
- 2026-03-31: `EngineParamSurface` now includes explicit `supports_img2img_masking`; frontend img2img/inpaint gating and the `/api/img2img` router must consume that backend semantic-engine truth instead of carrying local engine-id blocklists. Z-Image is a live masked-img2img exception again, so future flow-family cleanup must not flatten it back into a generic `supports_img2img_masking=false` bucket without proving the canonical masked owner path is actually broken.
- 2026-03-02: `capabilities.py` semantic surfaces now declare `supports_hires=true` for non-SD image families implemented in the shared hires second-pass (`flux1`, `flux1_chroma`, `zimage`, `anima`).
- 2026-03-04: Detector `SignalBundle` now carries source-format hints (`safetensors|gguf`) and header-backed `shape_of(...)` usage, so safetensors signature detection avoids full `state_dict.values()` scans/materialization during startup planning.
- 2026-03-05: Added `detectors/flux2.py` for the truthful FLUX.2 Klein 4B/base-4B core-only SafeTensors layout (`double_blocks.*`, `single_blocks.*`, `img_in.*`, `txt_in.*`, `single_stream_modulation.*`, `final_layer.*`); unsupported FLUX.2 variants (for example 9B) intentionally do not match.
- 2026-03-06: FLUX.2 runtime/capability metadata is now a single truthful family entry: `latent_channels=32`, scalar-identity FLUX.2 VAE normalization, `patch_size=2`, `flow_shift=None`, `context_dim=7680`, and a semantic surface for the supported Klein 4B/base-4B slice (`supports_txt2img=true`, `supports_img2img=true`, `supports_hires=true`, `supports_lora=false`, samplers `euler|dpm++ 2m`, scheduler `simple`). This capability surface matches the live FLUX.2 slice: masked img2img/inpaint is active, partial denoise is supported, and hires is supported for unmasked runs.
- 2026-03-07: `EngineParamSurface` sampler/scheduler capability fields were renamed to `recommended_samplers` / `recommended_schedulers`; these fields are UI recommendation hints (not engine-level filtering lists). Defaults and recommendations must still stay executable and scheduler-compatible for the live engine path.
- 2026-03-08: WAN22 semantic capabilities expose recommendation hints as `recommended_samplers=('uni-pc bh2', 'uni-pc', 'euler', 'euler a')` and `recommended_schedulers=('simple',)`, with defaults `default_sampler='uni-pc bh2'` and `default_scheduler='simple'`.
- 2026-03-07: WAN22 detector now recognizes the current local I2V base surface by `patch_embedding.weight` input channels (`C_in=36` for the 36-channel concat I2V path) instead of relying only on upstream `img_emb.proj.*` keys, which are absent from the current local WAN22 GGUF base artifacts.
- 2026-03-23: Strict monolithic LTX2 detection (`detectors/ltx2.py`) and parser dispatch/planner support (`model_parser/families/ltx2.py`) are now part of the live advertised `txt2vid` / `img2vid` slice. The detector accepts connector evidence only under `model.diffusion_model.*`, keeps checkpoint provenance truthful (`repo_hint=None` when filepath evidence is absent), and must stay aligned with the live semantic capability surface.
- 2026-03-23: LTX2 model inventory/capability metadata now includes a checkpoint-aware execution/default seam. `models/registry.py` forwards namespaced per-checkpoint metadata (`ltx_checkpoint_kind`, allowed/default execution profiles, default steps, default guidance), and `capabilities.py` exposes one nested LTX-only `ltx_execution_surface` object under `/api/engines/capabilities`. `unknown` LTX checkpoints stay blocked in this tranche instead of being guessed into executable kinds.
- 2026-03-26: LTX2 `two_stage` side-asset admissibility stays exact-file based, but the x2 spatial upsampler may live under either `ltx2_ckpt` or `ltx2_connectors` roots in current local layouts; `ltx2_execution.py` must search only those explicit sanctioned roots and never widen into generic filename heuristics.
- 2026-03-26: LTX2 family-hint recovery remains path-root aware, but checkpoint blocking must still reject exact side-asset path tokens (`lora|loras|upscaler|upscalers`) without substring soup; `flora`-style names must not false-block, and basename-only blocking is too weak for nested side-asset subtrees.

## TODO
- Add detectors for remaining launch families (KOALA, StableAudio, WAN22 camera/S2V/animate, Chroma Radiance).
- Extend Flux.1 detection to cover additional GGUF layouts when they appear; the current `FluxCoreGGUFDetector` targets core-only Flux.1 transformers (double_blocks.+guidance) with external TEnc/VAE.
- Expose CLI/inspect tooling for diagnostics.
- Wire registry outputs into loader/runtime paths and add regression fixtures.
- 2026-01-02: Added standardized file header docstrings to model registry modules and detectors (doc-only change; part of rollout).
- 2026-01-06: Updated Flux sampler recommendation lists in `capabilities.py` to use canonical `SamplerKind` strings (spaces/`++` preserved).

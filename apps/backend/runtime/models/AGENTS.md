# Runtime Models — AGENTS Notes
<!-- tags: runtime, models, loader, prediction -->
Date: 2025-12-05
Last Review: 2026-05-17
Status: Active

## Scope
Applies to `apps/backend/runtime/models/*` including `loader.py`, `registry.py`, and state-dict helpers.

## CLIP (TE) State‑Dict Keyspace Interpretation
- Goal: accept common WebUI-style and Diffusers-style layouts without guessing external context.
- Accepted inputs:
  - OpenCLIP legacy: `transformer.resblocks.*` (converted to `transformer.text_model.encoder.layers.*`).
  - Plain modern: `text_model.*` at root (lifted to `transformer.text_model.*`).
  - Aliased/wrapped surfaces: `clip_[lgh].transformer.text_model.*`, `conditioner.embedders.*`, `cond_stage_model.*`, `model.*` — these wrappers must be understood explicitly by the keymap; runtime key rewriting is forbidden.
- Policy:
  - Interpret known wrapper/container surfaces explicitly per key before conversion; do not rewrite stored keys.
  - Always attempt Codex converters (`convert_sdxl_clip_*`, fallback to `convert_sd20_clip`, `convert_sd15_clip`); treat success as “essential tensors present”.
  - Lift `text_model.*` into `transformer.text_model.*`, map `text_projection` into `transformer.text_projection.weight`, and forward `final_layer_norm.*` similarly.
  - Drop HF-only buffers (`*.position_ids`) and canonicalize `logit_scale` into the `IntegratedCLIP` keyspace (no `transformer.*` aliases); when the source omits it, keep the canonical `ln(100)` default instead of failing the load.
  - Abort with a `RuntimeError` when essential tensors (`token_embedding`, `position_embedding`, first-layer q_proj, `final_layer_norm`) remain missing after keyspace resolution — no silent degradation.

## UNet State‑Dict Keyspace Interpretation
- Accepted inputs:
  - LDM layout: keys already under `input_blocks./middle_block./output_blocks.` — forwarded untouched.
  - Diffusers layout: `conv_in`, `down_blocks.*`, `mid_block.*`, `up_blocks.*`, `time_embedding.*` — converted per config (`num_res_blocks`, `channel_mult`, transformer depths) using the shared UNet conversion map (`unet_to_diffusers`).
- Policy:
  - Any shared/generic path that strips wrappers like `model.diffusion_model.` outside the owning keymap is invalid and must fail loud until replaced with explicit keyspace interpretation.
  - Build diffusers→LDM keyspace lookup programmatically from the UNet config and load through source-key resolution.
  - Preserve optional leftovers (logged at DEBUG) and drop `logit_scale`-style noise.
  - Guard against missing essentials (`input_blocks.0.0.weight`, `time_embed.0.weight`, `out.2.weight`) by raising a `RuntimeError` with representative diffusers keys.

## Error Handling
- Missing/Unexpected above thresholds will be escalated by the loader; we do not degrade silently.
- SDXL: UNet/VAE/CLIP loads are strict — any missing/unexpected keys are fatal.
- Prefer clear messages naming a few representative keys and the active keyspace-resolution path.
- `safe_load_state_dict(...)` may suppress caller-declared allowed-missing prefixes only for already-proven staged partial-load seams; it must not be used to hide cross-keyspace incompatibility, wrapper-prefix drift, or renamed stored keys.

## Rationale
- Keyspace interpretation converges on Diffusers-style text encoder keys: converts legacy resblocks to `text_model.*` and accepts plain `text_model.*` roots without treating key rewriting as an allowed runtime operation.

## Updates
- 2026-05-23: `registry.py` preserves route-required Z-Image L2P denoiser GGUF metadata (`model.architecture`, profile/component/family, pixel/no-VAE, local-decoder, context-dim) in checkpoint inventory records after header validation so API preflight revalidation does not lose the required architecture key.
- 2026-03-21: `loader.py` now validates that VAE state dicts do not require wrapper-prefix rewriting before layout detection/lane resolution; any such path is now an explicit stop-ship failure instead of a silent key rewrite seam.
- 2026-03-25: `loader.py` now resolves SDXL diffusers base vs refiner truth from native `model_index.json` + `unet/config.json` evidence (no Diffusers config helper), expected-family SDXL checkpoint parsing can mark UNet-only checkpoints as `core_only`, SDXL CLIP slots no longer retype T5/CLIP payloads across family boundaries, and `ModelRegistry` marks SDXL UNet-only `.safetensors` checkpoints as `core_only` via header-only detection.
- 2026-03-20: `loader.py` no longer exports/consumes a loader-side `inpaint_model()` checkpoint heuristic; image runtime behavior must come from explicit request selectors plus validated inventory metadata, not channel-count guesses.
- 2025-11-22: VAE selection now prefers diffusers `AutoencoderKL` on diffusers-native layouts; native `AutoencoderKL_LDM` is selected only on native-LDM lane resolution (WAN22 constrained, other supported families policy-driven).
- 2025-11-23: VAE loader now fails fast when weights are missing (e.g., pruned checkpoints without VAE); error names missing key count and asks for a compatible VAE.
- 2025-11-23: VAE loader logs missing/expected/unexpected key counts before raising, making “frame cinza” cases debuggable when a single safetensors lacks VAE tensors.
- 2025-11-23: `_resolve_vae_class` no longer uses an implicit WAN-family fallback path for non‑WAN22 loads; class selection follows resolved lane policy (including native-LDM for supported non‑WAN22 families), while layout detection remains a key-mapping signal.
- 2025-11-24: `_maybe_convert_sdxl_vae_state_dict` now materialises lazy SafeTensors views before reshaping mid-attn projections to avoid torch_cpu.dll crashes on Windows during SDXL VAE conversion.
- 2025-12-11: `_maybe_convert_sdxl_vae_state_dict` expanded to cover `ModelFamily.ZIMAGE`, since Z Image uses the same Flow16 VAE layouts as Flux; external VAEs loaded via `runtime.common.vae.load_flow16_vae` reuse this converter.
- 2025-11-25: Loader now preserves scheduler-provided `prediction_type` when it disagrees with the model signature, logging the mismatch instead of forcing the signature value; the signature hint remains accessible via `scheduler.config.codex_signature_prediction_type`.
- 2025-12-04: `ModelRegistry` checkpoint discovery agora usa `apps/paths.json["checkpoints"]` como override primário, com fallbacks curados em `models/` (`sd15`, `sdxl`, `flux`) em vez de varrer múltiplas pastas legacy (`stable-diffusion`, `sd`, `checkpoints`).
- 2025-12-05: Text encoder overrides are resolved centrally by the loader using `TextEncoderOverrideConfig` + `resolve_text_encoder_override_paths` (now in `runtime.models.text_encoder_overrides`), mapping `(family, <family>/<path> label from paths.json, ModelSignature.text_encoders)` to per-component weights. Overrides fail fast when families mismatch, labels are unknown, or expected `<alias>.(safetensors|gguf|bin|pt)` files are missing under the configured root.
- 2025-12-05: Flux GGUF core-only checkpoints (signalled via `ModelSignature.extras["gguf_core_only"]`) now compose with an external VAE resolved from `apps/paths.json["flux1_vae"]`; `_load_flux_vae_state_dict()` scans configured roots for a suitable VAE weights file and fails fast with an explicit error when nothing usable is found, instead of silently running Flux without a VAE.
- 2025-12-06: `TextEncoderOverrideConfig` gained an `explicit_paths` map (`alias -> abs path`) for file-level overrides (e.g., Flux); `resolve_text_encoder_override_paths` supports two modes: explicit path mapping (skipping root lookup) and root-based lookup. In both cases, aliases are validated against `CodexEstimatedConfig.text_encoder_map`, and missing files or unsupported extensions raise `TextEncoderOverrideError` with clear messages.
- 2025-12-30: `apps/backend/runtime/models/__init__.py` switched back to lazy exports (no eager `import safety` / wildcard imports) so `create_api_app` and lightweight torch-stub validation paths can import the API.
- 2025-12-30: Text encoder overrides now accept `.gguf` weights; GGUF-packed state dicts are detected so T5 text encoders can load via the `"gguf"` quant path.
- 2026-01-01: `ModelRegistry` checkpoint discovery now lists only file-based weights under `*_ckpt` roots (`.ckpt/.safetensors/.safetensor/.gguf/...`); it no longer treats vendored Hugging Face metadata folders as selectable checkpoints.
- 2026-01-02: `runtime.models.api` gained `find_checkpoint_by_sha(...)` so API layers can resolve checkpoints from short-hash/sha256 identifiers (backed by `models/.hashes.json`).
- 2026-01-04: `ModelRegistry` now exposes public `hash_for(...)` + `flush_hash_cache()` so inventory and other subsystems can request hashes without importing private cache internals.
- 2026-01-06: VAE selection is expressed via engine options (`vae_source` + `vae_path`); the loader does not persist a separate `external_vae_path` metadata key.
- 2026-01-06: Loader now supports `tenc_path` (string or ordered list) as a shorthand for text encoder overrides: paths are mapped onto `ModelSignature.text_encoders` in order and loaded via the existing `TextEncoderOverrideConfig` pipeline (fail-fast on count/alias mismatch).
- 2026-01-06: Refreshed `loader.py` header block to document `tenc_path` shorthand semantics (doc-only change).
- 2026-01-02: Added standardized file header docstrings to `__init__.py`, `api.py`, and `types.py` (doc-only change; part of rollout).
- 2026-01-08: Split state-dict key normalization helpers into `key_normalization.py` and reused them from `loader.py` (UNet keyspace resolution + transformer prefix stripping).
- 2026-01-08: Moved text-encoder override definitions into `text_encoder_overrides.py`; loader now imports the shared config + resolver from that module.
- 2026-01-14: Flux expected-family loads now use vendored HF metadata to build the signature (selecting `FLUX.1-dev` vs `FLUX.1-schnell` by guidance key presence), avoiding registry detection failures on prefixed Flux checkpoints.
- 2026-01-18: `CheckpointRecord` now includes `core_only`, `core_only_reason` (e.g. `gguf_suffix`, `gguf_magic`), and optional `family_hint`; `/api/models` surfaces these so UIs stop guessing core-only status by suffix alone.
- 2026-01-18: `loader.py` now lazily imports `diffusers`/`transformers` (keeps `create_api_app` import-light for health/models endpoints and torch-stub validation paths).
- 2026-01-25: SDXL loads are strict on missing/unexpected keys (fail loud); CLIP normalization now drops `position_ids`, canonicalizes `logit_scale` (synthesizing the canonical `ln(100)` default when omitted), and keeps only `transformer.text_projection.weight`.
- 2026-01-25: Loader dtype selection no longer overrides memory-manager role defaults using a whole-file SafeTensors “primary dtype” guess; the hint is now debug-only (prevents TE bf16 vs UNet fp16 drift under AUTO).
- 2026-01-25: SDXL/Flow16 VAE key normalization now lives in `apps/backend/runtime/state_dict/keymap_sdxl_vae.py`; `_maybe_convert_sdxl_vae_state_dict` delegates to the keymap (single source of truth) and drops `model_ema.decay` / `model_ema.num_updates` metadata keys.
- 2026-01-25: SDXL base CLIP loads now reuse `apps/backend/runtime/state_dict/keymap_sdxl_clip.py` (HF/OpenCLIP → Codex IntegratedCLIP), including lazy in-proj QKV slicing and projection canonicalization.
- 2026-02-05: Anima minimal-bundle metadata now canonicalizes `tenc_path` to the single resolved external text-encoder path (exactly one required) instead of preserving raw override payload shapes; both `vae_path` and resolved `tenc_path` are checked for on-disk existence with fail-loud diagnostics.
- 2026-02-05: `registry.py` discovery roots now include Anima (`anima_ckpt`, `anima_vae`, and default `models/anima`) so checkpoint/vae inventory and family hints align with the Anima model directory layout.
- 2026-02-08: Fixed Flux GGUF T5 loader unbound-local bug in `_load_huggingface_component`: `IntegratedT5` construction now always executes before `load_state_dict(...)` for both GGUF and non-GGUF paths.
- 2026-02-09: `codex_loader(...)` now uses `torch.no_grad()` (not `torch.inference_mode()`), preventing inference-tensor parameters from being created during model assembly/load in long-lived WebUI processes.
- 2026-02-09: Smart-offload TE load policy now keeps text-encoder load-on-conditioning-device semantics in the loader (no forced CPU staging at initial load). TE offload sequencing remains conditioning-owned: encoders are unloaded after embeddings are generated, while Smart Cache hit paths can skip TE execution entirely.
- 2026-02-10: Removed ad-hoc T5 inline key normalization from `loader.py`; `_load_huggingface_component` now delegates T5 key-style resolution to canonical state-dict module `apps/backend/runtime/state_dict/keymap_t5_text_encoder.py`.
- 2026-02-10: `_parse_checkpoint` now routes original-format SDXL checkpoints through the canonical checkpoint keymap, which explicitly maps known wrapper/container source styles (including nested UNet label keys like `model.diffusion_model.label_emb.0.<idx>.*`) into parser lookup keys; parser-side alias collapse is gone.
- 2026-02-10: Structural conversion paths in runtime model helpers are globally policy-gated by `CODEX_WEIGHT_STRUCTURAL_CONVERSION`: `transformers_convert` (fused `in_proj` split) and CLIP projection transpose now fail loud in `auto`, allowing conversion only with explicit `convert` opt-in.
- 2026-02-11: `registry.py` cache schema is now versioned (`schema_version=2`) with canonical SHA layout metadata (`layout_by_sha`) for CLIP layout reuse (`qkv_layout` + `projection_orientation` + optional `source_style`), plus conflict/unknown-schema fail-loud guards.
- 2026-02-11: `loader.py` CLIP paths are layout-aware and cache-first: AUTO consumes/persists SHA layout decisions, explicit SDXL QKV overrides bypass cache reads/writes, and projection module orientation (`linear` vs `matmul`) is selected at model-construction time (no AUTO tensor transpose/split/concat).
- 2026-02-11: `clip_key_normalization.py` now delegates to canonical keymap ownership (`state_dict/keymap_sdxl_clip.py`), and the legacy entrypoint `normalize_codex_clip_state_dict_with_layout(...)` returns resolved layout metadata plus a lazy canonical lookup view; it does not rewrite stored keys.
- 2026-02-11: `_maybe_convert_sdxl_vae_state_dict` now preflights canonical SDXL mid-attention projection keys (`to_q/to_k/to_v/to_out.0`) immediately after keyspace resolution so lane/shape contract violations are raised explicitly before strict-load missing accounting.
- 2026-02-11: SDXL VAE loader now detects canonical projection lane (`linear_2d` vs `conv1x1_4d`) and applies native 4D replacement modules (`_Conv1x1Projection`) on mid-block attentions for the conv lane, so 4D checkpoints load without keymap flattening.
- 2026-02-11: Native 4D projection replacement accepts both plain `torch.nn.Linear` and Codex-op patched linear-like modules (inside `using_codex_operations`) to avoid loader-path type mismatch.
- 2026-02-11: SDXL/SDXL refiner loader path no longer injects shift-factor sanitization; SDXL no-shift compliance is now enforced at source by native LDM VAE config emission (`AutoencoderKL_LDM` defaults to `shift_factor=None`).
- 2026-02-11: Flux core-only GGUF path now has explicit VAE causality guards: rejects `vae_path` equal to the model checkpoint path and validates Flux VAE state_dict keyspace (`encoder.`/`decoder.`) before strict load, with family-specific error guidance when missing keys occur.
- 2026-02-16: WAN22 family handling is explicit by variant in `FAMILY_TO_ENGINE_KEY` (`WAN22_5B -> wan22_5b`, `WAN22_14B -> wan22_14b`, `WAN22_ANIMATE -> wan22_14b_animate`), with `wan22_14b` resolved through its dedicated 14B GGUF engine lane by default registration.
- 2026-02-18: GGUF core-loader smart-offload CPU staging now emits canonical INFO audit events via `log_smart_offload_action("cpu_stage_load", ...)` under `backend.smart_offload` (in addition to local loader diagnostics).
- 2026-02-18: `loader.py` now emits CPU-staging events with `SmartOffloadAction.CPU_STAGE_LOAD`; generic smart-offload `load/unload` ownership remains centralized in `runtime/memory/manager.py`.
- 2026-02-23: GGUF smart-stage construction in `loader.py` now resolves stage-load target from memory-manager offload policy (`offload_device`) instead of forcing a local CPU literal; logs/actions now report the resolved offload device explicitly.
- 2026-03-03: `registry.py` checkpoint discovery (`_iter_checkpoint_files`) no longer swallows `paths.json` resolution failures (fail-loud path config behavior) and now accepts file-level checkpoint entries in `*_ckpt` keys in addition to recursive directory roots.
- 2026-03-04: `loader.py` startup parsing is now inspection-first for safetensors (`checkpoint_inspect_start|done`), with explicit non-safetensors materialization events (`checkpoint_materialize_start|done`) and no eager root-denoiser tensor materialization during safetensors planning.
- 2026-03-05: `registry.py` discovery roots now include Flux.2 path keys (`flux2_ckpt`, `flux2_vae`) and `family_hint` now recognizes `models/flux2/*` prefix as `flux2`.
- 2026-03-05: `loader.py` now supports the truthful FLUX.2 Klein 4B/base-4B slice: expected-family loads build vendored-HF signatures for `FLUX.2-klein-4B` / `FLUX.2-klein-base-4B`, parser-validated core-only checkpoints load through `diffusers.Flux2Transformer2DModel`, external `Qwen3ForCausalLM` overrides reuse the native ZImage Qwen3-4B wrapper, `AutoencoderKLFlux2` is used for FLUX.2 VAEs, and diffusers repo detection rejects unsupported non-Klein / non-4B configs fail-loud.
- 2026-03-06: `_parse_checkpoint(...)` now applies family-scoped GGUF keyspace interpretation for expected-family Flux / FLUX.2 / Z Image loads before parser execution, so native/source keys (and supported fused legacy slices) are presented through lazy lookup views instead of remapped dicts. The GGUF core load path also passes mappings directly to `nn.Module.load_state_dict(...)` instead of materializing `dict(state_dict)`.
- 2026-03-11: `loader.py` now turns `ModelFamily.LTX2` parser output into a typed minimal bundle-planning contract (parser-owned `transformer` / `connectors` / `vae` / `audio_vae` / `vocoder` + exactly one external `gemma3_12b` asset + vendored tokenizer/config metadata). Engine registration/runtime execution still lands separately; generic diffusers component assembly remains forbidden for LTX2.
- 2026-03-12: `registry.py` discovery roots now include `ltx2_ckpt` and `ltx2_vae`, `family_hint` recognizes `models/ltx2/*`, and generic VAE inventory excludes `audio_vae` bundle filenames so the split LTX 2.3 package does not surface the audio bundle as a selectable video VAE.
- 2026-03-12: `loader.py` now normalizes the single resolved LTX2 external text-encoder path into the fixed `gemma3_12b` slot before bundle planning, writes the rewritten five-component core-only GGUF bundle map into `_build_diffusion_bundle(...)`, and fail-loud checks that `estimated_config.components` stayed aligned with that rewritten contract.
- 2026-07-30: `loader.py` admits Qwen Image only as the exact Edit-2511 core transformer GGUF plus external Qwen2.5-VL GGUF and VAE SafeTensors. Vendored HF metadata directories are validation-only and fail as executable model selections.

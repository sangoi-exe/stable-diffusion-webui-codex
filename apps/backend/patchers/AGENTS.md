# apps/backend/patchers Overview
Date: 2025-10-30
Last Review: 2026-08-01
Status: Active

## Purpose
- Hosts runtime patching utilities (LoRA injection, adapter application) that modify networks or inference behavior after models are loaded.

## Key Files
- `base.py` — Core `ModelPatcher` with typed registries (LoRA/object patches), explicit streamed-core runtime registration, and full/streamed lifecycle hooks.
- `lora.py` — Public LoRA facade (re-exports loader/merge/state-dict helpers).
- `lora_loader.py` — `CodexLoraLoader` transactional applier (backups, GGUF re-quantization, tqdm progress).
- `lora_merge.py` — Variant-aware weight merge helpers (diff/set/lora/loha/lokr/glora) with strict validation.
- `lora_state_dict.py` — LoRA tensor parsing + target-key mapping wrappers (backed by `runtime.adapters.lora`).
- `lora_apply.py` — Applies native LoRA selections to loaded networks.
- `unet.py` — Codex-native UNet patcher built on typed helpers (`SamplingReservation`, `ControlNetChain`) for deterministic sampling reservations, ControlNet chaining, and patch registration.
- `denoiser.py` — Generic `DenoiserPatcher` wrapper (ControlNet-free) for non-UNet denoisers; wraps `SamplerModel` and exposes the shared `ModelPatcher` surface.
- `vae_normalization_policy.py` — Typed VAE normalization policy resolver (`enum` + `dataclasses`) with explicit per-family shift contracts.
- `vae.py` — Canonical VAE wrapper; resolves diffusers/native geometry and prefers an encoder's explicit final `latents` tensor before sampling a fallback `latent_dist`.
- Additional patch modules (e.g., adapters) live here as they are ported.

## Notes
- Patchers should operate on runtime objects provided by `runtime/` and `engines/` without duplicating loading logic.
- 2026-07-30: encode outputs that expose both an explicit tensor-valued `latents` field and `latent_dist` use the explicit owner-produced tensor; the generic wrapper must not sample the posterior again or undo owner-controlled rank regulation.
- 2026-07-30: `VAE` resolves native 3D autoencoder geometry from `scale_factor_spatial` and `z_dim` when diffusers-style `down_block_types` and `latent_channels` are absent; missing or invalid geometry remains fail-loud.
- 2026-07-31: `ModelPatcher` clones preserve the same registered streamed runtime; streamed patch/unpatch delegates residency to that runtime, keeps `current_device` storage-oriented, and never invokes root `.to(...)` or whole-model pinning.
- 2026-08-01: `vae.py` regular/tiled encode paths pass pristine `[0,1]` pixels into one owned device-local normalization seam; regular and CPU-fallback decode clamp on the execution device before transfer, while tiled final-range semantics remain unchanged.
- 2026-03-22: `vae.py` encode paths now accept optional `encode_seed` and build a device-local posterior generator for diffusers-style `latent_dist.sample(...)`; regular, tiled, regular->tiled retry, and CPU-fallback encode paths must all recreate generators from the same seed when they restart full-image work, and seeded posterior sampling failures must fail loud instead of silently degrading to mean latents.
- LoRA merges are transactional: loaders snapshot parameters, track deterministic patch order, surface tqdm progress, and raise on any mismatched tensor metadata.
- When introducing new patch behaviour, add explicit configuration flags/options and document them in `.sangoi/backend/`.
- Mutator methods must raise on invalid payloads (no fallbacks) and emit backend debug logs; `ModelPatcher` now centralises logging/telemetry for patch registration.
- `patchers/__init__.py` is a package marker (no facade exports); import patcher APIs from their defining modules.
- ControlNet patching lives under `apps/backend/patchers/controlnet/`, with architecture-specific modules located in `architectures/` (SD today; Flux/Chroma placeholders ready). Use `apply_controlnet_advanced` or `UnetPatcher.add_control_node` to register controls.
- Extension-facing compatibility is preserved via the new graph-backed patcher—no linked lists remain, and `UnetPatcher.add_patched_controlnet` builds `ControlNode` instances directly.
- VAE patcher now respects the AUTO precision ladder: decode/encode paths inspect for NaNs and escalate fp16↔bf16↔fp32 via `memory_management.manager.report_precision_failure`; user-forced dtypes skip the ladder and surface explicit errors.
- 2025-11-03: Host pinning for offloaded models honours `RuntimeMemoryConfig.swap.pin_shared_memory`; disable the flag to avoid Windows pagefile pressure.
- 2025-11-22: VAE patcher unwraps diffusers `DecoderOutput`/`AutoencoderKLOutput` before `.to(...)`, preventing `'DecoderOutput' object has no attribute 'to'` when SDXL uses the standard diffusers VAE.
- 2025-12-05: VAE patcher gains a `smart_fallback` path that, when enabled, catches CUDA OOM during decode/encode and performs a single full-image CPU fallback. When smart fallback is off, regular decode/encode OOM retries tiled VAE; forced `vae_always_tiled` OOM now fails loud.
- 2026-01-02: Removed token merging patches; prompt token-merging tags are stripped but have no effect.
- 2026-01-02: Added standardized file header docstrings to patcher modules (doc-only change; part of rollout).
- 2026-01-04: Added `DenoiserPatcher` for Flux/Z-Image/WAN runtimes; `UnetPatcher` remains UNet/ControlNet-specific.
- 2026-05-02: Global LoRA apply mode resolves unset config to `online` (patch during forward); explicit `merge` remains available to rewrite weights once via `CODEX_LORA_APPLY_MODE` / `--lora-apply-mode`.
- 2026-05-18: `vae.py` tensor statistics are opt-in only (`CODEX_VAE_TENSOR_STATS=1` plus DEBUG logging) so default decode/encode paths do not run reduction + `.item()` diagnostics. Regular non-tiled decode now stores clamped `[-1,1]` output directly; tiled decode and CPU fallback retain their `[0,1]` internal buffers and convert once at the shared `decode_inner` finalization boundary.
- 2026-02-18: VAE decode/encode runtime now resolves a compute-preferred forward dtype and casts VAE residency to that effective forward dtype before execution/memory sizing to avoid mixed-dtype forward failures.
- 2026-02-18: Tiled VAE fallback was rewritten to a native context-padding + center-crop stitching flow (SUPIR-inspired, no fast/approximate path, no external tiled-scale dependency).
- 2026-02-18: Tiled encode/decode crop/index math now uses deterministic integer mapping (decode multiply, encode floor-div) to avoid border mismatch on odd image dimensions.
- 2026-02-18: Phase-1 logging cleanup removed residual stdout prints in patcher runtime paths (`controlnet/architectures/sd/t2i_adapter.py` state-dict mismatch notices and `vae.py` regular OOM retry notices), replacing them with structured backend logger warnings/debug entries.
- 2026-02-18: `lora_loader.py` now supports strict runtime toggles for merge/signature behavior: `CODEX_LORA_MERGE_MODE` (`fast=float32`, `precise=float64`) and `CODEX_LORA_REFRESH_SIGNATURE` (`structural` vs `content_sha256`) for deterministic refresh invalidation policy.
- 2026-02-18: `ModelPatcher.codex_unpatch_model` now emits canonical smart-offload INFO audit events (`pin_host_memory`) via `log_smart_offload_action(...)` when host pinning actually happens on CPU offload.
- 2026-02-18: `ModelPatcher.codex_unpatch_model` now tags that event with `SmartOffloadAction.PIN_HOST_MEMORY`; generic smart-offload `load/unload` emission remains centralized in `runtime/memory/manager.py`.
- 2026-02-08: `_NormalizingFirstStage` now supports optional per-channel latent stats (`latents_mean`/`latents_std`) in addition to scalar `scaling_factor`/`shift_factor`; 4D/5D rank, channel count, and non-finite/invalid stats are fail-loud.
- 2026-02-08: VAE normalization now resolves scale/shift via `vae_normalization_policy.py` with explicit family shift contracts: no-shift families reject explicit numeric shifts; shift-required families fail loud on missing/`None` shift.
- 2026-02-18: `vae.py` hot paths no longer hardcode fp32 outputs/buffers: `_decode_forward/_encode_forward` dropped unconditional `.float()`, decode/encode tiled+non-tiled buffers now allocate with `forward_dtype`, and storage-vs-compute contract now applies model residency using `desired_storage` while forward uses compute-preferred dtype.
- 2026-02-18: `vae.py` now gates storage-vs-compute split on manual-cast capability markers (`parameters_manual_cast` on base/modules). When markers are absent (plain diffusers/external VAE paths), forward dtype is forced to storage dtype to avoid mixed-dtype mismatch.
- 2026-02-19: `vae.py` decode/encode OOM regular→tiled retries now drop failed-path buffers before explicit cleanup (`unload_model` + `gc.collect()` + `soft_empty_cache(force=True)`) and deterministic VAE reload before tiled retry, avoiding allocator carry-over across fallback attempts.
- 2026-02-19: `vae.py` output staging now treats `DeviceRole.INTERMEDIATE=auto` as CPU-target by default, preventing large decode buffers from staying on GPU unless intermediate backend is explicitly overridden.
- 2026-02-20: `vae.py` native-LDM type checks now import `AutoencoderKL_LDM` from `runtime/common/vae_ldm.py` (canonical shared lane) instead of the WAN family path.
- 2026-02-23: `lora_loader.py::CodexLoraLoader.refresh(...)` now resolves default LoRA backup/offload placement from `memory_management.manager.offload_device()` when `offload_device` is omitted (no implicit CPU literal default).
- 2026-03-24: `lora_loader.py` now normalizes GGUF re-quantization inputs through a NumPy-safe CPU float bridge; BF16 merged tensors are promoted to FP32 before `.numpy()` so GGUF-backed LoRA refresh does not die on `Got unsupported ScalarType BFloat16` while non-GGUF merge/output behavior stays unchanged.
- 2026-03-02: `vae.py` now reports encode/decode block progress into `BackendState` during both tiled and non-tiled paths (including OOM fallback retries), so use-case progress polling can surface VAE phase progress in task streams.
- 2026-03-05: `vae.py` now consumes shared tiled geometry policy from `runtime/common/vae_tiled.py` (`resolve_vae_decode_tiled_geometry` + `VaeTileGeometry`/window iterator); decode fallback defaults remain `64/64/16` for non-Anima, with `ModelFamily.ANIMA` override `48/48/24`.
- 2026-03-29: `unet.py::_iter_transformer_coordinates()` must enumerate every internal `SpatialTransformer.transformer_blocks[]` entry, not merely the number of `SpatialTransformer` modules. The patch key `(block_name, block_index, transformer_index)` only has one transformer slot per block, so if a `TimestepEmbedSequential` ever carries more than one `SpatialTransformer`, the patcher must fail loud instead of silently collapsing coordinates.
- 2026-03-29: Heavy slot-fanout replace patches such as IP-Adapter attn2 must register through one bounded batch mutation on the owned patcher clone; repeated per-slot `model_options` reassignment destroys shared module owners and can explode host RAM.

### unet.py notes
- `control_nodes` é uma propriedade somente leitura (retorna cópia). Acesse como `unet.control_nodes`, não `unet.control_nodes()`.
- `activate_control()` recompõe o composite (`build_composite`) sempre que os nós mudam (ex.: após `add_control_node`).
- 2025-11-02: removido `@property` duplicado em `control_nodes` que podia levar a `TypeError: 'property' object is not callable` em tempo de acesso.
- 2026-02-09: LoRA apply now clears stale patch state on empty selections and fails loud on partial patcher contracts.
- 2026-02-09: LoRA refresh/merge must run outside `torch.inference_mode()`; internal inference-mode disabling was removed to keep version-counter fixes scoped to correct request entrypoints.
- 2026-02-11: `lora_apply.py` empty-selection reset now clears/refreshes denoiser + any available text-encoder patchers (engine-key agnostic) instead of hardcoding `text_encoders['clip']`; non-empty selection path remains fail-loud when CLIP mapping prerequisites are absent.
- 2026-02-18: `lora_apply.py` now fails loud when a selected LoRA produces zero compatible patches (`no compatible layers` / `zero parameters touched`) so SDXL key-layout mismatches do not silently no-op.
- 2026-02-25: `lora_apply.py` non-empty apply now starts from explicit patch-state reset (single-owner semantics), refreshes all text patchers after apply, hard-resets state on apply failure, and updates `engine.current_lora_hash` deterministically on apply/reset for conditioning-cache identity.

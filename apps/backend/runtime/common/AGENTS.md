# apps/backend/runtime/common Overview
Date: 2025-10-28
Last Review: 2026-08-01
Status: Active

## Purpose
- Shared runtime utilities available to multiple model families (base layers, helpers).

## Subdirectories
- `nn/` — Common neural network modules (core transformers/UNets, attention blocks, etc.).

## Notes
- 2026-03-21: `vae_lane_policy.py` now exposes `validate_vae_key_names(...)` as a fail-loud guard instead of stripping wrapper prefixes; `vae.py` load paths now reject any attempt to rewrite stored VAE keys before lane resolution or keyspace interpretation.
- 2026-03-20: `vae.py` now prepares external VAE overrides by enforcing the stored-key contract first and then filtering only the known SDXL/Flow bookkeeping metadata before lane-specific keyspace resolution, so SDXL override loads stay strict without model-class heuristics.
- 2026-05-17: `vae.load_flow16_vae(...)` and `vae.load_flux2_vae(...)` single-file paths load through `safe_load_state_dict(...)` and fail on any remaining missing or unexpected keys after the owned key-name, metadata, layout, and keyspace steps. Flow16 known training metadata filtering remains allowed; ratio-tolerant VAE acceptance is not allowed on these native/common VAE paths.
- Add reusable building blocks here to avoid duplication across model-specific runtimes.
- `vae.py` now validates Flow16 VAE source keys without wrapper rewrites, reuses strict SDXL/Flux LDM→diffusers keyspace resolution for Z Image, and fails fast on incompatible (non-16-channel) VAEs to avoid noisy decodes.
  - Flow16 config parity: includes `use_quant_conv=false` / `use_post_quant_conv=false` (HF Flux/Z-Image configs) so missing quant conv weights do not trigger false drift warnings.
- 2026-01-25: Flow16 VAE state_dict preparation now reuses the strict SDXL VAE keymap (`apps/backend/runtime/state_dict/keymap_sdxl_vae.py`), drops `model_ema.decay` / `model_ema.num_updates`, and fails loud on unknown non-weight keys (no silent “skip conversion” path).
- 2026-02-11: Global VAE lane policy now treats WAN22 as native-LDM-only: `resolve_vae_layout_lane` fails loud on non-LDM layout or `diffusers_native` override for WAN22 instead of allowing drift between policy and loader behavior.
- 2026-02-16: WAN22 native-LDM-only policy is now enforced across explicit family variants (`WAN22_5B`, `WAN22_14B`, `WAN22_ANIMATE`) via shared `_WAN22_FAMILIES` gate.
- 2026-02-11: `vae_ldm.sanitize_ldm_vae_config` now maps WAN alias config fields (`z_dim`, `base_dim`, `dim_mult`, `num_res_blocks`) into native LDM constructor fields before instantiation, keeping config-to-constructor contracts explicit.
- 2026-02-20: `vae_ldm.py` now owns the full shared native `AutoencoderKL_LDM` implementation (moved out of `runtime/families/wan22/vae.py`); `runtime/families/wan22/vae.py` remains a compatibility re-export shim and `vae_codex3d.py` imports `DiagonalGaussianDistribution` from `runtime/common/vae_ldm.py`.
- 2026-02-11: Added shared native temporal VAE lane module `vae_codex3d.py` (`AutoencoderCodex3D`) with strict diffusers→codex keyspace resolution (`resolve_codex3d_vae_keyspace`) and config normalization (`sanitize_codex3d_vae_config`) for no-flatten 3D runtime paths.
- 2026-02-16: `vae_codex3d.py` now delegates WAN22 3D VAE keyspace ownership to `apps/backend/runtime/state_dict/keymap_wan22_vae.py` to keep model keymaps centralized in `runtime/state_dict`.
- 2026-02-17: `vae_codex3d.py` now mirrors upstream WAN temporal cache semantics in native code (causal conv cache, chunked encode/decode loop, cached 3D upsample handling, nearest-exact upsample), closing WAN2.2 14B I2V decode parity drift without switching to Diffusers model classes.
- 2026-02-21: `vae_codex3d.py` now reduces encode peak overlap by preallocating the full encoded tensor and writing per-chunk slices directly (no chunk-list + terminal `torch.cat` in encode); cache paths also avoid redundant device-cast copies when merging temporal tails.
- 2026-02-22: `vae_codex3d.py::AutoencoderCodex3D.decode(...)` now supports optional chunk callbacks and always clears temporal caches in `finally`, enabling caller-side streamed frame export without materializing full decoded timelines on accelerator.
- 2026-03-01: `vae_codex3d.py::AutoencoderCodex3D.decode(...)` now runs `conv2` per temporal decode chunk with causal-cache carry-over (instead of materializing `conv2(z)` for full timeline upfront), reducing decode-boundary peak VRAM while preserving chunk-by-chunk decode semantics.
- 2026-03-01: `vae_codex3d.py` now emits block-level decode trace diagnostics under `CODEX_TRACE_INFERENCE_DEBUG=1` (`[wan22.vae.trace]`), including per-frame decode boundaries plus decoder block pre/post tensor dtype/device/shape and CUDA memory snapshots for OOM triage.
- 2026-03-01: `vae_codex3d.py::Codex3DCausalConv` now overrides `_conv_forward` to use direct `torch.cudnn_convolution` when running on CUDA fp16/bf16 with PyTorch>=2.9 and cuDNN>=91002, bypassing the known Conv3d dispatch regression path that can trigger large workspace spikes.
- 2026-03-05: Added `vae_tiled.py` as the shared owner of VAE tiled geometry typing/policy and tile-window iteration (`VaeTileGeometry`, `VaeTileWindow`, `resolve_vae_decode_tiled_geometry`, `iter_vae_tile_windows`), including Anima-specific decode fallback geometry override while preserving non-Anima defaults.
- 2026-02-21: `vae_ldm.py` `DiagonalGaussianDistribution.sample()` now allocates noise directly on the target device/dtype (`torch.randn(..., device=..., dtype=...)`) to avoid hidden alloc+cast churn.
- 2026-08-01: `vae_ldm.py` no longer stores eager posterior variance; `DiagonalGaussianDistribution.var` derives the exact prior `torch.exp(logvar)` value lazily and preserves deterministic zero semantics.
- 2026-01-06: `vae.load_flow16_vae(...)` now accepts `.gguf` weights (dequantized upfront) in addition to diffusers directories and `.safetensors` files.
- 2026-02-15: `vae.load_flow16_vae(...)` now routes GGUF and torch-file state-dict loading through the requested `device` to keep checkpoint placement consistent with runtime target-device selection.
- 2026-01-02: Added standardized file header docstrings to `nn/base.py`, `nn/clip.py`, and `nn/unet/{__init__,config,utils}.py` (doc-only change; part of rollout).

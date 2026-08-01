# apps/backend/engines/anima Overview
<!-- tags: backend, engines, anima, cosmos -->
Date: 2026-02-05
Last Review: 2026-08-01
Status: Draft

## Purpose
- Host the Anima engine implementation (Cosmos Predict2 + Anima adapter) for txt2img/img2img.

## Key Files
- `apps/backend/engines/anima/anima.py` — `AnimaEngine` implementation (engine facade; delegates mode pipelines to use-cases per Option A).
- `apps/backend/engines/anima/spec.py` — Runtime spec + assembly (`assemble_anima_runtime`), including predictor config.
- `apps/backend/engines/anima/factory.py` — Factory that builds `CodexObjects` from the assembled runtime.

## Notes
- Anima uses sha-selected external assets (VAE + Qwen3-0.6B text encoder); no raw client paths.
- Engine assembly contract is explicit and fail-loud: `engine_options` must provide existing non-empty string file paths for `vae_path` + `tenc_path`.
- Capability exposure follows the Anima conditioning contract (`crossattn` + `t5xxl_ids/t5xxl_weights/t5xxl_attention_mask`; no synthesized pooled `vector`); keep `AnimaEngine.capabilities()` in sync with `runtime/model_registry/capabilities.py`.
- Runtime device consistency checks normalize equivalent labels (`cuda` and `cuda:0`) before mismatch validation; only real device mismatches should fail, and missing `denoiser.load_device` fails loud.
- Conditioning requires dual tokenization (Qwen embeddings + T5 ids/weights/attention mask), per `.sangoi/research/models/hf-circlestone-labs-anima.md`.
- 2026-02-08: `spec._predictor()` opts Anima into `simple_schedule_mode="tail_downsample_sigmas"` so `scheduler=simple` follows tail-downsample ladder selection over `predictor.sigmas`, while other flow families keep their existing SIMPLE behavior.
- 2026-02-09: Anima conditioning entrypoints now use `torch.no_grad()` (not `torch.inference_mode()`) to avoid caching inference tensors across requests (version-counter faults).
- 2026-02-23: `AnimaEngineRuntime.device` default now resolves from memory-manager mount-device authority (no hardcoded CPU default in runtime metadata).
- 2026-03-03: `spec.assemble_anima_runtime(...)` now eagerly loads external Qwen3 text-encoder weights, WAN VAE weights, and T5 tokenizer through canonical loaders; qwen patcher now wraps the concrete loaded model.
- 2026-03-04: Anima core denoiser now follows WAN22 load philosophy in `spec.assemble_anima_runtime(...)`: the transformer state dict is wrapped by a lazy module and `AnimaDiT` materialization is deferred until first real denoiser use (`to/_apply/forward/explicit attr access`). External Qwen/VAE/T5 loading remains eager and fail-loud.
- 2026-03-31: `spec.assemble_anima_runtime(...)` must pass the resolved role load device into the external Qwen/VAE loaders and reuse the returned `AnimaQwenTextEncoder` wrapper directly; do not rebuild a second wrapper around the same `.model` or let those loaders be born on `device=None`.
- 2026-08-01: `spec.assemble_anima_runtime(...)` reads explicit CORE compute intent from the live memory-manager policy, requires exact storage/compute equality, makes `auto` inherit storage, and rejects mismatches before lazy-core or external-component materialization.

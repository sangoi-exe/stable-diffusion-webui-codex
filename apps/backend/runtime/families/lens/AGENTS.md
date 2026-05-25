# apps/backend/runtime/families/lens Overview
<!-- tags: backend, runtime, families, lens -->
Date: 2026-05-24
Last Review: 2026-05-25
Status: Bootstrap / runner-hook contract

## Purpose
- Own the Lens architecture-family runtime contracts for the single public `lens` engine id.
- Keep `default`, `turbo`, and `base` as internal variants selected through `extras.lens.variant`; do not create `lens_turbo` or `lens_base` engines.
- Own the bootstrap/tokenizer/text-feature gate and the keyword-only txt2img runner-hook contract while Lens remains parked and non-runnable.

## Structure
- `config.py` — variant specs, defaults, internal text-encoder quant policy, architecture constants, official resolution bucket table, and not-implemented messages.
- `bootstrap.py` — dependency/config/tokenizer/shard/header readiness probe; metadata-only mirrors must report `weights_missing` instead of runtime-ready.
- `text_encoder.py` — direct `PreTrainedTokenizerFast` sidecar loader, Lens chat-template rendering, GPT-OSS selected-layer wrapper, and synthetic prompt-feature helpers.
- `resolution.py` — official bucket validation and Lens image-sequence geometry helpers.
- `scheduler.py` — Lens FlowMatch scheduler metadata validation, empirical `mu`, sigma ladder, and transformer timestep scaling helpers.
- `sampler.py` — reserved Lens txt2img denoising owner; keyword-only skeleton raises `NotImplementedError` until runtime lands and validates decoded hook output shapes for the canonical runner seam.
- `__init__.py` — lightweight public contract exports only.

## Notes
- This folder is bootstrap/hook-contract only in the current tranche. It must not import upstream `lens`, construct Diffusers pipelines, load tensor payloads, or claim generation support.
- Lens hook noise remains sampler-owned from `SamplingPlan` seeds/noise policy; do not pass generic `ImageRNG` into Lens because Lens future latents are sequential `[B, H*W, 128]`, not C/H/W image/latent noise.
- Native MXFP4 is unsupported until a later tranche explicitly adds `kernels` support; current text-encoder policy is internal dequant BF16 only.
- Native checkpoint keys stay native. Any future keyspace bridge belongs in `apps/backend/runtime/state_dict/**`, not in prefix strippers or eager remap helpers here.
- The reasoner is hard-deferred. Do not add prompt rewrite config, API-key state, or dormant reasoner settings under this family.

# apps/backend/runtime/families/qwen_image
Date: 2026-05-17
Last Review: 2026-07-31
Status: Active

## Purpose
- Host repo-owned Qwen Image Edit-2511 runtime contracts for the single `qwen_image` architecture family.
- Keep the live surface img2img-only; txt2img and other Qwen checkpoint generations are unsupported in this epoch.

## Key Files
- `apps/backend/runtime/families/qwen_image/config.py` - Edit-2511 metadata, prompt-template constants, dimensions, and image geometry helpers.
- `apps/backend/runtime/families/qwen_image/loader.py` - Strict native transformer/TEnc/VAE construction, lazy keyspace binding, scheduler metadata, and patcher ownership.
- `apps/backend/runtime/families/qwen_image/runtime.py` - Single-image condition/reference preprocessing, multimodal prompt encoding, true-CFG denoise, VAE decode, and component lifecycle.
- `apps/backend/runtime/families/qwen_image/runtime_latents.py` - CPU-staged runtime value objects, native 2x2 latent pack/unpack, and true-CFG norm rescaling.
- `apps/backend/runtime/families/qwen_image/scheduler.py` - Diffusers-free FlowMatch Euler scheduler metadata validation, exact sigma ladder, Euler update, and sequence geometry.
- `apps/backend/runtime/families/qwen_image/text_encoder.py` - Qwen2.5-VL config validation, exact processor batch contract, prompt-template planning, and base-model forward boundary.
- `apps/backend/runtime/families/qwen_image/transformer.py` - Edit-2511 `QwenImageTransformer2DModel` config, topology, forward contract, and zero-conditioning validation.
- `apps/backend/runtime/families/qwen_image/transformer_layers.py` - Checkpoint-owned dual-stream attention, modulation, feed-forward, and timestep layers.
- `apps/backend/runtime/families/qwen_image/transformer_rope.py` - Non-checkpoint centered three-axis RoPE tables and application math.
- `apps/backend/runtime/families/qwen_image/vae.py` - Exact external `AutoencoderKLQwenImage` SafeTensors admission, vendored-config validation, and per-channel latent normalization helpers.
- `apps/backend/runtime/families/qwen_image/__init__.py` - Lightweight public family-runtime export surface.

## Notes / Decisions
- `Qwen-Image-2.0` is an architecture/frontier label in this repo, not a concrete checkpoint/repository contract.
- The canonical exact engine and family id is `qwen_image`; do not add checkpoint-generation-specific engine ids.
- `qwen_image_variant` is fixed internal runtime/task metadata for Edit-2511. Public request payloads must fail loud if they carry it.
- Qwen Image VAE assets must come from `qwen_image_vae` API roots, match the exact 194-tensor BF16 header, and use the vendored Edit-2511 `vae/config.json`; sibling config files beside weights are not runtime authority.
- The transformer and Qwen2.5-VL text encoder keep their stored native keys unchanged. Dedicated keymaps may expose lazy runtime lookup names only.
- Transformer and Qwen2.5-VL GGUF admission validates every stored logical tensor shape before module binding; exact names and counts are insufficient on their own.
- The runtime interface is exactly `encode_conditioning`, `encode_reference`, `denoise`, and `decode`; component stages use canonical patcher load/unload telemetry and return intermediate tensors to CPU between stages.
- When a component stage and its cleanup both fail, the stage error remains primary and unload/cache failures are attached as secondary diagnostic notes.
- Reference-image VAE encoding uses posterior `mode()` and Qwen per-channel normalization exactly once.
- Reference trees under `.refs/**` and metadata mirrors under `apps/backend/huggingface/Qwen/**` are source frontiers only. Active code must stay repo-owned and Diffusers-free.
- Header/keyspace work must obey the root keymap law: do not strip prefixes, rewrite punctuation, materialize remapped state dicts, or normalize stored checkpoint layer names.

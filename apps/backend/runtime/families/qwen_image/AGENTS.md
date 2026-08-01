# apps/backend/runtime/families/qwen_image
Date: 2026-05-17
Last Review: 2026-08-01
Status: Active

## Purpose
- Host repo-owned Qwen Image Edit-2511 runtime contracts for the single `qwen_image` architecture family.
- Keep the live surface img2img-only; txt2img and other Qwen checkpoint generations are unsupported in this epoch.

## Key Files
- `apps/backend/runtime/families/qwen_image/config.py` - Edit-2511 metadata, prompt-template constants, dimensions, and image geometry helpers.
- `apps/backend/runtime/families/qwen_image/loader.py` - Strict native transformer/TEnc/VAE construction, lazy keyspace binding, CPU-backed transformer storage, streamed-runtime attachment, scheduler metadata, and patcher ownership.
- `apps/backend/runtime/families/qwen_image/runtime.py` - Single-image condition/reference preprocessing, shared-vision prompt encoding, request-static transformer context, terminal denoise validity reporting, VAE decode, and atomic mutually exclusive text-encoder/VAE/streamed-core component-stage leases.
- `apps/backend/runtime/families/qwen_image/runtime_latents.py` - CPU-staged runtime value objects, native 2x2 latent pack/unpack, true-CFG norm rescaling, and private device-side denoise validity bits.
- `apps/backend/runtime/families/qwen_image/scheduler.py` - Diffusers-free FlowMatch Euler scheduler metadata validation, exact sigma ladder, device-valid Euler update, and sequence geometry.
- `apps/backend/runtime/families/qwen_image/text_encoder.py` - Qwen2.5-VL config validation, one-image positive/negative processor batches, prompt-template planning, and one-vision/two-language-forward boundary.
- `apps/backend/runtime/families/qwen_image/streaming.py` - Mandatory blocked-swap activation, full-transformer execution lease, streamed residency lifecycle, static/one-block transitions, exact host restoration, and peak CUDA telemetry.
- `apps/backend/runtime/families/qwen_image/streaming_slots.py` - Complete direct-slot inventory, physical alias/overlap validation, byte accounting, and staged tensor construction without stored-key rewriting.
- `apps/backend/runtime/families/qwen_image/transformer.py` - Edit-2511 `QwenImageTransformer2DModel` config, topology, request-static geometry/modulation/RoPE context, full-forward READY/device execution lease, one-block execution context, and zero-conditioning validation.
- `apps/backend/runtime/families/qwen_image/transformer_layers.py` - Checkpoint-owned dual-stream attention, modulation, feed-forward, and timestep layers.
- `apps/backend/runtime/families/qwen_image/transformer_rope.py` - Non-checkpoint centered three-axis RoPE tables registered as non-persistent buffers plus application math.
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
- Edit-2511 execution requires normalized `core_streaming_enabled=true` with `swap_method='blocked'`; unsupported, asynchronous, or absent streaming states fail before heavy runtime assembly.
- The transformer GGUF remains CPU-backed. Static transformer owners plus exactly one of the 60 blocks may be CUDA-resident during denoise; no async prefetch, whole-model pinning, LRU window, or second per-weight transfer scheduler belongs to this tranche.
- Text encoder, VAE, and streamed transformer stages are mutually exclusive under one runtime-owned component-stage lease. Denoise verifies its post-unload `OFFLOADED` state before releasing that lease, so the next request cannot prepare the shared core early. The streamed runtime holds one execution lease across all static and block transformer work; every block context and stage cleanup restores the exact original host tensor objects before the next component stage.
- When a component stage and its cleanup both fail, the stage error remains primary and unload/cache failures are attached as secondary diagnostic notes.
- Reference-image VAE encoding uses posterior `mode()` and Qwen per-channel normalization exactly once.
- Conditioning preprocesses the image once and executes the Qwen2.5-VL vision tower once; positive and negative language-model forwards stay serial and share the exact processor pixel/grid tensors plus computed image features.
- Each denoise request prepares transformer image geometry, modulation indices, and RoPE tensors once per distinct text sequence length under a bounded READY execution lease. The transformer accepts only that immutable request context and the CPU-staged all-valid conditioning mask.
- True-CFG and Euler validity remain device-resident signed `int32` bits through the denoise loop. CFG-off positive validity joins the same bitmask, and the runtime performs one terminal read after successful sampling; cancellation and exceptional exits perform no terminal read.
- Reference trees under `.refs/**` and metadata mirrors under `apps/backend/huggingface/Qwen/**` are source frontiers only. Active code must stay repo-owned and Diffusers-free.
- Header/keyspace work must obey the root keymap law: do not strip prefixes, rewrite punctuation, materialize remapped state dicts, or normalize stored checkpoint layer names.

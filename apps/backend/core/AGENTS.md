# apps/backend/core Overview
Date: 2025-10-28
Last Review: 2026-05-24
Status: Active

## Purpose
- Provides the fundamental building blocks for backend orchestration: device discovery, state tracking, RNG, request handling, engine contracts, and parameter parsing.

## Subdirectories
- `params/` — Typed parameter schemas and helpers (currently video-specific) that translate user requests into engine-friendly structures.
- `contracts/` — Backend “contract as code” modules shared across UI ↔ API ↔ runtime (e.g. per-engine asset requirements).

## Key Files
- `engine_interface.py` — Defines the base interfaces that engines must implement.
- `orchestrator.py` — Coordinates use-case execution, binding requests to engines and runtime contexts.
- `devices.py` / `state.py` — Track hardware availability and request-scoped generation state.
- `engine_loader.py` — Bundle-aware engine loader used by use cases for model loading and runtime option application.
- `rng.py` / `philox.py` — Native RNG stack (CPU/GPU/Philox) used across tasks.
- `requests.py` — Typed request objects and validation helpers.
- `registry.py` — Engine registration/lookup for orchestration.
- `exceptions.py` — Core exception types surfaced by orchestration.
- `strict_values.py` — Shared strict parsers for loose scalar values (strict boolean/integer parsing with fail-loud errors).

## Notes
- New engine integrations must conform to `engine_interface.py` and register via `registry.py`.
- Keep RNG and device logic centralized here—avoid duplicating random seeding in downstream modules.
- 2025-12-14: Video requests (`Txt2VidRequest`/`Img2VidRequest`) include `steps` explicitly (defaulting to 30 to match `/api/{txt2vid,img2vid}`) so API parsing and `build_video_plan()` stay aligned.
- 2025-12-16: Added `TaskType.VID2VID` + `Vid2VidRequest` for WAN video-to-video orchestration; video requests also carry optional `video_options` for export settings.
- 2025-12-30: `InferenceOrchestrator` now reloads an already-loaded engine when load-affecting `engine_options` change (e.g. `text_encoder_override`, VAE override, core streaming), so overrides actually apply and caches don’t go stale across requests.
- 2026-01-06: `InferenceOrchestrator` reload fingerprint now includes explicit `engine_options.vae_source`/`engine_options.tenc_source` to ensure built-in vs external asset selection changes trigger reloads.
- 2026-01-28: `InferenceOrchestrator` reload fingerprint now includes `engine_options.zimage_variant` so Z-Image Turbo/Base switches trigger a reload.
- 2026-05-17: `InferenceOrchestrator` reload fingerprint now includes internal `engine_options.qwen_image_variant` so Qwen Image txt2img (`2512`) and img2img edit (`edit_2511`) cannot reuse stale cached runtime state across variant switches.
- 2026-05-24: `InferenceOrchestrator` reload fingerprint now includes internal `engine_options.lens_variant` for the parked Lens skeleton so future `default|turbo|base` switches cannot reuse stale cached runtime state once Lens loading lands.
- 2026-01-01: `InferenceOrchestrator` now purges VRAM (unload cached engines + memory manager unload/empty_cache) before a generation when the requested `(checkpoint, text encoders)` signature differs from the previous generation (prevents OOM on model swaps).
- 2026-01-02: Added standardized file header docstrings across `apps/backend/core/**` modules (doc-only change; part of rollout).
- 2026-01-03: `apps/backend/core/__init__.py` no longer re-exports star-import facades; callers must import from specific modules (e.g. `core.requests`, `core.registry`).
- 2026-01-06: Refreshed the `orchestrator.py` module header block to reflect the current engine-options fingerprint fields (`vae_source`/`tenc_source`) (doc-only change).
- 2026-04-06: `Img2ImgRequest` now carries explicit mask/inpaint controls under `inpaint_mode` plus the existing blur/invert/full-res/filled-content knobs for Codex-native masked img2img.
- 2026-02-25: `Img2ImgRequest` masked defaults are now aligned with ADetailer-like behavior for UI/API parity (`inpainting_fill=1`, `inpaint_full_res_padding=32`).
- 2026-02-03: Image request dataclasses now carry hires config via `hires` (renamed field; no alias).
- 2026-02-09: `InferenceOrchestrator` no longer scrubs traceback chains before wrapping load/execution failures; wrapped `EngineLoadError`/`EngineExecutionError` now preserve source-frame causality for diagnostics.
- 2026-02-15: `BaseRequest` now carries `settings_revision` for strict generation contract propagation (routers validate revision against persisted options revision before dispatch).
- 2026-02-16: `InferenceOrchestrator` primary-device drift checks now probe canonical `codex_objects.denoiser` residency (`load_device`/`device`/parameter-device seams) with legacy `codex_objects.unet` fallback, fixing contract drift where device reload checks could silently skip reloading.
- 2026-02-17: `InferenceOrchestrator` reload fingerprint now also tracks `engine_options.dtype` (normalized string) so dtype override changes trigger a reload instead of reusing stale loaded engines.
- 2026-02-21: `engine_loader.py` now resolves default diffusers attention backend from runtime memory config (launcher/bootstrap authority) instead of a potentially stale saved option snapshot.
- 2026-02-21: Added `strict_values.parse_bool_value(...)` and wired orchestrator reload fingerprint streaming fields to strict bool parsing (`codex_core_streaming` / `core_streaming_enabled`) to remove permissive truthy coercion traps.
- 2026-02-21: Added `strict_values.parse_int_value(...)` for fail-loud integer parsing (used by runtime/services paths that previously coerced malformed numeric settings silently).
- 2026-02-21: `InferenceOrchestrator` generation-signature and reload-fingerprint surfaces are now aligned (`generation_signature` embeds `reload_fingerprint`), and invalid load-affecting engine options fail loud with `EngineLoadError` (no blanket swallow around fingerprint parsing).
- 2026-04-05: `InferenceOrchestrator` now expects the normalized internal streaming key `core_streaming_enabled` only. Do not reintroduce `codex_core_streaming` as a second internal engine-option owner after router/task normalization.
- 2026-02-21: `InferenceOrchestrator._purge_vram(...)` now fails loud on cleanup errors (cached-engine unload/memory-manager failures), and load-failure paths surface additive cleanup failure context instead of warning-only degradation.
- 2026-02-21: `InferenceOrchestrator.run(...)` now preserves error taxonomy on strict preflight/cleanup paths: pre-load purge/preflight failures are wrapped as `EngineLoadError`, and execution-path purge failures still return `EngineExecutionError` with additive cleanup context.
- 2026-02-22: `InferenceOrchestrator._purge_vram(...)` now treats cleanup-time CUDA OOM as non-fatal only when unwinding an execution failure (`reason='engine execution failure'`), logging a warning and preserving the original execution error as primary.
- 2026-02-23: `devices.default_device()` / `devices.cpu()` now resolve through memory-manager authority (`manager.mount_device()` / `manager.cpu_device`) and fail loud when manager contracts are invalid, removing local CUDA/CPU chooser hardcodes from core helpers.
- 2026-02-23: `InferenceOrchestrator` device-drift reload checks now compare full canonical device identity (`str(current) != str(desired)`) instead of backend type-only comparison, so CUDA index changes (`cuda:0` vs `cuda:1`) trigger deterministic reload.
- 2026-02-24: `InferenceOrchestrator.run(...)` now enforces a VRAM cleanup barrier between model unload/load transitions (`engine.unload()` -> `_purge_vram(...)` -> `engine.load(...)`) when reloading an already-loaded engine.
- 2026-02-28: `InferenceOrchestrator._purge_vram(...)` removed the optional `operations_gguf.clear_cache` import/call path; cleanup now relies on orchestrator unload + memory-manager unload/soft-empty + GC + CUDA cache release.
- 2026-03-02: `BackendState` now tracks VAE phase progress (`vae_phase`, `vae_block_index`, `vae_block_total`) with explicit snapshot/update helpers so image use-cases can stream encode/decode block progress alongside sampling.
- 2026-03-08: `BackendState.sampling_steps` may now be unknown (`None`) for open-ended native samplers such as `dpm adaptive`; progress consumers must treat missing totals as an honest unbounded-progress signal instead of coercing them to zero.
- 2026-04-06: `Img2ImgRequest` keeps additive `per_step_blend_strength` (`0..1`, default `1.0`) only for masked `inpaint_mode='per_step_blend'`; this remains a scalar on the generic branch, not a second mode owner.
- 2026-05-17: `InferenceOrchestrator` canonicalizes exact engine keys and aliases through `EngineRegistry.get_descriptor(...).key` before cache, generation-signature, and reload-fingerprint ownership. Unknown keys must preserve `EngineNotFoundError` taxonomy and must not be wrapped by load-preflight logic.
- 2026-03-31: `state.py` now tags raw sampling, VAE, and live-preview snapshots with per-run owner tokens, and `live_preview_snapshot()` is the canonical atomic preview read seam; downstream code must stop stitching together `current_image`, `id_live_preview`, and `current_image_sampling_step` by hand.

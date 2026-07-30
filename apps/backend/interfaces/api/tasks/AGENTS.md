# apps/backend/interfaces/api/tasks Overview
<!-- tags: backend, api, tasks, orchestration -->
Date: 2026-01-30
Last Review: 2026-05-30
Status: Active

## Purpose
- Host shared, import-light task orchestration helpers used by API routers.
- Keep routers thin: validate + dispatch + stream, while task boilerplate (queue/status/progress/result/end) lives here.

## Key Files
- `apps/backend/interfaces/api/tasks/generation_tasks.py` — common task runner helpers for generation endpoints (event streaming, engine options build, image encoding).
- `apps/backend/interfaces/api/tasks/upscale_tasks.py` — task workers for standalone `/api/upscale` and HF upscaler downloads (`/api/upscalers/download`) with explicit integrity verification (manifest sha256 when available).

## Notes
- 2026-05-30: `generation_tasks.py::build_engine_options(..., engine_key=...)` requires the worker-resolved engine key, resolves runtime VAE usage through `EngineAssetContract.uses_vae`, forwards explicit `vae_source` for every VAE-using engine, preserves `requires_vae` as the external-VAE requirement, and rejects `vae_source` / `vae_path` / `vae_sha` for no-VAE engines.
- 2026-05-24: `generation_tasks.py::build_engine_options(...)` reserves only nested `extras.lens.variant` for Lens variant propagation and rejects `extras.lens` on non-Lens engines; flat aliases such as `extras.lens_variant`, `extras.lensVariant`, `extras.variant`, and `extras.engine_options` fail loud.
- 2026-07-30: `generation_tasks.py::build_engine_options(...)` forwards the backend-derived fixed Qwen Image identity `edit_2511`; public payloads still cannot own `qwen_image_variant`.
- This package must remain import-light (avoid importing torch-heavy modules at import time). Prefer local imports inside functions.
- Cancellation semantics are owned by `apps/backend/interfaces/api/task_registry.py` and must be preserved (always emit `end`).
- 2026-02-09: Task workers now compare cancellation policy using `TaskCancelMode.IMMEDIATE` (enumized contract from `task_registry.py`) instead of raw string literals.
- 2026-02-15: `generation_tasks.py` now emits contract-trace JSONL events (opt-in via `CODEX_TRACE_CONTRACT`) with prompt hashing only (`prompt_hash`, never raw prompt text).
- 2026-02-15: task workers sanitize terminal errors through `apps/backend/interfaces/api/public_errors.py` before persisting `entry.error`, preventing raw exception leakage through task status/SSE.
- 2026-03-30: task workers now persist the full `PublicTaskError` envelope (public-safe `message` + additive `code/error_id`) instead of ad-hoc strings; engine-exception full dumps stay owned by `core/orchestrator.py`, while workers only persist the public envelope.
- 2026-02-16: `generation_tasks.py` logs typed `EngineExecutionError` explicitly to API console logs (`task_id`, `mode`, `engine`) before persisting sanitized `entry.error`; wire payload contract remains unchanged.
- 2026-02-16: `generation_tasks.py` now enforces strict bool parsing for request smart flags via `resolve_request_smart_flags(...)` and drains orchestrator iterators on immediate-cancel paths (no early-return teardown bypass; inference-gate release happens after iterator/use-case finalizers complete).
- 2026-02-18: `generation_tasks.py` now serializes PNG `parameters` as A1111-compatible infotext (prompt + optional negative prompt + KV line) instead of JSON blobs, while still preserving provenance text chunks separately.
- 2026-04-05: `generation_tasks.py::_format_parameters_infotext(...)` now reads only the canonical info keys emitted by the image use-cases (`sampler`, `scheduler`, `guidance_scale`, `model`, `vae`, `denoising_strength`, `rng`). Do not add alias-reader wash for old key names there.
- 2026-02-18: `generation_tasks.py` inference-gate wait cancellation now honors any requested cancel mode (`immediate` or `after_current`) before work starts; once running, only `immediate` interrupts in-flight generation.
- 2026-02-18: `upscale_tasks.py` follows the shared gate-wait rule: any cancel mode aborts before start while `immediate` remains the only in-flight interrupt mode.
- 2026-02-21: `generation_tasks.py` now exposes `force_runtime_memory_cleanup(...)` and invokes it on image worker exceptions to force best-effort cleanup of orchestrator cache, runtime memory manager state, and CUDA cache.
- 2026-02-28: `generation_tasks.py::force_runtime_memory_cleanup(...)` removed the optional `operations_gguf.clear_cache` import/call path to avoid stale helper warnings after GGUF cache-hook removal.
- 2026-02-22: `generation_tasks.py::force_runtime_memory_cleanup(...)` now logs cleanup failures without traceback payload (`exc_info=False`), preventing repeated stacktrace floods when CUDA is already in OOM/error state after engine failure.
- 2026-02-21: `generation_tasks.py::build_engine_options(...)` now parses settings key `codex_core_streaming` via shared strict bool parsing and emits canonical engine option `core_streaming_enabled` (no truthiness coercion from malformed option snapshots).
- 2026-02-21: `generation_tasks.py` now parses `samples_save` via shared strict bool parsing before output persistence, removing permissive `bool("false")==True` behavior.
- 2026-02-22: task workers now log warning-level diagnostics when inference-gate release fails (`generation_tasks.py`, `upscale_tasks.py`) instead of silently swallowing release errors.
- 2026-03-31: standalone SUPIR task workers were removed when SUPIR mode moved into the canonical SDXL `img2img.py` owner path; this package now owns only generic generation workers plus standalone upscale/download tasks.
- 2026-03-02: `generation_tasks.py` now preserves `ProgressEvent.message` and `ProgressEvent.data` in streamed `progress` task events, so frontend consumers can render phase-aware total-progress metadata from backend-emitted payloads.
- 2026-03-31: `generation_tasks.py` now clears raw progress state per task run and passes the expected task owner token into `live_preview_service.py`; workers remain thin forwarders and must not read raw preview fields from `core.state` directly.

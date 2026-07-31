# apps/backend/interfaces/api Overview
<!-- tags: backend, api, fastapi, routers -->
Date: 2026-01-08
Last Review: 2026-07-31
Status: Active

## Purpose
- FastAPI endpoint implementations and helper modules for the Codex backend API.

## Key Files
- `apps/backend/interfaces/api/run_api.py` — uvicorn factory + FastAPI assembly (router composition + SPA mount).
- `apps/backend/interfaces/api/routers/system.py` — health/version/memory endpoints.
- `apps/backend/interfaces/api/routers/settings.py` — settings schema + values endpoints.
- `apps/backend/interfaces/api/routers/ui.py` — tabs/workflows/blocks/presets persistence endpoints.
- `apps/backend/interfaces/api/routers/models.py` — model inventory + samplers/schedulers/embeddings + engine capabilities endpoints.
- `apps/backend/interfaces/api/routers/paths.py` — `apps/paths.json` endpoints.
- `apps/backend/interfaces/api/routers/options.py` — options store read/update/validate endpoints.
- `apps/backend/interfaces/api/routers/tasks.py` — task status/SSE/output endpoints.
- `apps/backend/interfaces/api/routers/tests.py` — bounded backend diagnostics endpoints.
- `apps/backend/interfaces/api/routers/tools.py` — GGUF converter + SafeTensors merge + file browser endpoints.
- `apps/backend/interfaces/api/routers/generation.py` — txt2img/img2img/txt2vid/img2vid/vid2vid endpoints.
- `apps/backend/interfaces/api/wan_video_request_keys.py` — backend API-owned WAN video request-key allowlists used by `routers/generation.py`.
- `apps/backend/interfaces/api/file_metadata.py` — GGUF/SafeTensors header readers for `/api/models/file-metadata` (UI/debug).
- `apps/backend/interfaces/api/path_utils.py` — repo-relative path normalization helpers.
- `apps/backend/interfaces/api/json_store.py` — JSON load/save helpers for persistence files.
- `apps/backend/interfaces/api/task_registry.py` — in-process task registry (SSE queue + cancel flags).
- `apps/backend/interfaces/api/public_errors.py` — public-safe async-task error envelopes (`message` + additive `code/error_id`) and synchronous HTTP error details.
- `apps/backend/interfaces/api/tasks/generation_tasks.py` — shared generation task worker helpers (image task runners + engine options + PNG encoding).
- `apps/backend/interfaces/api/serializers.py` — checkpoint serialization helper.
- `apps/backend/interfaces/api/upscalers_manifest.py` — `upscalers/manifest.json` schema validation/normalization (used by `/api/upscalers/remote`).
- `apps/backend/interfaces/api/dependency_checks.py` — backend-owned dependency-check builder used by `/api/engines/capabilities`.

## Notes
- `run_api.py` is composition-only: it wires routers and mounts the UI; route logic lives in `routers/`.
- `routers/tests.py` is the bounded live-diagnostics owner for backend validation routes; keep test/diagnostic logic out of `system.py` unless the surface is truly health-level and lightweight.
- 2026-03-28: launcher-started API fallback truth now lives in `apps/launcher/services.py`, while `run_api.py` remains the direct-run self-defense seam; `run_api.py` also wires the bounded `/api/tests/attention/sram/splitkv` router for live SRAM split-KV diagnostics, and that route is operator-facing: malformed payloads 400, expected execution outcomes return structured receipts.
- Task state is centralized in `task_registry.py` so generation + tasks routers share cancellation/status logic.
- 2026-03-30: terminal async-task errors are stored as one `PublicTaskError` envelope in `task_registry.py`; `public_errors.py` shapes that envelope, `routers/tasks.py` serializes it without rewording, and raw engine exception dumps stay owned by `core/orchestrator.py` + `runtime/diagnostics`.
- 2026-03-31: `/api/img2img` now rejects masked requests synchronously when `/api/engines/capabilities` says the active semantic engine does not support img2img mask/inpaint semantics (`supports_img2img_masking`), so frontend gating and router preflight stay on the same backend-owned truth.
- 2026-04-12: `/api/options` is now a mandatory compare-and-set lane only: live callers must provide `X-Codex-Expected-Revision`, stale writes fail with `409` under the locked store owner, and `/api/ui/presets/apply` is checkpoint-only now that preset-driven option mutation was removed instead of bypassing the revision-aware options path.
- 2026-04-05: checkpoint API egress now requires canonical `CheckpointRecord.filename` and `short_hash` at the API boundary. `serializers.py` and `routers/models.py` must not fall back to legacy `path` / `shorthash` readers, and `/api/models/checkpoint-metadata` must fail loud if `filename` is missing.
- 2026-04-05: request-device parsing is `device`-only now. `device_selection.py` may still migrate bootstrap authority from `codex_main_device`, but API payload readers must reject `codex_device` / `codex_diffusion_device` instead of laundering them.
- 2026-03-06: WAN video request allowlists are backend API-owned in `wan_video_request_keys.py`; `routers/generation.py` consumes them directly, and `runtime/state_dict/keymap_wan22_transformer.py` no longer owns HTTP request-key authority.
- 2026-03-08: `/api/samplers` is the executable sampler surface: backend support flags are implementation-backed (not enum/catalog-only), the route must fail loud if a `supported=true` sampler is missing registry metadata, and frontend filtering is expected to honor `supported !== false`.
- `/api/models/file-metadata` is intended for UI/debug; it returns `flat` plus a nested view of dotted keys. Codex-generated GGUF files use `model.*`, `codex.*`, and `gguf.*` keys (no legacy `general.*` provenance fields).
- 2026-03-20: `/api/models` checkpoint serialization now includes explicit selector metadata (`format`, `core_only`, `core_only_reason`, optional `family_hint`) so image clients can validate request selectors from inventory truth instead of guessing from filenames.
- 2026-03-20: the generic LTX video route no longer inherits WAN geometry/frame assumptions: `generation.py` now treats LTX width/height as strict `32px` multiples and frame counts as strict `8n+1`, while the later explicit `two_stage` lane keeps width/height as final output dimensions and requires both divisible by `64`; the route uses explicit safe geometry/frame defaults instead of the old WAN-style `768x432` / `17` fallback.
- 2026-03-26: the generic LTX video route now requires explicit `ltx_execution_profile` on direct requests and derives the canonical `euler` / `simple` lane internally; raw LTX `*_sampler` / `*_scheduler` request fields are no longer part of the public wire contract.
- 2026-01-18: `/api/engines/capabilities` now also includes `engine_id_to_semantic_engine` so UI callers can keep engine-id and semantic-engine key spaces explicit.
- 2026-01-20: Removed unreferenced API helper modules (`media_helpers.py`, `script_models.py`) (no call sites).
- 2026-01-21: WAN stage LoRA inputs are stage arrays (`loras[]` with `{sha, weight}` entries); legacy single-field stage keys (`lora_sha`/`lora_weight`) and raw-path stage `lora_path` are rejected during payload normalization/validation.
- 2026-01-24: Settings schema/values are now strict: schema is served from the generated registry (JSON fallback), and persisted values are pruned against the registry on startup (unknown keys dropped; invalid values clamped).
- 2026-01-25: `run_api.py` migrated the deprecated `@app.on_event("startup")` hook to FastAPI lifespan handlers (removes DeprecationWarning).
- 2026-01-31: Added `interfaces/api/tasks/` to keep routers thin by centralizing shared generation task worker boilerplate (status/progress/result/end + engine options build for image modes).
- 2026-02-06: Added backend-owned `dependency_checks` contract for `/api/engines/capabilities` (ready + per-row checks), built in `dependency_checks.py`.
- 2026-02-06: `/api/engines/capabilities` key-space map now includes `flux1_fill -> flux1` in `engine_id_to_semantic_engine` for strict frontend taxonomy mapping parity.
- 2026-02-08: Swap-model request contract now uses `switch_at_step` in both `extras.refiner` and `extras.hires.refiner` (step-pointer semantics, not step-count semantics) with strict bounds validation in `routers/generation.py`.
- 2026-02-08: Img2img numeric parsing now uses finite-float validation for core + hires float fields in `routers/generation.py` (rejects `NaN`/`Infinity` with HTTP 400).
- 2026-02-09: Task contracts are now typed in `task_registry.py` (`TaskEventType`, `TaskCancelMode`, `TaskStatusStage`) with strict non-terminal event normalization and fail-loud cancel-mode parsing (`immediate`/`after_current` only).
- 2026-02-10: `/api/engines/capabilities` now emits `asset_contracts` keyed by semantic engine (owner-resolved from canonical engine ids) so frontend semantic gating cannot drift from alias-heavy `engine_id_to_semantic_engine` maps.
- 2026-02-10: `dependency_checks.py` now resolves semantic asset contracts only via `contract_owner_for_semantic_engine(...)` (no local semantic-map duplication), so readiness rows fail loud if semantic/contract ownership drifts.
- 2026-07-30: `dependency_checks.py` keeps Qwen Image readiness exact: it counts only `qwen_image_ckpt`, `qwen_image_tenc`, and `qwen_image_vae` scoped roots and independently validates the complete vendored Edit-2511 model-index, processor/tokenizer, text-encoder, transformer, VAE, and scheduler metadata set.
- 2026-07-30: `routers/generation.py` admits Qwen Image only on canonical Edit-2511 img2img and rejects Qwen txt2img/public variant keys plus unsupported edit surfaces before task creation.
- 2026-07-31: Qwen Image img2img validates its exact top-level allowlist, required core fields, init-image payload, and exact asset selectors before task registration. `img2img_extras.model_sha` is the sole transformer selector; top-level `model` is rejected.
- 2026-02-15: Generation payloads now require `settings_revision` to match persisted options revision; stale requests fail with HTTP 409 (`current_revision` + `provided_revision`) and top-level `smart_*` payload keys are rejected.
- 2026-02-15: `run_api.py` startup settings normalization preserves `codex_options_revision` while pruning unknown/invalid persisted keys.
- 2026-02-15: `POST /api/options` responses now include `applied_now[]` and `restart_required[]` arrays with per-key reason metadata.
- 2026-02-15: `run_api.py` now publishes launcher trace toggles (`trace_contract`, `trace_profiler`) through bootstrap env keys (`CODEX_TRACE_CONTRACT`, `CODEX_TRACE_PROFILER`) and maps profiler toggle to `CODEX_PROFILE=1` for runtime diagnostics.
- 2026-02-15: Task error payloads now pass through `public_errors.py`; API task status/SSE channels expose public-safe terminal messages (`cancelled`/`out of memory`/stable error id) instead of raw exception text.
- 2026-02-15: `public_errors.py` also sanitizes synchronous HTTP error details for generation/upscale/supir routes (`public_http_error_detail`), removing raw exception text from `HTTPException.detail` and `/api/upscalers/remote` manifest parse errors while preserving actionable OOM classification.
- 2026-02-15: `public_errors.py` now keeps `EngineExecutionError` messages visible in task channels using stable `engine error: ...` formatting (idempotent on replay/snapshot re-serialization), so frontend task error panels can surface actionable runtime failures instead of opaque `internal error (error_id=...)`.
- 2026-02-16: Generation task workers now also emit explicit API-console logs for typed `EngineExecutionError` (`task_id` + `mode` + `engine` + message) before public-error sanitization, so local runtime failures remain visible in backend logs without changing task/SSE payload contracts.
- 2026-02-18: `run_api.py` phase-1 logging cleanup removed the `logging.basicConfig(...)` fallback in `ensure_initialized()` (fail-loud on logging bootstrap failure) and migrated startup/settings/port-guard/init console `print(...)` messages to structured logger calls.
- 2026-05-02: `run_api.py` LoRA apply-mode bootstrap publication now publishes resolved `CODEX_LORA_APPLY_MODE` when needed to override a conflicting process env while still avoiding implicit default pinning; LoRA math/signature bootstrap toggles remain published only for non-default values.
- 2026-02-20: `json_store.py` is now fail-loud for persistence faults: `_load_json` returns `{}` only for missing files and raises on parse/read/type violations; `_save_json` raises on write/serialization failures (no best-effort swallow).
- 2026-02-21: `run_api.py` startup settings normalization now parses checkbox values via shared strict bool parser and fails startup on invalid checkbox literals (no silent coercion of unknown strings to `False`).
- 2026-02-21: UI persistence routes now fail loud on malformed `tabs.json`/`workflows.json`/`presets.json` payloads (no silent fallback to defaults or empty objects), and `/api/options` now rejects out-of-range numeric values instead of silently clamping (aligned with `/api/options/validate`).
- 2026-02-21: `run_api.py` checkbox startup normalization now canonicalizes persisted checkbox values to strict `bool` type (including `0/1` -> `False/True`) to prevent numeric-bool type drift in `settings_values.json`.
- 2026-02-22: `routers/system.py` adds `POST /api/obliterate-vram` (quick-settings VRAM cleanup) with safe default behavior: internal runtime cleanup always runs, external process termination is opt-in via `external_kill_mode='all'`, and critical/process-self protections are enforced with structured report output for UI feedback.
- 2026-02-22: `inference_gate.py` release semantics now track lock ownership per acquisition (thread-local marker) so gate release remains deterministic even if `CODEX_SINGLE_FLIGHT` changes between acquire/release.
- 2026-02-22: WAN video request normalization now enforces strict GGUF runtime controls in-router (`gguf_cache_policy`/`gguf_cache_limit_mb` and `gguf_te_device`) with synchronous HTTP 400 on invalid combinations/values.
- 2026-03-13: API device selection stays explicit (`device_selection.py`): missing payload device is rejected fail-loud, and per-request device overrides that diverge from the configured main device are also rejected.
- 2026-03-13: `dependency_checks.py` now describes LTX2 checkpoint mixes in terms of the canonical asset contract: every LTX2 checkpoint still requires exactly 1 external Gemma3-12B text encoder, while core-only GGUF checkpoints additionally require an external video VAE; embeddings connectors and the combined audio bundle resolve from configured LTX2 roots.
- 2026-03-16: `dependency_checks.py` now exposes an explicit LTX2 `vendored_metadata` readiness row backed by the same fail-loud vendored-runtime validator used by loader assembly: the local `Lightricks/LTX-2` repo must provide `model_index.json`, tokenizer assets, and readable config dirs for `text_encoder`, `scheduler`, `connectors`, `transformer`, `vae`, `audio_vae`, and `vocoder` before the frontend unblocks LTX generation.
- 2026-03-26: LTX `two_stage` vendor gating is profile-scoped, not global: the common `vendored_metadata` row stays on shared runtime metadata, while `latent_upsampler/config.json` is enforced only when `/api/models` admissibility or the router-owned `ltx_execution_profile='two_stage'` lane actually asks for that profile.
- 2026-02-23: `device_selection.configured_main_device()` now resolves main-device from live memory-manager authority first (`manager.primary_device()`), then args/env fallback; this removes stale bootstrap-only device reads after hot device updates.
- 2026-03-31: SUPIR no longer has a standalone generation route/task under `/api/supir/enhance`; `routers/supir.py` is diagnostics-only and the live SUPIR-mode request contract is owned by `routers/generation.py` on canonical SDXL img2img/inpaint surfaces.
- 2026-02-23: `run_api.py` now logs effective allocator bootstrap state at startup (`PYTORCH_CUDA_ALLOC_CONF`, resolved `backend`, and `--cuda-malloc` flag).
- 2026-02-24: `run_api.py` now applies a Windows-only process startup patch via `SetProcessInformation(ProcessPowerThrottling)` to disable execution-speed throttling (EcoQoS background throttle path), with explicit startup logs for success/failure.
- 2026-02-24: Windows power-throttling startup patch now logs decoded Win32 error text (`ctypes.WinError`) on failure and warns explicitly when `SetProcessInformation` is unavailable in the runtime.
- 2026-03-01: `run_api.py` trace-debug bootstrap now treats category flags independently (`CODEX_TRACE_INFERENCE_DEBUG`, `CODEX_TRACE_LOAD_PATCH_DEBUG`, `CODEX_TRACE_CALL_DEBUG`): any one of them forces DEBUG visibility (`CODEX_LOG_DEBUG=1` + DEBUG logging bootstrap), while global call tracing via `sys.setprofile` remains exclusive to call-trace toggles.
- 2026-03-03: `run_api.py` settings-registry import comment now uses a generic generator note (no `.sangoi` path mention inside backend runtime code).
- 2026-03-05: `run_api.py` uvicorn access-noise filter now suppresses both `/api/tools/convert-gguf/*` and `/api/tools/merge-safetensors/*` polling endpoints to keep long-running tools jobs from flooding INFO access logs.

### ABSOLUTE LAW — DO NOT TOUCH LAYER NAMES

Read this twice.

If a checkpoint says a layer is `foo.bar.baz`, then in this repository it stays `foo.bar.baz`.
Not `bar.baz`.
Not `foo_bar_baz`.
Not "normalized".
Not "close enough".

Keymaps here **map keyspaces** so the engine/runtime can understand how different ecosystems name the same conceptual weight.
They do **not** rename stored model keys.
They do **not** strip prefixes.
They do **not** rewrite punctuation.
They do **not** slide dots around.
They do **not** materialize remapped state dicts.

If two ecosystems use different names for the same conceptual weight, the keymap resolves that relationship explicitly and the engine interprets the stored key as-is.
If the layout is unsupported, ambiguous, or structurally incompatible, you fail loud and extend the keymap properly.
You do **not** "normalize" a checkpoint by rewriting layer names in memory. Ever.

Lazy mappings are law too.
Checkpoint/state-dict seams stay lazy by default.
Keymaps and loaders inspect checkpoints through mapping/view APIs first (`shape_of(...)`, lookup views, computed views) and only touch real tensor values at the owner seam that actually needs them.
You do **not** materialize eager `dict(...)` copies of checkpoint mappings as convenience glue.

---

### ABSOLUTE LAW — USE THE LOCAL TEST TOOLCHAIN

Repository tests and validation commands use this checkout's toolchains.

- Python commands must run through local `uv`: `./.uv/bin/uv run --python .venv/bin/python --no-sync ...`.
- Backend CPU checks must keep the explicit CPU env pattern: `CODEX_ROOT="$PWD" PYTHONPATH="$PWD" CODEX_TORCH_MODE=cpu CODEX_TORCH_BACKEND=cpu`.
- Node/npm/frontend commands must use the local nodeenv by prepending `"$PWD/.nodeenv/bin"` to `PATH` from the repository root.
- Do **not** use system/global `python`, `uv`, `node`, `npm`, or `npx` for repository tests or validation.
- If `./.uv/bin/uv`, `.venv/bin/python`, or `.nodeenv/bin/{node,npm,npx}` is missing, stop and report the missing local toolchain path unless the user explicitly requests non-local execution for that command.

---

### ACT II – WHERE THE TRUTH LIVES: `.sangoi`, REUSE, AND THE WEBUI ATLAS

Before broad grep, use the WebUI Atlas below as the prompt-resident owner map for this repository.
If you don't know what to change, you do not guess. You open the Atlas section for the hotspot, pipeline, or owner seam first, then open the canonical owner path before followers.

Contract authority lives here in root `AGENTS.md` for policy and in `.sangoi/reference/**` for detailed contracts. Keep those roles split.
The Atlas is discovery-only: hotspot routing, bounded pipeline node chains, owner seams, generated artifact pointers, and policy pointers.
Do not turn the Atlas into contract matrices, migration ledgers, drift reports, or mutable backend counts/timestamps.
Generated reports under `.sangoi/reports/tooling/` own mutable backend snapshots, counts, hashes, and timestamps.

When a mapped node chain changes because of file moves, owner-path changes, public route additions/removals, or top-level owner functions changing, update this Atlas in the same tranche.
If a touched `apps/**` file is part of an Atlas pipeline, hotspot, or owner seam entry, refresh its file header block in the same tranche. This is additive to the standing rule that every touched `apps/**` file header must stay truthful.
When the Atlas or backend discovery surface changes, run this operational checklist before handoff:

- Keep the Atlas discovery-only; no contract matrices, drift ledgers, or stale migration notes.
- Ensure hotspot discoverability stays explicit for keymaps, `vae_codex3d.py`, and `hires_fix.py`.
- Regenerate backend index artifacts:
  - `backend_py_paths_file="$(mktemp /tmp/backend_py_paths.XXXXXX.txt)"`
  - `git ls-files apps/backend | rg "\\.py$" | LC_ALL=C sort > "$backend_py_paths_file"`
  - `python3 .sangoi/.tools/dump_apps_file_headers.py --out .sangoi/reports/tooling/apps-backend-file-header-blocks.md --root apps/backend --fail-on-missing`
  - `python3 .sangoi/.tools/build_backend_py_book_index.py --paths "$backend_py_paths_file" --headers .sangoi/reports/tooling/apps-backend-file-header-blocks.md --out .sangoi/reports/tooling/backend-py-book-index.md`
- Validate parity/checks:
  - `python3 .sangoi/.tools/build_backend_py_book_index.py --paths "$backend_py_paths_file" --headers .sangoi/reports/tooling/apps-backend-file-header-blocks.md --out .sangoi/reports/tooling/backend-py-book-index.md --check`
  - `bash .sangoi/.tools/link-check.sh .sangoi`
  - `bash .sangoi/.tools/link-check.sh .`

If you touch an `apps/**` source file, you keep its **file header block** honest. If the purpose or top-level symbols changed, you update them.

- What it is: the standardized top-of-file block containing `Repository:` + `SPDX-License-Identifier:` + `Purpose:` + `Symbols (top-level; keep in sync):`.
- Where it lives: `.py` = module docstring (first statement); `.ts` = top block comment (`/* ... */`); `.vue` = top HTML comment (before `<template>`).
- Standard: `.sangoi/policies/file-header-block.md`. Helper: `python3 .sangoi/.tools/review_apps_header_updates.py`.

## WebUI Atlas

Last reviewed on 2026-08-30 during the SeedVR2 R2 admission, cancellation, and media-correctness tranche.

<!-- Merge-safety anchor: this prompt-resident WebUI Atlas replaces the former split-file discovery front door; update it whenever a hot path, owner file, public route, or shipped entrypoint moves. -->

### How to use this atlas

- Optimize for owner-first navigation: open the governing seam before opening followers, generated reports, tests, or wide UI files.
- Treat this Atlas as the default "open this first" map so a fresh session does not need broad grep just to rediscover the runtime shape.
- Follow one hot path top-down before widening sideways. Most wrong reads here come from opening followers in parallel before the governing seam is locked.
- When you land in a folder that already has its own `AGENTS.md`, read that local sub-atlas immediately before editing inside that folder.
- Treat this Atlas as a first-pass routing map, not permission to guess gated names, stale symbols, or follower behavior without checking the live owner file.
- Use `.sangoi/reports/tooling/backend-py-book-index.md` only for generated backend file discovery and counts; it is not policy authority.

### High-churn hotspots and fanout seams

- Keymaps: `apps/backend/runtime/state_dict/key_mapping.py`
  - Aliases: checkpoint keyspace mapping; WAN LoRA logical keys; Anima raw `net.*` core keyspace.
  - Open this when mapping upstream checkpoint keyspaces into canonical runtime lookup views without rewriting stored keys.
  - Secondary seams: `apps/backend/runtime/state_dict/keymap_anima_transformer.py`, `apps/backend/runtime/state_dict/keymap_wan22_transformer.py`, `apps/backend/runtime/state_dict/AGENTS.md`.
- Native Conv3D VAE: `apps/backend/runtime/common/vae_codex3d.py`
  - Aliases: codex3d vae; conv3d native.
  - Open this for WAN-style native 3D VAE load/detect, shift/scale policy, or codex 3D lane questions.
  - When `memory_management.manager.vae_always_tiled` selects WAN VAE tiling, `AutoencoderCodex3D.enable_tiling()` caps spatial-attention query chunks while preserving global K/V and temporal-cache semantics.
  - Secondary seams: `apps/backend/runtime/families/wan22/vae_io.py`, `apps/backend/runtime/state_dict/keymap_wan22_vae.py`.
- Hires Fix: `apps/backend/runtime/pipeline_stages/hires_fix.py`
  - Aliases: hires second pass; high-res fix.
  - Open this for second pass, hires geometry, continuation semantics, or hires telemetry ownership.
  - Secondary seams: `apps/backend/runtime/families/sd/hires_fix.py`, `apps/backend/use_cases/img2img.py`.
- Image Automation: `apps/backend/use_cases/image_automation.py`
  - Aliases: infinite generate; folder loop; wildcard batch.
  - Open this for repeat/infinite runs, prompt-list/wildcard expansion, folder source cycling, or automation summaries.
  - Secondary seams: `apps/backend/interfaces/api/tasks/generation_tasks.py`, `apps/backend/interfaces/api/routers/generation.py`.
- IP-Adapter: `apps/backend/runtime/pipeline_stages/ip_adapter.py`
  - Aliases: image prompt adapter; reference image conditioning.
  - Open this for adapter model/image-encoder selection, reference-image preprocessing, or per-sampling patch application.
  - Secondary seams: `apps/backend/runtime/adapters/ip_adapter/`, `apps/backend/interfaces/api/routers/generation.py`.
- SUPIR Runtime: `apps/backend/runtime/families/supir/runtime.py`
  - Aliases: supir mode; sdxl restore; img2img restore branch.
  - Open this for native SDXL img2img/inpaint SUPIR mode, restore-anchor sampling, or SUPIR variant/base-checkpoint validation.
  - Secondary seams: `apps/backend/use_cases/img2img.py`, `apps/backend/interfaces/api/routers/generation.py`, `apps/backend/runtime/families/supir/loader.py`.
- attention_sram_v1: `apps/backend/runtime/attention/sram/__init__.py`
  - Aliases: SRAM attention; split-KV; shared-memory attention.
  - Open this for runtime bridge, extension build/warmup, split-KV heuristics, or shared-memory kernel lane work.
  - Secondary seams: `apps/backend/runtime/kernels/attention_sram_v1/`, `apps/backend/runtime/families/wan22/model.py`.
- Generation Router: `apps/backend/interfaces/api/routers/generation.py`
  - Aliases: generation endpoints; request parsing; SUPIR preflight.
  - Open this for public generation routes, payload parsers, task spawn, or route-level fail-loud guards.
  - Secondary seams: `apps/backend/interfaces/api/tasks/generation_tasks.py`, `apps/backend/interfaces/api/run_api.py`, `apps/backend/runtime/logging.py`.
- Task Streaming: `apps/backend/interfaces/api/routers/tasks.py`
  - Aliases: task SSE; replay buffer; `/api/tasks/{id}/events`.
  - Open this for task snapshots, SSE replay/gap recovery, terminal result/end emission, or cancellation semantics.
  - Secondary seams: `apps/backend/interfaces/api/task_registry.py`, `apps/backend/interfaces/api/tasks/generation_tasks.py`.
- Orchestrator: `apps/backend/core/orchestrator.py`
  - Aliases: engine dispatch; model load/cache; runtime coordinator.
  - Open this for engine resolution, load/unload/cache ownership, or wrapper dispatch.
  - Secondary seams: `apps/backend/core/engine_interface.py`, `apps/backend/engines/common/base.py`.
- Qwen Streamed Core Residency: `apps/backend/runtime/families/qwen_image/streaming.py`
  - Aliases: Qwen block streaming; Edit-2511 mixed residency; CPU-backed Qwen transformer.
  - Open this for mandatory blocked-streaming activation, static residency, one-block CUDA transitions, host-object restoration, or residency telemetry.
  - Secondary seams: `apps/backend/runtime/families/qwen_image/streaming_slots.py`, `apps/backend/runtime/families/qwen_image/transformer.py`, `apps/backend/runtime/families/qwen_image/loader.py`, `apps/backend/runtime/families/qwen_image/runtime.py`.

### Pipeline owner paths

#### txt2img

- Public route: `apps/backend/interfaces/api/routers/generation.py` (`/api/txt2img`)
  - Validates payload, route capability, and explicit device; creates `TaskEntry`; starts `run_txt2img_task`.
- Shared image task worker: `apps/backend/interfaces/api/tasks/generation_tasks.py`
  - Calls `prepare_txt2img(...)`, owns inference-gate/task lifecycle, and reuses shared image result packaging before `InferenceOrchestrator.run(...)`.
- Orchestrator: `apps/backend/core/orchestrator.py`
  - Resolves engine registry/load/cache state and dispatches to the engine `txt2img(...)` wrapper.
- Engine wrapper: `apps/backend/engines/common/base.py`
  - Delegates the mode to `run_txt2img(...)` instead of owning a second pipeline.
- Canonical use-case: `apps/backend/use_cases/txt2img.py`
  - Owns progress/result emission, decode, and cleanup inside the worker-thread envelope.
- Stage runner: `apps/backend/use_cases/txt2img_pipeline/runner.py`
  - Executes the staged txt2img pipeline and returns the `GenerationResult`.
  - Exact `zimage_l2p` remains inside this canonical runner: it guards unsupported state before conditioning, calls the engine pixel sampler hook, and returns the pixel tensor in `GenerationResult.decoded` so no VAE decode fallback runs.
  - Parked exact `lens` hook contract remains inside this canonical runner: it guards unsupported state before generic prompt application/RNG/conditioning, calls the engine Lens txt2img hook, and expects decoded RGB output in `GenerationResult.decoded` so no classic VAE decode fallback runs; public/default capabilities still keep Lens parked.
- Terminal surfaces: `apps/backend/interfaces/api/tasks/generation_tasks.py` and `apps/backend/interfaces/api/routers/tasks.py`
  - Encode/save images, store the result payload, and expose terminal result/end through `GET /api/tasks/{id}` and `/api/tasks/{id}/events`.

#### img2img

- Public route: `apps/backend/interfaces/api/routers/generation.py` (`/api/img2img`)
  - Validates payload and route capability, rejects masked requests when semantic capability says masking is unsupported, validates exact-engine `img2img_inpaint_mode`, preflights native `img2img_extras.supir` and exact SDXL Fooocus/BrushNet assets, then creates the task and picks the explicit device.
- Shared image task worker: `apps/backend/interfaces/api/tasks/generation_tasks.py`
  - Calls `prepare_img2img(...)`, owns inference-gate/task lifecycle, and packages the terminal image result.
- Orchestrator: `apps/backend/core/orchestrator.py`
  - Resolves engine/load/cache state and dispatches to the mode wrapper.
- Engine wrapper: `apps/backend/engines/common/base.py`
  - Delegates to `run_img2img(...)`.
- Canonical use-case: `apps/backend/use_cases/img2img.py`
  - Owns the early exact Qwen Image Edit-2511 branch, classic-family dispatch, init-image planning, prompt/sampling plans, optional native SUPIR mode, optional masked img2img, exact-engine SDXL Fooocus/BrushNet branching, and optional hires continuation.
- Shared stage helpers: `apps/backend/runtime/pipeline_stages/masked_img2img.py`, `apps/backend/runtime/pipeline_stages/sampling_execute.py`, `apps/backend/runtime/families/sd/fooocus_inpaint.py`, `apps/backend/runtime/families/sd/brushnet.py`, and `apps/backend/runtime/pipeline_stages/hires_fix.py`
  - Prepare generic masked bundles/image conditioning/hires latents; `sampling_execute.py` activates the post-LoRA sampling snapshot and enters request-scoped SDXL Fooocus/BrushNet patch sessions before sampler/IP-Adapter.
- Terminal surfaces: `apps/backend/interfaces/api/tasks/generation_tasks.py` and `apps/backend/interfaces/api/routers/tasks.py`
  - Store the encoded result payload and expose terminal snapshot/SSE state.
- Branch notes:
  - Qwen Image Edit-2511 branches before classic backend resolution: the canonical use-case sequences family conditioning, reference VAE encode, native denoise, and final VAE decode, then returns decoded RGB through `GenerationResult` without entering classic `sampling_execute.py`.
  - Qwen denoise uses the canonical memory-manager streamed-residency contract: the transformer GGUF remains CPU-backed, static owners plus exactly one transformer block are CUDA-resident, and the text encoder and VAE are offloaded before core sampling.
  - Classic base img2img resolves SD-vs-flow dispatch locally in `apps/backend/use_cases/img2img.py` before masked/unmasked prep.
  - SDXL SUPIR mode stays inside canonical img2img: route preflight lives in `apps/backend/interfaces/api/routers/generation.py`; request-scoped restore runtime lives in `apps/backend/runtime/families/supir/runtime.py`.
  - SDXL exact-engine inpaint stays inside canonical img2img: exact-engine mode/asset preflight lives in `apps/backend/interfaces/api/routers/generation.py`; `img2img.py` installs the temporary request-scoped sampling-session factory only for non-SUPIR exact modes; `sampling_execute.py` enters the Fooocus/BrushNet session after canonical LoRA activation.
  - Kontext-specific img2img work stays local to `apps/backend/use_cases/img2img.py`.
  - FLUX.2 keeps its own engine-side img2img seam at `apps/backend/engines/flux2/img2img.py`; the public route still enters through the same router/task/orchestrator chain.

#### image_automation

- Public route: `apps/backend/interfaces/api/routers/generation.py` (`/api/image-automation`)
  - Parses the backend-owned automation contract, selects the explicit device from the template payload, and creates `run_image_automation_task(...)`.
- Automation task worker: `apps/backend/interfaces/api/tasks/generation_tasks.py`
  - Owns task lifecycle, inference gate, per-iteration execution wrapper, automation summary, and terminal task result storage.
- Automation loop owner: `apps/backend/use_cases/image_automation.py`
  - Expands prompts/wildcards, selects folder/init/reference inputs, manages loop cancellation, and emits iteration/progress events.
- Iteration delegate: `apps/backend/interfaces/api/tasks/generation_tasks.py` (`_execute_prepared_image_request`)
  - Turns each prepared iteration back into the canonical txt2img/img2img request path.
- Underlying mode: `apps/backend/use_cases/txt2img.py` or `apps/backend/use_cases/img2img.py`
  - Executes the actual generation for one iteration.
- Terminal surfaces: `apps/backend/interfaces/api/tasks/generation_tasks.py` and `apps/backend/interfaces/api/routers/tasks.py`
  - Publish `automation_iteration` events, store the final automation summary/result, and expose replay through task snapshot/SSE.

#### txt2vid

- Public route: `apps/backend/interfaces/api/routers/generation.py` (`/api/txt2vid`)
  - Validates payload/capability, rejects legacy aliases, creates the task, and selects the explicit device.
- Shared video task worker: `apps/backend/interfaces/api/routers/generation.py`
  - Parses the request, owns task lifecycle for video modes, and stores the terminal result payload.
- Orchestrator: `apps/backend/core/orchestrator.py`
  - Resolves engine/load/cache state and dispatches to the engine `txt2vid(...)` wrapper.
- Engine wrapper: `apps/backend/engines/ltx2/ltx2.py` or `apps/backend/engines/wan22/wan22_14b.py`
  - Delegates to `run_txt2vid(...)` for the active engine family.
- Canonical use-case: `apps/backend/use_cases/txt2vid.py`
  - Owns execution-profile branching, shared video plan/export helpers, optional interpolation, and terminal `ResultEvent` emission.
- Shared video helpers: `apps/backend/runtime/pipeline_stages/video.py`
  - Own `build_video_plan(...)`, `build_ltx2_video_plan(...)`, WAN Diffusers stage-LoRA preflight/apply, export helpers, and post-generation video stages.
- Terminal surfaces: `apps/backend/interfaces/api/routers/generation.py` and `apps/backend/interfaces/api/routers/tasks.py`
  - Store the final result in the task entry and expose terminal snapshot/SSE state.
- Branch notes:
  - LTX2 keeps its execution-profile branch inside `apps/backend/use_cases/txt2vid.py`.
  - WAN22 keeps GGUF/diffusers branch decisions inside the same use-case after wrapper/orchestrator hand-off.

#### img2vid

- Public route: `apps/backend/interfaces/api/routers/generation.py` (`/api/img2vid`)
  - Validates payload/capability, applies route-specific preflight, creates the task, and selects the explicit device.
- Shared video task worker: `apps/backend/interfaces/api/routers/generation.py`
  - Parses the request, owns task lifecycle, and stores the terminal result payload.
- Orchestrator: `apps/backend/core/orchestrator.py`
  - Resolves engine/load/cache state and dispatches to the engine `img2vid(...)` wrapper.
- Engine wrapper: `apps/backend/engines/ltx2/ltx2.py` or `apps/backend/engines/wan22/wan22_14b.py`
  - Delegates to `run_img2vid(...)` for the active engine family.
- Canonical use-case: `apps/backend/use_cases/img2vid.py`
  - Owns image-video execution profiles, WAN temporal-mode branching, shared video plan/export helpers, and terminal `ResultEvent` emission.
- Shared video helpers: `apps/backend/runtime/pipeline_stages/video.py`
  - Own plan/export/interpolation helpers shared with txt2vid plus WAN Diffusers stage-LoRA preflight/apply.
- Terminal surfaces: `apps/backend/interfaces/api/routers/generation.py` and `apps/backend/interfaces/api/routers/tasks.py`
  - Store the final result in the task entry and expose terminal snapshot/SSE state.

#### video_upscale

- Public route: `apps/backend/interfaces/api/routers/upscale.py` (`/api/video-upscale`)
  - Before task registration, validates one backend-visible video path, exact decoded count/timing and stream origins, the curated SeedVR2 controls, upstream-matched target geometry, current host-memory/scratch capacity, and a configured CUDA or MPS device. It does not accept browser-uploaded video bytes.
- Dedicated task worker: `apps/backend/interfaces/api/tasks/video_upscale_tasks.py`
  - Owns inference-gate coordination, progress forwarding, cancellation, terminal result/error state, and cleanup for one dedicated video-upscale task.
- Canonical use-case: `apps/backend/use_cases/video_upscale.py`
  - Consumes the admitted source/resource evidence and owns SeedVR2 frame execution, task-work cleanup, verified H.264 MP4 export, and terminal `ResultEvent` emission.
- Runtime and file owners: `apps/backend/video/io/ffmpeg.py`, `apps/backend/video/upscaling/seedvr2.py`, and `apps/backend/video/export/ffmpeg_exporter.py`
  - Respectively own cancellable exact probing/timing/extraction, uniform-padding-aware SeedVR2 execution with complete child-tree cleanup, and relative A/V-origin-verified sidecar-before-MP4 publication under the existing output route.
- Terminal surfaces: `apps/backend/interfaces/api/task_registry.py` and `apps/backend/interfaces/api/routers/tasks.py`
  - Keep task snapshot, SSE replay, cancellation, and output-file serving unchanged.

#### vid2vid

- Public route: `apps/backend/interfaces/api/routers/generation.py` (`/api/vid2vid`)
  - Current public truth: the route is parked and fails fast with `400` before staging/task creation while no families are implemented.
- Dormant wrapper: `apps/backend/engines/wan22/wan22_14b.py`
  - In-tree wrapper hook still exists for the future re-enable path.
- Dormant use-case: `apps/backend/use_cases/vid2vid.py`
  - Holds the bounded vid2vid execution owner once the public route is re-enabled.

#### API bootstrap

- App bootstrap: `apps/backend/interfaces/api/run_api.py`
  - Builds the FastAPI app, validates startup/runtime settings, and mounts routers. Repo-owned bootstrap/server logs consume the canonical wrapper family from `apps/backend/runtime/logging.py`; the remaining raw logger carve-out is the explicit `uvicorn.access` seam.
- Router mount: `apps/backend/interfaces/api/run_api.py`
  - Includes `system`, `settings`, `ui`, `models`, `paths`, `options`, `tasks`, `tests`, `tools`, `upscale`, `supir`, and `generation`; the `supir` router is diagnostics-only.
- Public entry: router modules under `apps/backend/interfaces/api/routers/`
  - Expose task-backed generation/system/tool surfaces.

### Owner seam map

- Generation Router seam: `apps/backend/interfaces/api/routers/generation.py`
  - Owns public generation routes, payload parsing, route-level capability guards, exact-engine SUPIR-mode preflight for canonical img2img/inpaint, exact `zimage_l2p` no-VAE txt2img admission, task creation, and worker thread hand-off.
  - Do not move mode execution into this file; it stays validate + dispatch + stream.
- Video Upscale utility seam: `apps/backend/interfaces/api/routers/upscale.py`, `apps/backend/interfaces/api/tasks/video_upscale_tasks.py`, and `apps/backend/use_cases/video_upscale.py`
  - The router validates and dynamically admits the dedicated source-path request before task creation. The worker owns task lifecycle. The use case owns one video-upscale pipeline. Reuse the cancellable video I/O, SeedVR2 runner, verified export, task registry, and SSE owners instead of reactivating vid2vid or adding a browser upload path.
- Image Task Worker: `apps/backend/interfaces/api/tasks/generation_tasks.py`
  - Owns shared image task lifecycle, inference-gate integration, encoded image result packaging/save/provenance hooks, and automation task wrapper around canonical image modes.
  - Open this file when the question is task result packaging rather than public payload parsing.
- Task Registry + SSE: `apps/backend/interfaces/api/task_registry.py` and `apps/backend/interfaces/api/routers/tasks.py`
  - Own in-memory task snapshots, bounded replay buffer and gap detection, terminal `result|error|end` emission, and cancellation API.
  - Open these files first when reconnect/replay/status drift is the seam.
- Backend Logging seam: `apps/backend/runtime/logging.py`
  - Owns normalized repo-owned logger acquisition via `get_backend_logger(...)`, logger-style repo-owned emission through `BackendLoggerProxy`, human-readable wrapper emission via `emit_backend_message(...)`, structured telemetry emission via `emit_backend_event(...)`, and canonical uvicorn/bootstrap logging config via `build_backend_uvicorn_log_config(...)`.
  - Secondary seams: `apps/backend/interfaces/api/run_api.py`, `apps/backend/runtime/diagnostics/error_summary.py`, `apps/backend/runtime/diagnostics/exception_hook.py`, `apps/backend/infra/stdio.py`.
  - Open this for logger namespace normalization, repo-owned backend log formatting/emission, bootstrap/server log config, or the split between operational logs, concise runtime summaries, full exception dumps, and exact stdout/stderr contracts.
- Orchestrator seam: `apps/backend/core/orchestrator.py`
  - Owns engine registry lookup, load/unload/cache residency decisions, and dispatch from task workers into engine wrappers.
  - It coordinates execution; it is not the terminal result owner.
- Shared pipeline stages: `apps/backend/runtime/pipeline_stages/`
  - High-value entries: `hires_fix.py`, `masked_img2img.py`, `video.py`, `ip_adapter.py`.
  - Open this directory when the question is a shared stage reused by multiple canonical use-cases.
  - `video.py` is the current shared owner for WAN Diffusers stage-LoRA preflight/apply and shared generation video helpers across `txt2vid`, `img2vid`, and `vid2vid`. Dedicated SeedVR2 execution belongs to `use_cases/video_upscale.py` plus `video/upscaling/seedvr2.py`.

### Generated artifact and policy pointers

- Backend header snapshot: `.sangoi/reports/tooling/apps-backend-file-header-blocks.md`
  - Use it when you need the current backend file-header snapshot that feeds the grouped book index.
- Backend Python book index: `.sangoi/reports/tooling/backend-py-book-index.md`
  - Use it when you need the full grouped backend file list and current generated subtree/timestamp scope.
- Tool command catalog: `.sangoi/.tools/AGENTS.md`
  - Use it for current commands/failure modes for header dumps, backend book-index rebuilds, and link/header lint checks.
- Report catalogs: `.sangoi/reports/AGENTS.md` and `.sangoi/reports/tooling/AGENTS.md`
  - Use these to distinguish active report-catalog truth from retained audit artifacts.
- File header standard: `.sangoi/policies/file-header-block.md`
  - Use it when a touched `apps/**` file header needs truthful `Purpose` / `Symbols` sync.

---

### ACT III – GIT, COMMITS, AND HISTORY

`.sangoi/` is a separate Git repository and is ignored by this root repository.

- Root commits (`git add/commit` from this repo) do not include `.sangoi/**`.
- When a task targets `.sangoi`, run Git commands explicitly against that repo (`git -C .sangoi ...`).
- Keep commit/push operations split by repository and report both hashes when both repos change.

When your turn is done:

- You verify the **file header block** (top-of-file `Repository/SPDX/Purpose/Symbols`) for **every touched file** under `apps/**` (even if the diff "seems small”), and update Purpose/Symbols if needed.
- Use `python3 .sangoi/.tools/review_apps_header_updates.py --show-body-diff` to review "changed body, unchanged header” cases.
- If the touched `apps/**` file is referenced by a mapped node chain, hotspot entry, or owner seam in the WebUI Atlas, update this root `AGENTS.md` in the same tranche instead of leaving the Atlas stale.

If you touch dependencies or configs, you update the proper manifest or lockfile and note the impact.

---

### ACT IV – ARCHITECTURE, LEGACY, MODELS, PYTHON

- The default core for attention is PyTorch SDPA.
- You list risks, side effects, globals.
- Codex prefix or suffix is used where it actually adds meaning.
- `Codex` is an intentional project naming convention. Do **not** strip it just because a symbol looks long.
- If naming or structure is bad, fix the fake namespace, owner shape, module boundary, or alias soup. Do **not** "clean it up" by deleting the `Codex` prefix, and do **not** invent pseudo-namespaces like `CodexProcessing.Txt2Img` unless a real namespace or module exists.
- You always code in Codex style:
  - Dataclasses, enums and similar.
  - Small modules with clear seams.
  - Explicit and fail-loud errors.
  - Readable names.

Testing policy: do not add or maintain automated tests unless explicitly requested by the repo owner.
Prefer fail-loud runtime contracts and manual validation workflows.

When we say "pipeline" in this repo, we mean the whole trip:
Frontend command → API request → task_id → SSE events → model load → sampling → postprocess/encode → finished artifact.

Drift is not a vibe. Drift is a bug.
Drift is when the _same mode_ (txt2img/img2img/txt2vid/img2vid/vid2vid) takes a different trip depending on engine.
Drift Also counts as drift when any of this changes per engine for the same mode:

- Contract drift: request schema/defaults, progress semantics, preview semantics, error semantics, or result fields.
- Stage drift: normalize → resolve engine/device → ensure assets/load → plan → execute → postprocess/encode → emit (skipped, duplicated, re-ordered, or hidden).
- Ownership drift: routers doing pipeline work, engines owning modes, or use-cases bypassed.

**Policy (Option A): one canonical use-case per mode.**

- `apps/backend/use_cases/{txt2img,img2img,txt2vid,img2vid,vid2vid}.py` owns the mode pipeline.
- Engines are adapters and hooks. They load models and expose primitives. They do **not** re-implement the mode.
- Routers stay thin: validate + dispatch + stream.
- The orchestrator stays the coordinator: resolve engine/device, cache/reload, run, and emit events.
- Shared, reusable stages live in `apps/backend/runtime/pipeline_stages/`. If it's shared, it goes there. If it's not shared, it stays in the canonical use-case.

**Ownership law: one concept, one owner path.**

- If a concept already lives under a typed nested owner, it stays there. You do **not** mirror it into flat shadow fields for convenience.
- Examples of forbidden shadow-owner drift:
  - `processing.hires.swap_model` plus `processing.hires_swap_model`
  - `processing.hires.refiner` plus `processing.hires_refiner`
  - `processing.hires.*` plus `processing.hr_*`
  - nested selector/config ownership plus sibling `*_name` / `*_path` aliases
- If a callsite wants a flatter shape, redesign the callsite. Do **not** duplicate the owner.

**Native names stay native.**

- If a field is named after a native concept, it carries only that concept.
- `refiner` is the native SDXL refiner seam. It is **not** a generic model-swap bucket.
- Generic model swap must live under explicitly generic naming such as `swap_model`, with its own typed owner.
- `extras.swap_model` is the top-level first-pass stage config:
  - it owns `enable` + `switch_at_step` semantics for mid-generation base-pass swapping;
  - it is **not** selector-only.
- `extras.hires.swap_model` is the selector-only second-pass owner:
  - it replaces the whole hires engine for the second pass;
  - it does **not** grow stage-pointer fields.
- `extras.refiner` / `extras.hires.refiner` remain SDXL-native refiner stages only.
- When a public/runtime seam is renamed to the native owner, the old name dies everywhere in the same tranche: router payloads, frontend state, component props/emits, run helpers, docs, and AGENTS.
- `hires.checkpoint`-style ghosts are forbidden once `hires.swap_model` exists.

**Derived-plan law: execution-only, selector-free.**

- A derived plan/helper struct may carry computed execution values such as target size, step count, denoise, or chosen upscaler.
- A derived plan/helper struct must **not** own selectors, model references, checkpoint names, swap-model config, refiner config, modules, or any other request-shaped contract data.
- If a plan/helper struct needs to carry those fields, that struct should not exist; compute from the canonical typed owner instead.
- `HiResPlan`-style shadow containers are forbidden as a destination for contract ownership.

**Unsupported seams fail loud.**

- If a mode/surface does not support a typed seam yet, you do **not** hide the payload and continue.
- Hide or clear the control in the UI when possible, and still fail loud at request build/runtime boundaries if stale state survives.
- Example: img2img must not silently drop `swap_model` / refiner state that only exists for txt2img hires.
- Public-state law: if a seam exists in frontend state, request payloads, or router parsing, it must also have a real execution owner.
  - No hidden/store-only `swap_model` surfaces.
  - No request/runtime surfaces that quietly do nothing.

**Recurring failure classes: stop recreating these.**

- **Owner-path drift**
  - Symptom: a router, engine, or convenience helper starts owning mode/stage work that belongs to a canonical use-case or shared stage.
  - Do: keep mode ownership in `apps/backend/use_cases/*.py`; if exact sampling mutation is needed, install it from `apps/backend/use_cases/img2img.py` and enter it from `apps/backend/runtime/pipeline_stages/sampling_execute.py` after canonical LoRA activation.
  - Do **not**: wrap whole runtime branches earlier, add engine-specific mode pipelines, or move slot/layout ownership out of owner modules such as `apps/backend/runtime/adapters/ip_adapter/layout.py`.

- **Contract drift**
  - Symptom: UI, request, parser, processing, and runtime names/defaults/allowlists stop matching each other.
  - Do: cut over the whole seam in one tranche across `apps/interface/**`, `apps/backend/interfaces/api/routers/generation.py`, and the runtime/processing owners.
  - Do **not**: keep alias readers, stale allowlists, or half-renamed payloads.
  - Examples: `inpaintMode` / `img2img_inpaint_mode`; `switch_at_step`; `apps/backend/runtime/families/supir/runtime.py` resolving the loaded checkpoint from the canonical model reference.

- **Lazy/load-path drift**
  - Symptom: checkpoint helpers materialize mappings into `dict(...)`, move tensors too early, or rebuild IO semantics in convenience glue.
  - Do: keep mapping/view ownership in `apps/backend/runtime/state_dict/views.py`; keep checkpoint IO in `apps/backend/runtime/checkpoint/io.py`; keep logical-shape load truth at the loader seam.
  - Do **not**: materialize eager copies, normalize keys in helpers, or move load semantics into family-local shortcuts.

- **Keymap/keyspace drift**
  - Symptom: family-specific remap logic reappears in runtime helpers, stage loaders, or patch code instead of canonical keymap owners.
  - Do: put keyspace rules in `apps/backend/runtime/state_dict/*.py`, for example `apps/backend/runtime/state_dict/keymap_wan22_transformer.py`, then have runtime consumers call that owner.
  - Do **not**: add prefix strippers, rekey helpers, or stage-local logical remappers.

- **Parity drift**
  - Symptom: the same mode takes a different trip or emits different public semantics across engines/tabs without an approved canonical hook.
  - Do: keep mode trips aligned through shared owners such as `apps/backend/use_cases/img2img.py`, `apps/backend/runtime/pipeline_stages/video.py`, and the shared frontend results/history owners.
  - Do **not**: special-case one engine/tab by cloning request/result/progress logic into a peer-owned surface.
  - Example: Z-Image masked img2img stays on its truthful runtime path in `apps/backend/runtime/families/zimage/model.py`; it does **not** get shoved through SD-style image-conditioning just because another family uses that seam.

- **Fail-loud erosion**
  - Symptom: unsupported or incomplete states survive as silent fallback, quiet no-op, or “best effort” continuation.
  - Do: reject stale state at UI/request/runtime boundaries and raise explicit errors when a required seam has no real execution owner.
  - Do **not**: keep hidden/store-only surfaces, default missing mandatory stages, or silently accept zero compatible layers.
  - Examples: `apps/backend/patchers/lora_apply.py` must fail when compatibility is zero; `apps/backend/interfaces/api/routers/generation.py` must reject unsupported request surfaces before task creation.

**Pre-merge anti-rerun checklist**

- Is the owner path still singular and canonical?
- Did every public field/default/allowlist change across UI/request/runtime in the same tranche?
- Did the loader/keymap path stay lazy and single-owner?
- Did family-specific mapping/layout logic stay in the canonical owner module?
- Does the same mode still take the same trip across engines and tabs?
- Will unsupported or stale state fail loud instead of degrading quietly?

If an engine needs special behavior, you add a hook that the canonical use-case calls.
If you can't express it as a hook, you stop and redesign until you can.
No engine-specific pipelines. No zoo.

Imports outside `/apps` are banned.
Only `apps.*` lives in active code.

If a feature has not been implemented, you raise:

```python
NotImplementedError("<feature> not yet implemented")
```

Model loading is a minefield you cross with a map.
You follow `.sangoi/research/models/model-loading-efficient-2025-10.md`.

- Supporting a family in `diffusers` format does **not** delegate contract truth to external `diffusers` helpers/imports.
- If this repo supports a `diffusers` surface, classification, component requirements, and family-specific constraints stay in repo-owned loader/detector/parser seams.
- Family-native external asset slots stay explicit and named. Do **not** collapse multi-slot families into generic selector bags when the contract depends on slot identity.
- When debugging model/adapter/runtime integration, prefer bounded seam proofs before E2E guesses:
  - preprocess parity against the canonical processor;
  - encoder/load parity against the same checkpoint through the canonical repo seam;
  - projector/module parity against the real checkpoint tensors;
  - patch math parity in isolation;
  - binding/layout parity by translated parameter names, not width-only heuristics.
- Use real checkpoints plus the appropriate local/official reference shelf under `.refs/`, and keep each proof scoped to one seam so you know exactly what failed.

**IP-Adapter image-encoder postmortem: this was done wrong once, and never again.**

- A prior IP-Adapter implementation loaded the image encoder through bespoke helpers instead of the canonical loader stack:
  - `_normalize_image_encoder_state_dict(...)`
  - `cleaned_state_dict(...)`
  - `rekey_vision_state_dict(...)`
  - `convert_openclip_checkpoint(...)`
  - raw `nn.Module.load_state_dict(...)` inside `ClipVisionEncoder`
- That was wrong because it bypassed the exact repo mechanisms that already exist to keep model loading honest:
  - canonical keyspace resolution
  - lazy mapping/view ownership
  - `fail_on_key_name_rewrite(...)`
  - `safe_load_state_dict(...)`
- That bespoke path also violated the architectural rule already stated above:
  - it eagerly materialized checkpoint mappings into `dict(...)`
  - it rewrote stored keys in memory
  - it created component-specific loader drift
  - it caused avoidable RAM blow-ups during image-encoder load
  - it violated the canonical build/mount pattern by constructing the runtime module outside the memory-owner device/dtype path, then relying on later runtime offload wrappers to clean it up
- The correct rule is simple:
  - the IP-Adapter image encoder is **not** a special loader class
  - if a VAE or text encoder would load through canonical keymap/view resolution plus `safe_load_state_dict(...)`, then the image encoder must do the same
  - if a CLIP-vision layout needs support, extend the canonical loader/keymap ownership and keep it lazy
  - if the layout is unsupported, fail loud
  - if a component is memory-managed, the module itself must also be built/mounted through the canonical owner path before weight load: resolve the owner device/dtype first, construct under the same `using_codex_operations(**to_args, ...)` pattern used by the central loaders, place the module on that owner device/dtype, then call `safe_load_state_dict(...)`, and only after that rely on runtime offload/reload
- Never again:
  - do **not** add adapter-local “cleaned state dict” helpers, prefix strippers, rekey shims, or raw `module.load_state_dict(...)` shortcuts just to get an auxiliary component loading quickly
  - do **not** bypass the rewrite guard by normalizing keys before the canonical loader sees them
  - do **not** ship a bespoke image-encoder loader when the rest of the repo already has the right loading contract
  - do **not** assume that wrapping a module in `ModelPatcher` later makes an off-pattern build/load canonical; the birth/load path must already follow the memory-manager owner contract

Keymap law: see **ABSOLUTE LAW — DO NOT TOUCH LAYER NAMES** at the top of this file.
The same no-rename/no-strip/no-punctuation-rewrite rule applies during model loading and engine/runtime keyspace interpretation.

You prefer SafeTensors.
You call `torch.load(..., weights_only=True, mmap=True)` when it applies.

Keep Python disciplined.
You do not add shebangs to source files.

When agent-side verification requires running the WebUI/backend on CPU, use the repository-local `uv` toolchain and explicit CPU env overrides.

- Prefer local `uv`: `./.uv/bin/uv` (never system/global `uv` for this workflow).
- Required env for CPU lane: `CODEX_ROOT="$PWD" PYTHONPATH="$PWD" CODEX_TORCH_MODE=cpu CODEX_TORCH_BACKEND=cpu`.
- Example check command pattern:
  - `CODEX_ROOT="$PWD" PYTHONPATH="$PWD" CODEX_TORCH_MODE=cpu CODEX_TORCH_BACKEND=cpu ./.uv/bin/uv run --python .venv/bin/python --no-sync -m apps.backend.interfaces.api.run_api --help`
- When the API is running on the Windows host and you need to hit it from WSL, do **not** assume `localhost` forwarding works. Probe the WSL default gateway first and use it as the base URL:
  - `WINDOWS_API_HOST="$(ip route | awk '/default/ {print $3; exit}')"`
  - `curl "http://$WINDOWS_API_HOST:7850/api/version"`
  - Fall back to a public tunnel only if the default-gateway probe fails.
- WSL heavy-model safety rule: for LTX/WAN-class giant assets, default to header-only / metadata-only inspection in WSL. Do not materialize tensors, assemble full runtimes, initialize full pipelines, or run forward passes unless the user explicitly asks. Prefer GGUF metadata readers and SafeTensors header readers first.

---

### ACT V – FRONTEND, LAYOUT, AND CSS

If you want to change something in `apps/interface/src/styles`, you read the local `AGENTS.md` before you touch a single selector.
Ignore that, and your pull request does not pass.

Styles for `apps/interface/src/styles` are not a dumping ground.
Common rules belong where they will be reused.
Variants are named with intent.
Do not litter with vague utilities that hide confusion.

---

### ACT VI – `.refs/` REFERENCE SOURCES

This repo keeps a serious local reference shelf under `.refs/`. Use it before you reach for `web.run`.

What lives there right now:

- upstream or related codebases such as `ComfyUI`, `ComfyUI-GGUF`, `ComfyUI-SeedVR2_VideoUpscaler`, `Forge-A1111`, `diffusers`, `flash-attention`, `pytorch`, `sd-scripts`, `llama.cpp`, `k-diffusion`, `LyCORIS`, `adetailer`, `WanVideoWrapper`, `Stable-Video-Infinity`, `LightX2V`, `IP-Adapter` and `open-tv`
- git-backed extensions and nested repos inside `.refs/Forge-A1111`
- model/index/reference artifacts such as `hf-model-indexes`, normalized stream JSON dumps, and local Gemini helper/reference files

Rules:

- When you need upstream behavior, loader details, extension behavior, API shape, or model-family reference code, search `.refs/` first instead of flailing with `web.run`.
- Before researching any git-backed reference under `.refs/`, run `git -C <that-reference-repo> pull --ff-only` so you are reading the freshest code.
- In this repo, that pull-first rule applies to the current git-backed references under `.refs/`, including repositories like `ComfyUI`, `ComfyUI-GGUF`, `ComfyUI-SeedVR2_VideoUpscaler`, `diffusers`, `pytorch`, `sd-scripts`, `llama.cpp`, `k-diffusion`, `LyCORIS`, `adetailer`, `WanVideoWrapper`, `Stable-Video-Infinity`, `LightX2V`, `open-tv`, and the nested git-backed repos inside `.refs/Forge-A1111`.
- `.refs/` stays reference-only. Durable conclusions belong in tracked docs, comments, or contracts inside this repo, not as ephemeral memory.

You read them. You do not import them into `apps/**`. You do not copy them into active code. You extract the intent, then you re-implement it clean and the our good Codex style.

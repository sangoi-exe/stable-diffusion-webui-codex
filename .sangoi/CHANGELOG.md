2025-10-17

- Env: Upgraded pip toolchain in `~/.venv` and installed project requirements from `requirements_versions.txt`.
- Env: Upgraded local PyTorch to 2.9.0 (+cu128) and torchvision 0.24.0 via `pip install torch torchvision` (no index-url) in `~/.venv`.
- Env: Verified CUDA runtime 12.8 present (PyTorch reports `torch.version.cuda == '12.8'`). GPU not visible in sandbox; validate on host WSL2.
- Test: Added `scripts/smoke_infer.sh` (API-only smoke) and `scripts/ui_headless_click.sh` + `tools/ui-click-generate.js` for headless Chrome click test of the WebUI Generate button.

2025-10-18

- Fix: Strict JSON submit wiring
  - `modules/ui.py`: hidden slots now use `gr.JSON` for `txt2img_named_active`/`img2img_named_active`; server accepts JSON strings.
  - `javascript/ui.js`: `submit_named()` now injects object (not stringified JSON) into the hidden slot.
- Fix: JS submitters always attach strict payloads and export named handlers for Gradio `_js` hooks (see `javascript/ui.js`).
- Fix: Typed strict submit flow (`javascript/ui.js`) to satisfy `npm run typecheck` (added `StrictBuilder` typedef tightened to `IArguments`, error formatter, and UIWindow exports).
- UI: Restored VAE selector to checkpoint-style dropdown and moved text encoders to a separate multiselect (`modules_forge/main_entry.py`, `modules/ui_settings.py`, `javascript/ui.js`).
- Test: Added Playwright variant `scripts/ui_headless_click_pw.sh` + `tools/ui-click-generate.playwright.mjs` with local browsers cache.
- Fix: VAE/Text Encoder dropdown wouldn’t open due to global `.hidden { display: none !important; }` override. Removed the rule from `style.css`; dropdown overlays now render correctly under Gradio 5 while keeping overflow visible via `div.gradio-dropdown { overflow: visible !important; }`.
- UI: Removed legacy combined selector. Split Hires selector into `Hires VAE` (single) and `Hires Text Encoder(s)` (multi); kept hidden compatibility field `hr_vae_te` to preserve payload shape (`modules/ui.py`).
 - UI: Quicksettings VAE (`sd_vae`) and Text Encoders (`sd_text_encoders`) now render unconditionally inside Quicksettings row; removed `render=False` and explicit `.render()` calls to avoid duplicate/hidden components. Element ids updated to `sd_vae` and `sd_text_encoders`.
 - UX: Quicksettings VAE now espelha o seletor de Checkpoint (mesmo dropdown, refresh, eventos). Unificamos discovery (inclui `sd_vae.vae_dict`) e tornamos a resolução nome→caminho mais resiliente (aceita rótulos com sufixo `[hash]`).
- Refactor: Eliminated legacy submit path. Server no longer scans args to locate strict JSON; it must be the last input. Removed pre-submit fallback that copied incoming positional values when strict JSON is absent (`modules/ui.py`).
 - Refactor(UI/JS): Removed legacy JS exports `window.submit` e `window.submit_img2img`. `submit_txt2img_upscale()` agora usa `submit_named()`. Adicionada verificação de inicialização para slots JSON ocultos e handlers nomeados.
 - DX: Erros de submit sem JSON estrito agora trazem diagnóstico aprofundado: amostra dos últimos args (tipos/parse JSON), busca por JSON estrito fora de ordem (índice), elem_id esperado vs. real do último input, IP/UA do cliente e local do código. Mantemos fail‑fast sem reintroduzir fallback.
- Fix(UI Inject): Evitamos colisões de basenames ao injetar JS. `modules/ui_gradio_extensions.py` agora ignora assets sob `legacy/` e deduplica por basename, preferindo arquivos não‑legacy. Isso previne que `legacy/javascript/ui.js` sobrescreva o `javascript/ui.js` atual e evita perda do `submit_named` no navegador.
 - Fix(JS Submit): `submitWithProgress()` agora injeta o payload estrito como string JSON (fallback para objeto). Isso maximiza compatibilidade com `gr.JSON` como input oculto e previne valores `None` no último slot.

- Docs/Structure: Added `.sangoi/one-trainer/` with research and plan to align repo structure with Nerogar’s OneTrainer patterns (no Tkinter, keep Gradio). Includes mapping, multi‑path proposal, implementation plan, UI integration, and rollout.
- Docs/Architecture: Added `.sangoi/one-trainer/architecture/*` describing full inference rewrite (per‑model engines, orchestrator/registry, and new `Txt2Vid`/`Img2Vid` tabs) with migration and rollout plan.
- Research: Added ComfyUI/SwarmUI research, compatibility matrix, and UI dropdown plan for inference type.
- Core: Scaffolded `backend/core/` package (engine interface, request dataclasses, registry, orchestrator) plus unit tests to support the new modular pipeline.
- Process: Updated implementation/rollout plans to enforce manual-only validation and verbose error logging (sem pytest).
- Engines: Added `backend/engines/` scaffolding with SD15/SDXL/Flux stubs and registration helpers; documented manual validation steps in `.sangoi/one-trainer/architecture/manual-validation.md`.
 - Engine(sd15): Implemented functional SD15 engine over existing WebUI processing for `txt2img`, `img2img`, `inpaint`, and `upscale` via hires-fix; loads checkpoints through Forge loading parameters (`modules.sd_models.model_data.forge_loading_parameters`).
- Engine(sdxl): Implemented functional SDXL engine (`txt2img`, `img2img`, `inpaint`, `upscale`), same loading/bridging strategy as sd15.
- Engine(flux): Implemented functional Flux engine (`txt2img`); unsupported tasks raise `UnsupportedTaskError` with full context.
- UI: Replaced legacy radio preset with `Engine` dropdown (left of `Checkpoint`) populated from engine registry; persists as `codex_engine` and mirrors legacy preset for compat; updates default dims/CFG/sampler/scheduler per engine.
- Policy: Zero legacy compatibility — removed preset radio wiring and any fallback/adapters from the migration plan. `txt2img` handler now calls the new Orchestrator directly (no legacy processing path).
- Cutover(img2img): `img2img_from_json` now builds `Img2ImgRequest` and routes through `InferenceOrchestrator` using `codex_engine` + current checkpoint. No legacy processing path.
- Video: Added Txt2Vid and Img2Vid tabs (UI) wired to Orchestrator with strict JSON submit; added engine stubs `svd` (img2vid) and `hunyuan_video` (txt2vid/img2vid) to the registry.
- Video(Wan 2.2): Added WAN engines: `wan_ti2v_5b` (txt2vid/img2vid), `wan_t2v_14b` (txt2vid), `wan_i2v_14b` (img2vid) — stubs with explicit errors until integration.
- Docs: Added `.sangoi/one-trainer/param-matrix.md` mapping required/optional parameters per engine/task.
- UI: Added Mode dropdown (Normal/LCM/Turbo/Lightning) in Quicksettings; requests carry `metadata.mode`; engines may adapt or ignore with explicit logs.
- Core: Added per-engine parameter interfaces + parsers:
  - Code: `backend/core/params/*`, `backend/core/enums.py`, `backend/core/param_registry.py`.
  - Engines now validate/normalize requests via `parse_params(engine, task, request)`.
- Presets: Added loader/applier (`backend/core/presets.py`) and seed preset files under `configs/presets/`. Engines apply preset patch (fill-only) before validation and log `applied_preset_patch`.

- UI: Filter sampler/scheduler choices per engine using backend policy. Image sampler (ScriptSampler) and video tabs (Txt2Vid/Img2Vid) now update when `Engine` changes. Introduced JS components namespace (`codex.components.*`) for modular front-end.

- UI: Added component skeletons for Prompt/Hires/Canvas under `javascript/codex.components.*` and wired injection allowlist. These are helpers only (no behavior change) and will host future UI logic; strict JSON and server contracts remain the source of truth.

- Settings: Added `System ▸ Codex ▸ Export generated video to mp4/webm` option. When enabled, video engines receive `engine_options.export_video=true`; WAN TI2V‑5B may export video via ffmpeg. Env `CODEX_EXPORT_VIDEO=1` also works.

- DX: Add `run-webui.bat` (Windows launcher) — checks Python 3.10, creates/activates `.venv`, installs requirements, verifies core libs, warns on missing ffmpeg, and launches `webui.py`.

- Feat(Logging): centralized logging + SDXL debug instrumentation
  - New: `backend/logging_utils.py` configures logging once (default DEBUG). Level via `CODEX_LOG_LEVEL` (or `SDWEBUI_LOG_LEVEL`/`WEBUI_LOG_LEVEL`). Optional `CODEX_LOG_FILE` adds file handler.
  - SDXL: Detailed DEBUG logs in `backend/engines/sdxl/engine.py` (request summary, device/dtypes, lifecycle around `process_images`).
  - Builders: DEBUG summaries in `backend/engines/util/adapters.py` for txt2img/img2img (hr, denoise, sizes, samplers, seed).
  - Startup: `backend/__init__.py` initializes logging early for consistent verbosity.
  - Windows: `webui.settings.bat.example` shows how to set `CODEX_LOG_LEVEL`; `run-webui.bat` echoes the active level.

- Deps: Add `colorama==0.4.6` and `rich==13.9.2` to enable colored console/logging; logging setup already uses Rich/Colorama with tqdm-aware handler and safe fallback.
- Deps: Pin `torch==2.7.1` and `torchvision==0.22.0` for Windows stability (recreate venv recommended).

- Engine(wan_ti2v_5b): Loader attempts Diffusers `WanPipeline`/`AutoencoderKLWan` (local-only by default). Engine wired to call pipeline for txt2vid/img2vid, returning frames + `info` JSON, with prepare/run progress and VRAM metrics. If pipeline is unavailable or weights missing, raises explicit `EngineExecutionError` with upgrade/instructions.

- Engine(wan_ti2v_5b): Added sampler/scheduler mapping and per-engine limits. Allowed samplers: Euler a, Euler, DDIM, DPM++ 2M, DPM++ 2M SDE, PLMS. Scheduler flags: Karras → `use_karras_sigmas`, Simple → `timestep_spacing='trailing'`, Exponential → `use_exponential_sigmas`. Warnings emitted when unsupported.

- Engine(wan_ti2v_5b): Optional video export to mp4/webm via ffmpeg when `CODEX_EXPORT_VIDEO=1`. Paths under `artifacts/videos/`. Payload still returns frames; `info.video` carries metadata.

- Docs: Updated handoff `.sangoi/handoffs/2025-10-18-codex-inference-rewrite.md` with 6 solution paths for WAN 2.2 integration, selected Path A (Native PyTorch) as intended approach, and added MVP/validation/risks sections. Validation is UI-first; smoke script now optional. No user-facing changes yet.

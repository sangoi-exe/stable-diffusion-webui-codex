 # Handoff — 2025-10-18 — Codex modular inference, engines + params + presets

 Summary
 - Rewrote the inference architecture skeleton with a modular core (engines per model) and integrated it with the existing Gradio UI (no legacy paths). Added video tabs. Introduced typed param interfaces and an engine/mode/task preset system (fill-only), plus enums for Engine/Task/Mode.

 Scope Completed
 - Core
   - Enums: `backend/core/enums.py` (Mode, EngineKey, SamplerName, SchedulerName, Precision)
   - Param interfaces: `backend/core/params/*` (SD15, SDXL, Flux, Video variants)
   - Parser/registry: `backend/core/param_registry.py` — `parse_params(engine, task, request)`
   - Presets: `backend/core/presets.py` — YAML loader + fill-only patch with logs
   - Orchestrator in place; engines call presets->parse_params before running
 - Engines
   - Image: SD15/SDXL/Flux functional (bridge to existing processing) with validation + presets
   - Video stubs: SVD, HunyuanVideo, WAN 2.2 (TI2V‑5B, T2V‑14B, I2V‑14B) registered; strict errors not silenced
 - UI
   - Quicksettings Engine dropdown (left of Checkpoint)
   - Quicksettings Mode dropdown (Normal/LCM/Turbo/Lightning)
   - Tabs: Txt2Vid and Img2Vid with strict JSON submit; wired to Orchestrator
   - Strict JSON enforced for txt2img/img2img and new video tabs
 - Docs/Logs
   - `.sangoi/one-trainer/architecture/*` (overview, API, params interfaces, manual validation)
   - `.sangoi/one-trainer/param-matrix.md` (required/optional per engine/task)
   - Task logs for param interfaces, presets logic, engine stubs, UI dropdowns
   - CHANGELOG updated accordingly

 Files Touched (high-level)
 - Core: `backend/core/{engine_interface.py,requests.py,enums.py,options.py,param_registry.py,presets.py}`
 - Params: `backend/core/params/{base,sd15,sdxl,flux,video}.py`
 - Engines: `backend/engines/{sd15,sdxl,flux}/engine.py` + `backend/engines/video/{svd,hunyuan,wan/*}`
 - UI: `modules/ui.py`, `javascript/ui.js`, `modules_forge/main_entry.py`, `modules/{txt2img.py,img2img.py,txt2vid.py,img2vid.py}`
 - Presets: `configs/presets/{sd15,sdxl,flux,hunyuan_video,wan_ti2v_5b}/.../txt2img|txt2vid.yaml`
 - Docs: `.sangoi/one-trainer/architecture/*.md`, `param-matrix.md`
 - Logs: `.sangoi/task-logs/*`, `.sangoi/CHANGELOG.md`

 Validation Performed (manual-only per directive)
 - Verified new UI elements render and submit strict JSON (txt2img, img2img, txt2vid, img2vid)
 - Confirmed Orchestrator path used for txt2img/img2img; legacy paths removed
 - Confirmed `applied_preset_patch` logs when omitting fields present in preset (e.g., steps)
 - Engines raise explicit ValidationError or UnsupportedTaskError with engine/task/model context

Open Items / Next Steps
 1) Presets coverage
    - Add remaining presets: sd15/sdxl para img2img/inpaint/upscale; flux para LCM/Lightning; vídeo para Img2Vid (hunyuan/wan) e SVD
    - Validar presets com referência da ComfyUI (sampler/scheduler/steps/fps/num_frames)
 2) Mode-aware behavior
    - Implementar avisos por engine quando `mode` não for suportado pelo checkpoint (ignorar com log)
    - (Opcional) Ajuste interno de defaults quando presets não cobrirem, respeitando fill-only
 3) WAN 2.2 integração (prioridade)
    - Loader: transformer + VAE + text encoder (local `models/Wan/...` ou HF repo ids)
    - Forward: TI2V‑5B, T2V‑14B, I2V‑14B; mapear motion/guidance; frames → gallery e opcional mp4/webm
    - Usar ComfyUI como referência de parâmetros/schedulers; presets específicos do engine
 4) Hunyuan/SVD
    - Implementar carregamento/forward dos dois
 5) Métricas/diagnóstico
    - Garantir logs de `applied_preset_patch`; adicionar `mode_ignored_reason` quando aplicável
    - Cronometria/VRAM por etapa
6) Naming
    - Renomear Forge→Codex nos pontos tocados quando seguro
    - Adicionar exemplos de strict JSON por engine/tarefa (UI e API)

Solution Paths — WAN 2.2 Integration (enumerate first)

- Path A — Native PyTorch engine (chosen):
  - Implement loaders and forward pass in `backend/engines/video/wan/` using PyTorch 2.9 FP16 + channels_last.
  - Map parameters from our strict JSON to WAN’s expected inputs (fps/num_frames/width/height/cfg/seed/motion guidance).
  - Use ComfyUI as reference only for parameter semantics and safe defaults; keep our engine self-contained.
  - Pros: lowest integration risk, no cross‑process complexity, preserves our orchestrator, enables fine metrics.
  - Cons: requires careful VRAM management; upstream changes must be tracked manually.

- Path B — In‑proc ComfyUI nodes as library:
  - Import ComfyUI WAN nodes and call their Python modules directly in‑process.
  - Pros: fast to wire, inherits their scheduler defaults; easy to compare outputs.
  - Cons: fragile runtime coupling; version drift; implicit globals; harder to guarantee strict JSON contracts.

- Path C — Out‑of‑proc microservice (HTTP/IPC):
  - Run WAN in a separate process/server and call via HTTP/IPC.
  - Pros: isolation (VRAM), crash containment, can scale independently.
  - Cons: added ops surface, latency, stream/IPC complexity, more moving parts.

- Path D — Diffusers‑style pipeline port:
  - Wrap/port WAN to a Diffusers pipeline and use Diffusers schedulers.
  - Pros: standardized schedulers, community familiarity, easier future epi‑integration.
  - Cons: higher upfront cost; risk of parity gaps; large review surface.

- Path E — ONNX/TensorRT backend:
  - Export model blocks to ONNX and run with ORT/TensorRT for perf.
  - Pros: potential speedups.
  - Cons: heavy toolchain, export friction, limited flexibility early in bring‑up.

- Path F — Torch compile/AOT acceleration:
  - Keep native engine but add guarded `torch.compile`/AOT for hot paths.
  - Pros: incremental perf without architectural change.
  - Cons: kernel dynamism can limit wins; debug complexity.

Chosen approach: Path A now; design engine to allow switching to C or F later per backend flag.

Intended Solution (before implementation)

- Engine layout: `backend/engines/video/wan/{__init__.py, engine.py, loader.py, params.py, schedulers.py}`.
- Loader: resolve local `models/Wan/...` or env `WAN_REPO_ID`; build components (text encoder, transformer, VAE) with FP16 + `channels_last` and autocast guards; explicit logs for precision/memory.
- Forward (TI2V‑5B first): accept `Txt2VidRequest`/`Img2VidRequest`, apply preset fill‑only, validar via `parse_params`, executar via Diffusers `WanPipeline` (local files only por padrão), retornar frames como `PIL.Image` list e `info` JSON; opcionalmente, montagem mp4/webm (guardada por `ffmpeg`).
- Parameters: fps, num_frames, width, height, cfg, seed, motion_guidance (or analogous), scheduler/sampler names; unsupported fields raise ValidationError with engine/task context.
- Presets: seed `configs/presets/wan_ti2v_5b/txt2vid.yaml` and `img2vid.yaml` with conservative VRAM defaults; log `applied_preset_patch` keys.
- Metrics: time each stage (tokenize/denoise/decode); VRAM peak via `torch.cuda.max_memory_allocated()` when GPU visible; include `mode_ignored_reason` if Mode not supported.
- UI: no new widgets; use existing Video tabs; ensure strict JSON examples in docs.

Sampler/Scheduler Mapping
- Allowed (WAN TI2V‑5B): `Euler a`, `Euler`, `DDIM`, `DPM++ 2M`, `DPM++ 2M SDE`, `PLMS`.
- Mapping → Diffusers scheduler classes with optional flags:
  - `Euler a` → `EulerAncestralDiscreteScheduler`
  - `Euler` → `EulerDiscreteScheduler`
  - `DDIM` → `DDIMScheduler`
  - `DPM++ 2M` → `DPMSolverMultistepScheduler(algorithm_type='dpmsolver++', solver_order=2)`
  - `DPM++ 2M SDE` → `DPMSolverMultistepScheduler(algorithm_type='sde-dpmsolver++', solver_order=2)`
  - `PLMS` → `PNDMScheduler(skip_prk_steps=True)`
- Scheduler (UI) → flags:
  - `Karras` → `use_karras_sigmas=True` (quando suportado pela classe)
  - `Simple` → `timestep_spacing='trailing'`
  - `Exponential` → `use_exponential_sigmas=True`
  - `Automatic` → sem alterações
- Comportamento: Avisos explícitos no log quando uma opção não for suportada; resposta inclui `sampler_in`, `scheduler_in`, `sampler_effective`, `scheduler_effective` no `info` JSON.

Front Modularization (C+D)
- JS components namespace `window.Codex.Components` com módulos: core (bus), sampler, prompt, hires, canvas.
- ui.js permanece orquestrador; componentes expõem helpers read/write e keybinds (Ctrl+Enter gerar, Alt+Enter skip, Esc interrupt) idempotentes.
- Política de sampler/scheduler vem do back (`backend/core/sampler_policy.py`); UI só reflete e aplica.

Vídeo opcional (export)
- Exportação opcional para mp4/webm via `ffmpeg` quando `CODEX_EXPORT_VIDEO=1` e ffmpeg disponível. Diretório: `artifacts/videos/<run>/`.
- `info` JSON inclui `video` com meta (caminhos, fps). Frames continuam retornados normalmente.

MVP Deliverables (TI2V‑5B)

- Loader + forward usando Diffusers `WanPipeline` (local-only) com presets e progress events.
- Registry entries wired; erros explícitos quando `WanPipeline` indisponível ou pesos ausentes.
- Presets + validação; métricas (tempo/VRAM) em `info` JSON.
- UI‑first validation; optional smoke script can be added later for diagnostics.

Validation Plan

- UI-first: validar via UI (Txt2Vid/Img2Vid) com JSON estrito; logs imprimem contagem/dimensões de frames e métricas sem exigir script externo.
- Comparar step/fps/cfg com ComfyUI como referência e documentar desvios intencionais.
- Capturar métricas (tempo/VRAM) em resoluções e contagens de frames típicas. O script de smoke é opcional (apenas para diagnóstico), não requisito.

Risks & Fallbacks

- High VRAM pressure: add `--tiling`/`--offload` flags in loader (no-op for MVP, logged as unsupported if not implemented).
- Scheduler mapping mismatch: pin to a single, validated scheduler for MVP; expand later.
- Codec availability: if `ffmpeg` missing, export frames only; log actionable hint.

 Risks/Notes
 - WAN 2.2 e vídeo demandam VRAM/precisão específicos; considerar isolamento por processo no futuro
 - Strict JSON permanece obrigatório; sem preenchimento “mágico” para inputs inválidos

 Commands/Where to look
 - Engine registration: `backend/engines/__init__.py`, `backend/engines/registration.py`
 - Param interfaces: `backend/core/params/*`, parsers em `backend/core/param_registry.py`
 - Presets loader: `backend/core/presets.py`, YAML sob `configs/presets/`
 - UI engine/mode: `modules_forge/main_entry.py`
 - Video tabs: `modules/ui.py`, `javascript/ui.js`

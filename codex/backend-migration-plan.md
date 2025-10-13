Backend Migration Plan (Modules → backend/*)
===========================================

Goal
- Consolidate "backend-ish" logic scattered across `modules/` into the `backend/` package with explicit boundaries and services. Preserve API/UI behaviour while reducing global coupling.

Principles
- No functional regressions; identical inputs/outputs.
- Fail fast on missing preconditions (no silent fallbacks).
- Small, atomic steps; keep `modules/` as thin adapters until full migration.
 - No rigid parity: outputs do not need to be bit‑exact; ensure features continue working (preferivelmente melhor) and public API contracts remain stable unless versioned.

Targets and Moves
1) Generation flow (txt2img/img2img)
- Today: duplication across `modules/txt2img.py`, `modules/img2img.py`, and API glue in `modules/api/api.py`.
- Target: introduce `backend/services/image_service.py::process_request(mode, args)`
- Responsibilities: build `p` (processing context), run `process_images`, manage queue/lock, and package results.
- `modules/api/api.py` becomes a thin adapter that: (a) valida/sanitiza inputs, (b) delega ao ImageService, (c) retorna o mesmo schema de antes.

2) Media encode/decode
- Today: base64/URL handling sprinkled in API and UI layers.
- Target: consolidate in `backend/services/media_service.py` (já existe). Ensure all API endpoints call this service.

3) Sampler/scheduler resolution
- Today: aliases e validação espalhados.
- Target: `backend/services/sampler_service.py` (já existe) centraliza mapeamento e erros; API/UI não fazem lookups diretos.

4) Options/config
- Hoje: leitura direta de `modules.shared`.
- Target: `backend/services/options_service.py` (já existe) expõe getters/setters e flags correntes de cmd/opts; adaptação gradual de chamadas em API/UI.

5) Progress/telemetria
- Introduzir `backend/services/progress_service.py` para EMS (ETA, steps/s, job id) e expor `/internal/memory` via `webui.py` FastAPI (VRAM/RAM/flags de `backend/memory_management.py`).

6) Model loading & tokenizers
- Já consolidado em `backend/loader.py`. `modules/sd_models.py` passa a chamar o loader/guess do `backend/` (adapter fino). Documentado strict offline.

Adapters (anticorruption layer)
- Enquanto migramos, manter funções em `modules/` como wrappers que chamam serviços do `backend/`. Documentar em docstrings que o path definitivo é o serviço e que o módulo é legado.

Directory Map (final)
- `backend/`
  - `services/`: image, media, options, sampler, progress
  - `diffusion_engine/`, `nn/`, `operations*`, `patcher/`, `sampling/`
  - `loader.py`, `memory_management.py`, `stream.py`, `utils.py`
- `modules/` (thin)
  - `api/*` calls `backend/services/*`
  - `sd_models.py` calls `backend/loader.py`
  - `processing/*` uses `backend/sampling/*` via service

Validation
- Snapshot tests: same outputs for canonical prompts (small seeds, low steps) pré/pós-migração.
- Perf counters: tempo por etapa (loader, sampling); pico de VRAM.

Milestones
- M1: unificar txt2img/img2img via ImageService (sem mudança de API).
- M2: endpoints usam MediaService/SamplerService/OptionsService.
- M3: `/internal/memory` e ProgressService.
- M4: adapters concluídos; módulos legados apenas reexportam tipos/constants necessários.

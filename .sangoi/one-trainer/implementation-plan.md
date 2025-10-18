# Implementation Plan — OneTrainer‑Like Structure (Incremental)

Phase 0 — Docs and scaffolding (this change)
- Add `.sangoi/one-trainer/*` docs; align on plan.
- Add enums/dataclasses para Engine/Task/Mode e interfaces de params por engine/tarefa; registry de parsers.

Phase 1 — Presets and resources
- Create `configs/presets/` with `schema_version: 1`.
- Add sample presets:
  - `txt2img/basic.yaml`, `img2img/restore.yaml`, `inpaint/simple.yaml`, `upscale/real-esrgan.yaml`.
- Add loader: `modules/ui_loadsave.py` already handles some presets; extend to read YAML from `configs/presets/` behind a feature flag.
- Validation: roteiro manual (carregar/aplicar YAML via UI e CLI); registrar logs completos (schema, preset, diffs). Sem automatizar testes.

Phase 2 — Registry wrappers (no renames)
- Files:
  - `backend/pipeline_registry.py` — maps keys → existing pipeline entrypoints.
  - `backend/sampler_registry.py` — maps keys → existing sampler factories.
  - `backend/model_registry.py` — maps keys → existing model loading flows.
- Behavior: thin functions with verbose logging; raise explicit `KeyError` for unknown keys. Engines devem validar requests via `parse_params(engine, task, request)`.
- Validação manual: importar registries pelo console e executar chamadas básicas; capturar logs que provem chave→função e erros para chaves inválidas (sem suites automatizadas).

Phase 3 — Scripts suite (CLI)
- Add `scripts/generate.py` using registries and reading a preset file.
- Add `scripts/export_debug.py` to zip env info (`pip freeze`, `torch` versions), recent logs, and active config.
- Add `scripts/benchmark.py` (optional, seed‑fixed) to measure sampler throughput.

Phase 4 — Launch helpers
- Add `tools/start-ui` (sh/bat) that delegates to `webui.py` with existing flags.
- Add `tools/run-cmd` to proxy `python -m` calls for scripts.
- Add `tools/export-debug` that calls `scripts/export_debug.py`.

Phase 5 — UI integration (Gradio)
- Add preset browser in existing settings panel (non‑modal) and an apply button.
- Persist last applied preset in `config_states/`.
- Promote schema errors/warnings to a Gradio notification with actionable hints.

Phase 6 — Optional follow‑ups
- Split requirements by backend (CUDA/ROCm/CPU) à la OneTrainer.
- RFC for `src/` packaging; defer unless needed.

Acceptance (manual) por fase
- P1: Demonstrar manualmente presets aplicados (UI/CLI) com logs anexados mostrando schema/version/config.
- P2: Console/manual mostrando registries resolvendo chaves válidas e erros verbosos para chaves inválidas.
- P3: Executar scripts manualmente (`python scripts/generate.py ...`) e anexar outputs/logs.
- P4: Rodar `tools/export-debug` e anexar artefato/manual.
- P5: Gravar walkthrough (texto/capturas) mostrando aplicação de preset no UI, com mensagens de warnings explícitas.

Rollback
- Feature flags guard new code paths; fall back to current behavior by disabling flags.
- Erros nunca devem ser suprimidos: sempre logar task, engine, modelo, preset, opções e stack trace para facilitar diagnóstico manual.

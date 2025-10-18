# Proposal — Adopt OneTrainer-Like Structure (Without Trainer, Keep Gradio)

Guiding constraints
- Do not introduce a trainer.
- Keep existing names/functions/variables where possible.
- Keep Gradio UI; do not add Tkinter.
- Prefer incremental, low-risk structural changes with strong logs and tests.

Goals
- Mirror OneTrainer’s clarity of structure: separation de core vs scripts; presets como arquivos; registries; launch helpers.
- Reescrever pipeline de inferência com engines por família de modelo (SD15, SDXL, FLUX, etc.), mantendo layout do Gradio.
- Adicionar novas tarefas: `txt2vid` e `img2vid` com abas dedicadas.

Non-Goals
- Wide renames or breaking public APIs.
- Removing existing features to simplify the refactor.

Solution paths (≥5)
1) Documentation‑only mapping
   - Write structure docs; keep code as‑is. Lowest risk, least benefit.

2) Presets + resources first
   - Add `configs/presets/` (YAML/JSON) and `resources/` to host templates/icons.
   - Wire Gradio to load/apply presets; no core changes.

3) Registry wrapper layer
   - Introduce small factories/registries (no renames) like `backend/pipeline_registry.py`, `backend/model_registry.py`, `backend/sampler_registry.py` that map string keys to existing functions/classes.
   - Scripts and UI import registries; core logic remains unchanged.

4) Scripts suite (CLI)
   - Add `scripts/` Python entrypoints mirroring OneTrainer’s pattern for common tasks (e.g., `generate.py`, `benchmark.py`, `export_debug.py`, `cache_models.py`, `serve.py`).
   - Provide thin shell wrappers (`tools/start-ui`, `tools/run-cmd`) that proxy to Python.

5) Launch helpers + debug report
   - Add `tools/install.*`, `tools/update.*`, `tools/export-debug.*` to standardize environment and bug reports.

6) Full repo reshape (src/ packaging)
   - Move code under `src/`, re‑package modules, split requirements by backend (CUDA/ROCm/CPU). Max benefit but highest risk; defer.

7) Microservices por engine (processo externo, gRPC/HTTP)
   - Isolação forte e escalabilidade; maior complexidade operacional.

Chosen approach (balanced: 2 + 3 + 4 + 5 + parte de 7)
- Start with presets/resources (2), then registries (3), then scripts (4), and launch helpers (5). Avoid (6) for now; keep (1) as living docs.
 - Reescrever pipeline usando `EngineInterface` + `Orchestrator` (arquivos em `backend/core/`); engines pluggable por modelo; considerar isolamento por processo (7) somente para engines de vídeo.

Why
- Presets and registries deliver value without breaking call sites. Scripts/launch helpers increase repeatability and parity with OneTrainer’s operator experience. Gradio remains intact and benefits from presets.

Acceptance criteria
- `configs/presets/` exists with ≥3 sample presets; loading from UI and from CLI.
- Simple registries present (`create_pipeline`, `create_sampler`, `load_model`) that call existing implementations; verbose logs show chosen keys.
- `scripts/export_debug.py` outputs a zip with env, config, and recent logs.
- `tools/start-ui` delegates to `webui.py` and preserves existing flags.

Risks and mitigations
- Accidental renames: enforce wrapper approach; no changes to existing function signatures.
- Drift between UI and CLI: both consume the same registry.
- Preset schema creep: version presets with `schema_version` and validate.

Rollout (phased)
1) Add folders (`configs/presets/`, `resources/`), sample presets, and docs.
2) Introduce registries (pure wrappers) + unit tests.
3) Wire UI to apply presets; add CLI `scripts/generate.py` using registries.
4) Add export‑debug tooling and shell wrappers in `tools/`.
5) Optional: split requirements by backend (follow‑up RFC).

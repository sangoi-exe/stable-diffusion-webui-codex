# Mapping — OneTrainer Patterns → This Repository

Legend: OT = OneTrainer; WR = this repo.

- OT `modules/` (core training logic)
  → WR `backend/`, `modules/`, `modules_forge/` (core inference, loaders, samplers).

- OT `scripts/` (CLI entrypoints)
  → WR `scripts/` (exists) + add `scripts/generate.py`, `scripts/export_debug.py`, `scripts/benchmark.py`, `scripts/cache_models.py`.

- OT `training_presets/`
  → WR `configs/presets/` (new) for generation presets (txt2img, img2img, inpaint, upscale). Versioned YAML.

- OT `resources/`
  → WR `resources/` (new) for templates/icons/defaults used by UI and scripts.

- OT `embedding_templates/`
  → WR `resources/prompt_templates/` (new), optional, to keep prompts reusable.

- OT `docs/`
  → WR `codex/` (already exists) + new docs in `.sangoi/one-trainer/` for engineering plan.

- OT `modules/util/create.py` (factory/registry)
  → WR `backend/pipeline_registry.py`, `backend/model_registry.py`, `backend/sampler_registry.py` (thin wrappers mapping string keys to existing call sites).

- OT UI (Tkinter)
  → WR UI (Gradio) — no change; UI calls registries and applies presets.

- OT launch helpers (`start-ui.*`, `update.*`, `run-cmd.sh`)
  → WR `tools/start-ui`, `tools/run-cmd`, `tools/export-debug` sh/bat wrappers.

Concrete key paths in WR
- Pipelines entry: `modules/txt2img.py`, `modules/img2img.py` → expose via `backend/pipeline_registry.py:create_pipeline(key, **kw)` with keys `txt2img`, `img2img`, `inpaint`, `upscale`.
- Samplers: `modules/sd_samplers*.py`, `backend/services/sampler_service.py` → expose via `backend/sampler_registry.py:create_sampler(key, **kw)`.
- Model loaders: `modules/modelloader.py`, `modules/sd_models.py`, `backend/loader.py` → expose via `backend/model_registry.py:load_model(key, **kw)`.

Presets
- Location: `configs/presets/{area}/{name}.yaml` where `area ∈ {txt2img, img2img, inpaint, upscale}`.
- Schema: versioned; include sampler, scheduler, steps, CFG, size, VAE, LoRA refs, seed strategy, and UI‑specific fields kept in a `ui:` subnode.


# Research — OneTrainer (Nerogar)

This note summarizes the public structure of Nerogar’s OneTrainer project and highlights reusable patterns for structure and pipeline organization.

Key findings (Oct 18, 2025):
- Repository entry: `README.md` shows a top-level layout with `modules/`, `scripts/`, `docs/`, `resources/`, `training_presets/`, `embedding_templates/` and a set of launch/update helpers. It supports both UI (Tk) and CLI modes. [Ref]
- The UI is Tkinter-based. A request to move to Gradio was explicitly declined (“Not planned”), confirming UI separation from core modules. This strengthens our plan to keep UI-agnostic core and use Gradio here. [Ref]
- There is a structured “project structure” doc under `docs/ProjectStructure.md` describing how the core is modularized (trainers, models, datasets, configuration), and emphasising factory/registry creation helpers (e.g., `modules/util/create.py`). [Ref]
- Launch scripts consolidate environment setup and proxy the actual `python` module entrypoints (`scripts/train.py`, etc.). [Ref]

Directory overview (from public tree and docs):
- `modules/` — core logic (trainers, model loaders, utilities, registry/factory helpers).
- `scripts/` — CLI entrypoints (e.g., `train.py` plus UI/bootstrap scripts: `start-ui.*`, `install.*`, `update.*`, `run-cmd.sh`).
- `training_presets/` — curated YAML/JSON presets for common training configurations.
- `resources/` — UI/packaging resources (icons, defaults).
- `embedding_templates/` — prompt templates used during training.
- `docs/` — conceptual documentation (e.g., project structure, usage, launch scripts).

Reusable patterns:
- Clear split between “core modules” and “runnable scripts”.
- Registry/factory helpers to construct components by name (model, dataset, trainer).
- Presets as first-class, versionable files instead of spreading defaults in code.
- Launch scripts standardize environment ops (install, update, debug report) and call into Python entrypoints.
- UI kept thin; core remains headless and importable.

Implications for this repo:
- We can mirror the structural ideas without switching UI toolkit. Gradio will call into the same core APIs used by scripts.
- Presets (generation/config) provide a safer way to evolve defaults and share configurations.
- A small registry layer can stabilize cross-module creation without renaming existing functions.

References (public pages viewed on Oct 18, 2025):
- OneTrainer GitHub repo entry and README. 
- Issue declining Gradio UI request (“Not planned”).
- Docs page: `docs/ProjectStructure.md` (structure overview).

Notes:
- We will not import or embed OneTrainer code. Only patterns are adopted.


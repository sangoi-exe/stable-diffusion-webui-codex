 # Task: Research OneTrainer structure and draft alignment plan for this repo (no Tkinter)

 Date: 2025-10-18

 Summary
 - Researched Nerogar’s OneTrainer repo structure and UI choices; mapped reusable patterns.
 - Drafted a multi-path proposal and selected a balanced, low-risk plan.
 - Authored docs in `.sangoi/one-trainer/` covering research, mapping, plan, UI integration, and rollout.

 Files Touched
 - `.sangoi/one-trainer/README.md`
 - `.sangoi/one-trainer/research-onetrainer.md`
 - `.sangoi/one-trainer/proposal.md`
 - `.sangoi/one-trainer/mapping.md`
 - `.sangoi/one-trainer/implementation-plan.md`
 - `.sangoi/one-trainer/ui-integration.md`
 - `.sangoi/one-trainer/rollout.md`
 - `.sangoi/CHANGELOG.md` (append summary)

 Key Findings
 - OneTrainer separates `modules/` (core), `scripts/` (entrypoints), `training_presets/` (configs), `resources/`, and `docs/`.
 - UI is Tkinter-based (Gradio explicitly declined), which supports keeping our Gradio UI and adopting only structure/patterns.
 - Factory/registry helpers exist to create components by key name and keep the core UI-agnostic.

 Plan Snapshot
 - Add `configs/presets/` + sample presets (YAML) and a loader.
 - Add registries (`backend/*_registry.py`) that wrap current implementations without renames.
 - Add scripts: `scripts/generate.py`, `scripts/export_debug.py`, plus `tools/*` shell wrappers.
 - Integrate presets into Gradio (picker + apply).

 Validation
 - No behavior changes yet. Next phases will include unit tests and smoke runs for presets and registries.

 Risks / Notes
 - Avoid accidental renames; wrappers only. Schema validation must not fail silently.


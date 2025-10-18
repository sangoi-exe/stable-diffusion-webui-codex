# OneTrainer → WebUI (Structure Alignment)

Purpose: capture OneTrainer’s project patterns and define a safe, incremental plan to mirror its structure here without turning this repo into a trainer or renaming existing public code paths. Gradio stays as the UI; no Tkinter.

Scope:
- Document OneTrainer’s layout, pipelines, and scripts at a high level.
- Propose ≥5 viable options to adopt the same organization style here.
- Select a robust, low‑risk path with clear steps and acceptance criteria.
- Keep current names, functions, and variables wherever possible.

Deliverables in this folder:
- `research-onetrainer.md` — repository structure and pipeline notes (with references).
- `proposal.md` — solution options, trade‑offs, and the chosen approach.
- `mapping.md` — matrix mapping (OneTrainer → this repo modules/paths).
- `implementation-plan.md` — phased plan, milestones, validations, and rollback.
- `ui-integration.md` — Gradio integration for presets/configs.
- `rollout.md` — PR sequencing and hygiene.

Non‑Goals:
- Implementing a trainer.
- Migrating UI to Tkinter.


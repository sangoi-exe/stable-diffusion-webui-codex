# Task: Replace radio preset with Engine dropdown (left of Checkpoint)
Date: 2025-10-18

Summary
- Added `Engine` dropdown in Quicksettings (left of `Checkpoint`) to select model family (`sd15`, `sdxl`, `flux`, dynamic via registry if available).
- Removed legacy radio `UI` preset; dropdown change atualiza defaults (dims/CFG/sampler/scheduler) e persiste seleção em `opts.codex_engine` (sem espelhar para chaves legacy).

Files/Paths
- `modules_forge/main_entry.py`: created `ui_codex_engine` (gr.Dropdown), removed `ui_forge_preset` radio logic, hooked up change handler.
- `modules/shared_options.py`: added hidden option `codex_engine`.
- `.sangoi/one-trainer/ui-dropdown-plan.md`: updated placement/details.

Manual Validation
- Launch UI; in Quicksettings, confirm `Engine` dropdown appears left of `Checkpoint`.
- Switch between `sd15`, `sdxl`, `flux` and observe width/height/CFG/sampler/scheduler updates per engine; logs persist selection.

Notes
- Naming migration (Forge→Codex) begins here with `codex_engine` option; remaining identifiers keep legacy names for compatibility.

# Rollout — PR Sequencing and Hygiene

PR 1 — Presets scaffolding
- Add `configs/presets/` and sample YAMLs.
- Extend `ui_loadsave.py` guarded by `--enable-presets` flag.
- Docs: update README usage and `.sangoi/one-trainer`.

PR 2 — Registries
- Add `backend/*_registry.py` wrappers; unit tests.
- Wire scripts to registries.

PR 3 — Scripts + tools
- Add `scripts/generate.py`, `scripts/export_debug.py`, `tools/start-ui`, `tools/run-cmd`, `tools/export-debug`.

PR 4 — UI integration
- Preset picker + apply in Gradio; logs and notifications.

PR 5 — Optional requirements split (RFC)
- Propose `requirements/{global,cuda,rocm}.txt` mirroring OneTrainer.

Git hygiene
- Small, cohesive commits with Conventional Commits.
- Verify `rg -n "<<<<<<<|=======|>>>>>>>"` returns empty.
- Validate only intended files are staged.


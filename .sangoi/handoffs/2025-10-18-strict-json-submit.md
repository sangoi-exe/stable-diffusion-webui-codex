Handoff: 2025-10-18

- Summary: Forced strict JSON payload injection via `submitWithProgress` builder hook; exposed submitters for Gradio `_js`, typed the flow (StrictBuilder now `IArguments`), and refactored quicksettings so VAE uses a checkpoint-style dropdown while text encoders stay multiselect.
- Files: `javascript/ui.js`, `modules_forge/main_entry.py`, `modules/ui_settings.py`, `modules/ui_extra_networks_checkpoints_user_metadata.py`, `modules_forge/shared_options.py`
- Commands: `npm run typecheck` (blocked in sandbox)
- Validation: Manual audit only; host run reported success for `javascript/ui.js`
- Next steps: 1) Re-run `npm run typecheck` on host (remaining Forge JS warnings persist outside this patch). 2) Headless/UI click test to confirm strict payload reaches backend (watch for cached JS). 3) Exercise the new VAE dropdown/text encoder multiselect in the UI (including hires overrides) to confirm module reloading behaves as expected.

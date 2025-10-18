# Gradio Integration — Presets and Registries

Principles
- Keep UI thin: Gradio calls registries and preset loader.
- No renames of existing UI event handlers; add adapters where needed.

Integration points
- `modules/ui_settings.py` — add preset file picker and an “Apply Preset” button.
- `modules/ui_loadsave.py` — extend to support YAML presets in `configs/presets/`.
- `modules/ui.py` / `modules/txt2img.py` / `modules/img2img.py` — adapt submit handlers to accept a resolved `Preset` object (optional arg) that overrides controls.

DX details
- Show a toast with `schema_version` and any non‑applied fields (with reasons).
- Log full, sanitized preset content to `.webui-api.log` at `INFO` when applied.
- Provide a “Copy as JSON” action for quick sharing (keeps parity with current state saving).

Validation
- Unit tests for preset parsing and field mapping.
- Headless smoke test (`scripts/generate.py`) to apply a preset and render.


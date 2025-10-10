# Frontend & UI Layer

The user interface builds on Gradio 4 with extensive customizations. Understanding the composition of UI assets helps coordinate refactors between Python callbacks and client interactivity.

## Gradio Layout
- `modules/ui.py`, `modules/ui_components.py`, and `modules/ui_tempdir.py` define tabs, accordions, and shared controls.
- `modules_forge/ui/` introduces Forge-specific widgets (Flux sliders, GPU weight controls, canvas enhancements) and overrides for Gradio behaviours.
- API endpoints in `modules/api` expose REST routes that mirror UI actions for automation.

## Static Assets
- `style.css` and `html/` provide base styling and HTML fragments injected into the Gradio app.
- `javascript/` contains modular JS enhancements (canvas tools, keyboard shortcuts, telemetry toggles). Forge scripts extend or replace assets from the upstream project.
- `script.js` is the legacy entry point that Gradio loads; it imports bundles from `javascript/` during runtime.

## Canvas & Interactive Tools
- Integrated editors (Photopea, OpenPose, etc.) live under `html/` and `modules/ui_components.py`.
- Forge-specific canvas logic is primarily implemented in `javascript/canvas/` and `modules_forge/ui/canvas.py`.
- Tablet support (e.g., Wacom pressure levels) relies on event handlers within `javascript/events/` combined with configuration in `modules_forge/ui/config.py`.

## Localization & Accessibility
- `localizations/` stores translation JSONs. When adding UI elements, ensure translation keys are defined and fallback strings exist.
- Use semantic HTML in custom templates and avoid hard-coding language strings inside JavaScript; prefer translation helpers in `modules/ui_common.py`.

## Refactor Considerations
- Maintain compatibility with Gradio updates. Test both dark/light themes and ensure Forge overrides degrade gracefully when upstream markup changes.
- Coordinate Python callback signatures with JavaScript event payloads. Schema changes should be reflected in `codex/testing-and-tooling.md` to prompt regression tests.
- Avoid large monolithic JavaScript files. Prefer feature-scoped modules to keep tree shaking effective and reduce merge conflicts.

## Flux Preset Surface
- Flux toggles live in two places: layout defaults in `modules_forge/main_entry.py` (width/height, CFG sliders, GPU memory budget) and sampler/scheduler defaults in `modules/processing_scripts/sampler.py`.
- Both files now rely on `backend/diffusion_engine/flux_config.py` for the authoritative list of fields, ensuring UI labels, defaults, and slider ranges stay synchronised with backend expectations.
- When introducing new Flux controls, extend the schema first, then wire them through `build_flux_option_info` so future audits only need to inspect a single table of metadata.

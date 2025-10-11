Session Log
===========

2025-10-11 – Gradio migration start
- Align expected Gradio to 5.49.1 in `modules/errors.py`.
- Switch global queue to `concurrency_limit=64` in `webui.py` per Gradio 5 (replaces positional arg).
- Cross-checked guide “Guia de Migração do Gradio 4.40 para 5.md”; no `every=` usage found; no listener-level limits added yet (defaults inherit from Blocks.queue).
 - Replaced legacy JS tab switching for img2img copy buttons with Python-only `gr.Tabs.update` chain; capture inner `mode_img2img` Tabs for programmatic selection.
 - Added optional SSR gating via `GRADIO_SSR_MODE` env; defaults off.

2025-10-11 – Frontend cleanup
- Removed `javascript/compat_gradio5.js` which wrapped `onUiUpdate/onUiLoaded` and `executeCallbacks` to suppress errors.
- Rationale: avoid silent failures; surface exceptions for proper fixes per repo guidance (no workarounds/hacks).
- Expected impact: console may show real callback errors that were previously swallowed; no script tag updates needed (scripts autoload from `javascript/`).
- Follow-ups: fix any callback code that throws instead of masking issues at the wrapper layer.

- Hardened callback plumbing in `script.js`:
  - Validate registrations for `onUiUpdate/onUiLoaded/onAfterUiUpdate/onUiTabChange/onOptions*`; ignore non-functions with a clear warning.
  - `executeCallbacks` now tags logs with context and skips non-function entries.
  - Updated callers to pass context strings (e.g., `onOptionsChanged`).

2025-10-11 – Runtime + Tooling
- Backend: advanced txt2img runtime to include RNG initialization, script batch hooks, extra network activation, post‑sample callbacks, and progress updates.
- LoRA: added diagnostics and safer activation path; continuing to harden registry initialization order across entrypoints.
- Logging: introduced Rich/Colorama pipeline with tqdm‑aware handler (fallback to std logging when unavailable).
- Deps: added updater `tools/update_requirements.py`; refreshed pins, excluded torch family for local builds; resolved open‑clip/protobuf pin conflict.
- Validation: agreed to prioritize UI manual tests; capture CLI remains available for deterministic artefacts.

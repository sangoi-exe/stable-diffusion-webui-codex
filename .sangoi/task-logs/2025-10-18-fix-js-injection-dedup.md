Task: Prevent legacy UI js from overriding strict UI js
Date: 2025-10-18

Change
- modules/ui_gradio_extensions.py
  - Exclude any `legacy/` javascript assets from injection.
  - Deduplicate by basename when injecting `.js`/`.mjs` (prefer non-legacy paths).

Rationale
- Both `javascript/ui.js` and `legacy/javascript/ui.js` share the same basename. Depending on order, the legacy file could override window exports and break strict submit builders, leading to an empty hidden JSON slot on submit.

Validation
- Grep shows only one `ui.js` tag emitted in `<head>`.
- Strict submit self-check passes in Console.

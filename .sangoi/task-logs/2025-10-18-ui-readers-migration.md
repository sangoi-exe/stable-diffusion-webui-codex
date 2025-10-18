Task: Migrate video builders to component Readers (phase 1)
Date: 2025-10-18

Changes
- Added `javascript/codex.components.readers.js` with DOM read helpers (text/int/float/dropdown/radio/seed).
- Updated `javascript/ui.js` video builders (`buildNamedTxt2Vid`/`buildNamedImg2Vid`) to call Readers if present, falling back to legacy functions.
- Allowlist updated in `modules/ui_gradio_extensions.py` to inject Readers early.

Notes
- Non-breaking: ui.js retains legacy helpers; Readers are used opportunisticamente.
- Next: progressively adopt Readers in txt2img/img2img builders and other utilities, then slim down ui.js.


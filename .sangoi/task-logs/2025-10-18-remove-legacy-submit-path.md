Task: Remove legacy submit path (positional/scan) and enforce strict JSON last-slot only
Date: 2025-10-18

Context
- Clicking Generate raised ValueError: missing `__strict_version` and hints to use `_js="submit_named"` and last-slot JSON.
- Goal: “Explode” legacy path and remove tolerance in server submit.

Changes
- modules/ui.py
  - `_extract_strict_payload`: now requires the last argument to be the strict JSON; removed scanning across args. Keeps JSON-string parse attempt for `gr.JSON` edge cases.
  - `_normalize_args`: removed non-strict fallback (`out = list(values)`), avoiding any positional carry-over when strict JSON is missing.

Validations
- Grep: no remaining server-side checks accept positional submit for txt2img/img2img.
- UI bindings: `toprow.*` use `_js="submit_named"` and `_js="submit_img2img_named"`.
- JS path: `submitWithProgress()` already places the strict payload in the last slot.

Risks
- Any extension adding inputs after the hidden JSON will trigger explicit server error with diagnostics (by design).
- Clients caching old JS that call `submit()` without the named handlers will fail fast; clearing cache or updating assets required.

Next
- Optionally add a UI startup self-check and console warning if `window.submit_named` is missing.

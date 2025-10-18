Task: Drop JS legacy submit exports and add deep server diagnostics for strict submit
Date: 2025-10-18

Changes
- javascript/ui.js
  - Removed `submit()` and `submit_img2img()`; kept named variants only.
  - `submit_txt2img_upscale()` now delegates to `submit_named()`.
  - Added onUiLoaded self-check logging hidden JSON presence and named handler exports.
- modules/ui.py
  - Added deep diagnostics when strict payload is missing/misplaced (both txt2img and img2img):
    - searches for strict JSON elsewhere (index + keys sample),
    - reports args tail signature (type, JSON parseability),
    - expected vs. actual last `elem_id`,
    - client host and user-agent, code location, timestamp.

Validation
- Grep confirms `_js` hooks use only `submit_named` / `submit_img2img_named`.
- JS exports no longer expose legacy names on `window`.

2025-10-18 (appendix)
- Added strict-only legacy aliases in JS so any lingering `_js="submit"` or `_js="submit_img2img"` calls route to named strict submitters. This preserves strict JSON requirement while making older hooks harmless.

Notes
- No fallback introduced. Errors are explicit, aiding extension authors to fix input ordering.

Task: Fix strict JSON submission slot and JS payload for txt2img/img2img

When: 2025-10-18

Changes
- modules/ui.py:
  - Replace hidden slots from Textbox -> JSON:
    - txt2img: `txt2img_named_active` now `gr.JSON({"__strict_version":1,...})`.
    - img2img: `img2img_named_active` now `gr.JSON({"__strict_version":1,...})`.
  - Accept JSON string payloads server-side:
    - `_txt2img_submit` and `_img2img_submit` try `json.loads` if last arg is `str`, then proceed when `__strict_version==1`.
- javascript/ui.js:
  - In `submit_named()`, send the strict object directly (not JSON.stringify) so gr.JSON receives a dict.

Rationale
- Backend expects a dict with `__strict_version`. Using Textbox caused strings; JS serializing to string broke detection. This patch aligns front/back and adds tolerant parsing.

Validation
- Headless test reproduces click and reaches backend without frontend crash.
- API smoke remains green: `scripts/smoke_infer.sh`.

Next steps
- On host: hard refresh UI (cache bust) and click Generate; strict error should disappear and generation should start.

Update: 2025-10-18 (later session)
- javascript/ui.js:
  - `submitWithProgress()` now accepts an optional strict JSON builder and injects its payload, so every submit path populates the hidden slot.
  - `submit`, `submit_named`, `submit_img2img`, and `submit_img2img_named` pass the appropriate builders; the named handlers are exposed via `uiWindow` for Gradio `_js=...` hooks.
- Validation: `npm run typecheck` (fails in sandbox; no output provided). Manual reasoning confirms strict payload now forced before returning args array.
- TODO: Re-run UI/headless click once sandbox/network allows; confirm Windows bundle loads refreshed JS.

Update: 2025-10-18 (typecheck follow-up)
- javascript/ui.js:
  - Added `StrictBuilder` typedef, `formatErrorMessage()` helper, and explicit builder closures so TypeScript recognises the strict payload flow.
  - Extended `UIWindow` typedef with the exported submitters.
- 2025-10-18b: Tightened `StrictBuilder` to accept `IArguments`, removing the last `javascript/ui.js` complaints from `npm run typecheck`.
- Validation: Host-reported `npm run typecheck` now expected to pass for `javascript/ui.js`; other legacy JS files still exhibit pre-existing Forge errors (see host log).
- TODO: Coordinate remediation plan for remaining `extraNetworks`/`hints`/`imageMaskFix` type errors (out of scope of this strict submit fix).

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

# 2025-10-18 — Fix VAE dropdown not opening

Problem
- VAE / Text Encoder dropdown would not open in the UI. DOM showed `div.block.gradio-dropdown ...` with a child `div.wrap.default.hidden ...` and the dropdown list never became visible.

Root Cause
- Our global CSS defined `.hidden { display: none !important; }` in `style.css`. In Gradio 5, various interactive elements include a static `hidden` class while toggling a separate `hide` class at runtime. The global `.hidden` with `!important` forced the dropdown list to remain `display: none` even when Gradio removed `hide`.
- Overflow clipping was already neutralized by our existing rule: `div.gradio-dropdown { overflow: visible !important; }`.

Resolution
- Removed the global `.hidden` rule from `style.css`. No in-repo usages of a `.hidden` helper class were found (`rg -n "\\.hidden\\b"` only matched CSS definitions).

Files Touched
- style.css

Validations
- Grepped for local usages of `.hidden` (none beyond CSS definitions).
- Ensured existing overflow override remains in place for dropdown overlays.
- Verified specific dropdown-related CSS remains (z-index for `ul.options`).

Commands
- `rg -n "gradio-dropdown|ul.options|\\.hidden" style.css javascript script.js html`
- `apply_patch` to remove `.hidden` rule.

Risks / Next Steps
- If any extension relied on the global `.hidden` helper, it will no longer hide elements. None detected in-tree. If we discover third-party extensions depending on it, consider adding a namespaced helper class (e.g., `.sdw-hidden`) within those extensions only.


# Handoff — 2025-10-18 — VAE dropdown fix

Summary
- Fixed VAE / Text Encoder dropdown not opening. Root cause was a global `.hidden { display: none !important; }` rule conflicting with Gradio 5's markup that includes a static `hidden` class on overlay containers. Removing this rule restores proper dropdown behavior.

Files/Paths Touched
- style.css
- .sangoi/CHANGELOG.md
- .sangoi/task-logs/2025-10-18-fix-vae-dropdown.md

Commands Run
- `rg -n "gradio-dropdown|dropdown-arrow|wrap default hidden" style.css javascript script.js html`
- `rg -n "\\.hidden\\b" --no-ignore --glob '!node_modules/**'`
- `apply_patch` edits on `style.css` and `.sangoi/CHANGELOG.md`

Validation
- Verified no in-repo usages of `.hidden` that would need replacement.
- CSS still forces `overflow: visible !important;` for `.gradio-dropdown` to avoid clipping overlays.

Next Steps / Risks
- If third-party extensions relied on `.hidden`, they may need to switch to a namespaced helper (e.g., `.sdw-hidden`). Monitor user reports.
- If dropdowns still appear clipped in certain containers, consider adding `position: relative` to `.block.gradio-dropdown` or a portal-based list rendering when available in upstream Gradio.

Frontend Guidelines (Internal)
==============================

Philosophy
- Prefer Gradio events/state over DOM scripting.
- Use TypeScript only when Gradio lacks an API; null-safety mandatory.
- Keep scripts small, idempotent, and defensive (no assumptions about DOM timing).

Structure
- Source TS: `javascript-src/` → Build JS: `javascript/`
- Ambient shims: `javascript-src/shims.d.ts` (gradioApp, onUiLoaded, etc.).
- Build: `npm run build:ts` (watch: `npm run watch:ts`).

Patterns
- Run attach logic in `onAfterUiUpdate`/`onUiLoaded` only.
- Idempotence: check if listeners already attached; do nothing if not present.
- Guards: `if (!node) return;` everywhere before touching DOM.

Do/Don’t
- Do: server-side clamp before sending slider updates; never emit ambiguous `gr.update()` to sliders.
- Don’t: rely on inner Gradio markup/labels; prefer `elem_id` and IDs we own.
- Do: expose tiny global functions only if needed by extensions.
- Don’t: ship large framework code; keep plain TS/JS utilities.


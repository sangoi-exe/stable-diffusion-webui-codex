Gradio 5 Migration (Internal)
=============================

Purpose
- Stabilize Forge UI on Gradio 5.49.x without heavy client JS; prefer Python-first wiring.
- Only use TS/JS where Gradio lacks an equivalent event/state API.

Constraints
- Python ≥ 3.10, Gradio 5.49.x, Windows support (WSL and native), minimal dependencies.
- Avoid breaking existing UX; preserve features: Send-to, token counters, PNG Info paste, extras/upscalers, ControlNet pages.

Key Differences vs 4.x
- Stricter component preprocessing (sliders/numbers must satisfy min/max before Python runs).
- Tabs selection can be updated programmatically; regressions existed earlier but are fixed in late 5.x — still return a clean `gr.Tabs(selected=...)` update in a single hop.
- Queue uses `concurrency_limit` at listener level (not `concurrency_count`).
- Optional SSR (Node ≥ 20) — keep off by default for now.

Policy
- No DOM manipulation if a Gradio 5 API exists.
- TypeScript only where needed (Canvas overlays, legacy extension UIs, hotkeys).
- JS/TS must be null-safe (no DOM assumptions, idempotent attach, run in onAfterUiUpdate/onUiLoaded only).

Implemented Fixes
- Static shim: `/file=…` route in FastAPI to serve legacy JS/CSS assets.
- PNG Info paste: clamp sliders and avoid empty updates (prevent `-1 < 64`).
- Legacy JS: null checks in `inputAccordion.js`, `token-counters.js`, `ui.js`.
- Tokenizer: fallback to `tokenizer.json` fast tokenizer when `merges.txt` absent.

Upcoming Replacements (Python-first)
- Send-to tab switch: return `gr.Tabs(selected='…')` updates instead of `_js`.
- Token counters: `.change/.click` wiring with optional lightweight TS for local debounce.
- Accordions: use `gr.Accordion(open=…)`, drop DOM-proxy checkbox pattern.

SSR (Optional Later)
- Gate behind `launch(ssr_mode=True)` when Node 20+ exists; keep CSR default.

Validation Checklist
- PNG Info → Send to txt2img/img2img (no errors, tab switches, sliders valid).
- Generation hotkeys (Ctrl+Enter/Esc) work; if kept JS, ensure guards.
- ControlNet pages: no recurring console errors; fallbacks idempotent.


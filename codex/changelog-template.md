Changelog (Template)
====================

Date: YYYY-MM-DD

Added
- SSR default enabled (override via `GRADIO_SSR_MODE=0`).
- `GRADIO_JS_ALLOWLIST` and `GRADIO_JS_DENYLIST` to control injected JS.
- Extra Networks server-side filter/sort controls per page.

Changed
- InputAccordion now syncs `Checkbox` -> `Accordion.open` purely in Python (no JS hooks).
- Settings search handled on server (removes DOM filtering).

Removed
- Default injection of `token-counters.js` and `settings.js` when `GRADIO_JS_ALLOWLIST=auto`.

Migration Notes
- Extensions should avoid DOM queries reliant on Gradio internals; use `js=` or Python updates instead.
- For "send to …" flows, switch to `gr.Tabs.update(selected=...)`.

Validation
- SSR on/off load check; generation flows; settings search; accordion behavior; Extra Networks filter/sort.


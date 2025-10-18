Handoff: 2025-10-18

- Summary: Forced strict JSON payload injection via `submitWithProgress` builder hook; exposed submitters for Gradio `_js`, and added typings/error helper so `npm run typecheck` accepts the flow.
- Files: `javascript/ui.js`
- Commands: `npm run typecheck` (blocked in sandbox)
- Validation: Manual audit only; host run reported success for `javascript/ui.js`
- Next steps: 1) Re-run `npm run typecheck` on host (remaining Forge JS warnings persist outside this patch). 2) Headless/UI click test to confirm strict payload reaches backend (watch for cached JS).

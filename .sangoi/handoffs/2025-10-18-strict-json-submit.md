Handoff: 2025-10-18

- Summary: Forced strict JSON payload injection via `submitWithProgress` builder hook; exposed `submit_named`/`submit_img2img_named` for Gradio `_js` calls.
- Files: `javascript/ui.js`
- Commands: `npm run typecheck` (blocked by sandbox, no output)
- Validation: Manual audit only; typecheck pending due to sandbox failure.
- Next steps: 1) Re-run `npm run typecheck` on host. 2) Headless/UI click test to confirm strict payload reaches backend (watch for cached JS).

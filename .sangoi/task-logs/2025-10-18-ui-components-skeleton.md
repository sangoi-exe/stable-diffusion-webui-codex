Task: Front modularization (C + D) — components skeleton
Date: 2025-10-18

Changes
- JS components namespace `window.Codex.Components` and event bus:
  - javascript/codex.components.core.js
  - javascript/codex.components.sampler.js
  - javascript/codex.components.prompt.js
  - javascript/codex.components.hires.js
  - javascript/codex.components.canvas.js
- JS injection allowlist updated to include new component files.

Notes
- Contracts: front remains thin; back defines sampler/scheduler policy and strict JSON builders remain authoritative. Components expose read/write helpers for future refactors, without changing existing handlers.
- No behavior removed from ui.js; only added modular utilities.

Next
- Migrate specific behaviors from ui.js into the components progressively (prompt shortcuts, canvas tools, hires toggles), guarded by feature flags.


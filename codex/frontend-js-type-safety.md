Frontend JS Type Safety – Migration Notes (2025-10-12)
======================================================

Scope
- Enforce strict type-checking (`allowJs`+`checkJs`) across first‑party JS.
- Remove implicit `any`/DOM assumptions; add JSDoc types and guards.
- Preserve behaviour; no UX regressions or feature removals.

Tooling
- Typecheck (no emit): `npm run typecheck` → uses `tsconfig.json` (allowJs, noEmit).
- Build TS (optional): `npm run build:ts` → `tsconfig.build.json` (emits `javascript/`).
- Ambient shims: `javascript-src/shims.d.ts` (declares `gradioApp`, `onUi*`, `opts`, etc.).

Core Helpers (use these instead of ad‑hoc DOM)
- `getAppElementById(id: string): HTMLElement | null` – resolves IDs inside Gradio root with safe fallback.
- `updateInput(target: HTMLInputElement|HTMLTextAreaElement)` – dispatches a real `input` event for Gradio.
- `submitWithProgress(args, galleryContainerId, galleryId)` – standard submit flow + queue progress.
- `normalizeSubmitArgs(tabname, args)` – coerces numeric strings → numbers to avoid slider type errors.

Refactored Modules (high‑level summary)
- ui.js: split helpers; typed submit/restore flows; defensive query selection; exposed globals via `Object.assign(window, …)`.
- progressbar.js: typed `ProgressResponse`, wake‑lock guards, safe timers, live‑preview loop with `id_live_preview` continuity.
- resizeHandle.js: unified mouse/touch logic, explicit ResizeState, min‑width rules, debounced resize.
- imageviewer.js (+gamepad): strict modal lifecycle, safe preview updates, typed wheel/gamepad logic.
- imageMaskFix.js: guarded sibling lookups, image load handler with `{ once: true }`, sane scales.
- localization.js: declarative traversal, safe component resolution, typed dictionary with rtl support.
- notification.js: permission checks, volume clamping, options typed (icon+image), dedup count.
- profilerVisualization.js: typed row builder, stable sorting/expansion, safe number formatting.
- inputAccordion.js: JSDoc’d component API (visibleCheckbox/onVisibleCheckboxChange), idempotent attach.
- generationParams.js: gallery observers + button resolution with guards.
- edit‑attention.js / edit‑order.js: selection math typed; shortcut handlers guarded.
- hints.js / ui_settings_hints.js: tooltip pipeline and settings comments with batching and null safety.
- token-counters.js: guarded ID resolution under Gradio root, typed `onEdit` usage, centralized visibility toggle; keeps hidden-button click semantics.
- textualInversion.js: typed training kick-off (`requestProgress`) with safe element lookups and progress text rendering.

Patterns & Conventions
- All DOM reads are guarded (`instanceof` checks or query null checks) before usage.
- Shadow root safety: prefer `getAppElementById` over `document.getElementById`.
- Event typings: annotate listener params (`MouseEvent | TouchEvent | KeyboardEvent`) to avoid leak‑through anys.
- Never mutate argument order for Gradio submit; create new arrays (`Array.from(arguments)`).
- Wake Lock optionality: feature‑detect `navigator.wakeLock` and handle release failures.

What changed for integrators
- The legacy globals still exist (e.g., `window.submit`), but now point to typed implementations.
- Functions called from Python `_js=` hooks are preserved; signatures are unchanged.
- Where extensions used internals (e.g., DOM IDs), the behaviour is preserved; unsafe assumptions remain guarded.

How to add new JS safely
1) Add logic under `onUiLoaded`/`onAfterUiUpdate` and make it idempotent.
2) Resolve elements with `getAppElementById` or guarded `querySelector`.
3) If editing inputs, call `updateInput` to sync Gradio state.
4) For submit‑like flows, reuse `submitWithProgress`.
5) Run `npm run typecheck` and address any warnings before sending PRs.

Known follow‑ups
- Continue trimming implicit anys in `settings.js` and minor edge cases.
- Add unit smoke checks for wake‑lock path (feature detection only).

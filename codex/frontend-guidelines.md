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

Module Map (Hooks → Helpers)
- ui.js — hooks: window.submit, window.submit_txt2img_upscale, window.restoreProgress*; helpers: getAppElementById, submitWithProgress, normalizeSubmitArgs, updateInput, onEdit
- progressbar.js — hooks: requestProgress; helpers: formatTime; wakeLock guarded
- resizeHandle.js — hooks: onUiLoaded(setupAllResizeHandles); helpers: setLeftColGridTemplate; mouse/touch guards
- imageviewer.js — hooks: onAfterUiUpdate, DOMContentLoaded(buildModal); helpers: getAppElementById, modalImageSwitch
- imageviewerGamepad.js — events: gamepadconnected, wheel; helpers: sleepUntil, modalImageSwitch
- imageMaskFix.js — hooks: onAfterUiUpdate, window.resize; helpers: guarded canvas/image lookups
- localization.js — hooks: DOMContentLoaded, onUiUpdate; helpers: processNode, getTranslation
- notification.js — hooks: onAfterUiUpdate; helpers: getAppElementById, localSet/has focus checks
- profilerVisualization.js — hooks: showProfile (called from UI); helpers: createRow, addLevel
- inputAccordion.js — hooks: onUiLoaded; helpers: setupAccordion, updateInput
- generationParams.js — hooks: onAfterUiUpdate (MutationObserver), attachGalleryListeners; helpers: safe button resolution
- edit-attention.js — events: keydown; helpers: selection math functions
- edit-order.js — events: keydown; helpers: resolvePromptTextarea
- hints.js — hooks: onUiUpdate/onUiLoaded; helpers: enqueueTooltipCheck, updateTooltip
- ui_settings_hints.js — hooks: onOptionsChanged/onUiLoaded/onUiUpdate; helpers: requestGet, getAppElementById
- extraNetworks.js — hooks: onUiLoaded; helpers: requestGet, filtering/sorting registry, popup
- extensions.js — hooks: extensions_apply/check/install/toggle_all_extensions/toggle_extension; helpers: getExtensionCheckboxes
- dragdrop.js — events: document dragover/drop/paste; helpers: isValidImageList, dropReplaceImage, isURL
- settings.js — hooks: onUiLoaded/onUiUpdate/onOptionsChanged; helpers: getAppElementById, onEdit, settingsShowAllTabs
- token-counters.js — hooks: onUiLoaded/onUiUpdate/onOptionsChanged; helpers: onEdit, update_token_counter
- textualInversion.js — hooks: start_training_textual_inversion; helpers: requestProgress
- localStorage.js — helpers: localSet/localGet/localRemove (wrapped access to window.localStorage)

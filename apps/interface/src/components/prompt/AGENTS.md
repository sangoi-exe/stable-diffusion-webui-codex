<!-- tags: prompt, components, tiptap -->
# apps/interface/src/components/prompt Overview
Date: 2025-12-03
Last Review: 2026-04-08
Status: Active

## Purpose
- Houses the prompt editor widgets (PromptEditor, PromptBox/Fields) and prompt token parsing/serialization for generation views.

## Notes
- PromptEditor is built on Tiptap StarterKit plus the custom `PromptToken` node.
- `PromptToken` is an inline atomic node rendered through `PromptTokenChip`/`VueNodeViewRenderer`; keep its `renderHTML()` leaf-safe and never expose a ProseMirror content hole/contentDOM for that node.
- Serialization supports both ProseMirror JSON and Node shapes; validate prompt-token changes with `cd apps/interface && npm run typecheck` plus manual prompt round-trip checks in the UI.
- 2025-12-17: `PromptBox.vue` hides the char-count badge when empty (`0 chars`) to reduce visual noise in the WAN tab.
- 2025-12-22: `PromptBox.vue` badge now shows whitespace token count (`tok`) instead of raw character count.
- 2025-12-25: `PromptCard.vue` centralizes the prompt panel header (TI/LoRA/Styles) and negative-prompt visibility; most views default to hidden negative, while SDXL opts in to show it by default.
- 2025-12-26: `PromptCard.vue` panel header now renders the title as plain text (no wrapper `<span>`), consistent with the global panel header convention.
- 2025-12-27: Removed the redundant Checkpoints button from `PromptCard.vue` (checkpoint selection lives in QuickSettings); Styles “New/Apply” are now rendered as input-inline actions.
- 2026-01-03: Added standardized file header blocks to prompt components (non-test files) as part of the rollout (doc-only change).
- 2026-02-17: `PromptTokenChip.vue` now resolves the live ProseMirror node from `tr.doc.nodeAt(getPos())` before toggle/weight/remove mutations, fixing stale chip-state updates that previously required view switching.
- 2026-03-02: `PromptCard.vue` now accepts `hideNegative` as an explicit override (in addition to capability-driven hiding), so callers can suppress Negative Prompt fields when CFG semantics require it.
- 2026-04-08: `PromptCard.vue` still owns which prompt asset tools are exposed, but LoRA/TI modal chrome now comes from `components/modals/PromptAssetInsertModal.vue`; when `hideNegative` is active, the LoRA modal also hides its Negative target action instead of offering a dead insert path.
- 2026-07-31: `PromptBox.vue` treats `tokenEngine` as genuinely optional: when a generation surface has no truthful text-only tokenizer count, it hides the badge and sends no `/api/models/prompt-token-count` request instead of rendering a synthetic `tok !` error.

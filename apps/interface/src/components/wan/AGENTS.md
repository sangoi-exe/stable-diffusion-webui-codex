<!-- tags: frontend, components, wan22, video -->
# apps/interface/src/components/wan Overview
Date: 2025-12-14
Last Review: 2026-08-26
Status: Active

## Purpose
- WAN-derived presentational components used by current image/video surfaces and the reference-only WAN workspace.

## Key Files
- `WanStagePanel.vue` — High/Low stage controls (sampler/scheduler/steps/cfg/seed + optional flow-shift), no stage-level LoRA controls.
- `WanSubHeader.vue` — Small section sub-header used by WAN surfaces to keep “Video / High / Low” headers consistent.
- `WanVideoOutputPanel.vue` — Video export and interpolation controls (format/pix_fmt/loop/CRF/pingpong/return-frames plus interpolation output FPS).

## Notes
- Keep these components dumb: props in, emits out. Do not fetch inventory or call backend APIs here.
- Prefer shared selectors (`SamplerSelector.vue`, `SchedulerSelector.vue`) over ad-hoc `<select>` blocks.
- 2025-12-20: Removed stage-level “Lightning/Use LoRA” checkboxes; LoRA selection is now a per-stage `<select>` shown only in `LightX2V` mode (WAN QuickSettings).
- 2025-12-22: `WanStagePanel.vue` now uses SDXL-style sliders + steppers for Steps/CFG and moves seed actions (🎲/↺) inside the seed input.
- 2025-12-23: WAN panels use shared gen-card layout primitives (`gc-row`, `gc-col`, `row-split`) and the `gen-card--embedded` variant; do not reintroduce parallel `wan22-*` or `cdx-form-row` layout helper layers.
- 2025-12-23: `WanStagePanel.vue` renders Steps/CFG via `components/ui/SliderField.vue` (label+input header, slider below) for parity with the rest of the WebUI.
- 2025-12-23: WAN sliders use `cdx-input-w-md` sizing (removes WAN-only `w-step/w-cfg` CSS).
- 2025-12-26: `WanStagePanel.vue` now places Sampler/Scheduler/Steps on the first row and Seed/CFG on the second; stage LoRA UI no longer has a dedicated component owner in this folder.
- 2025-12-28: Added `WanSubHeader.vue` and made `WanVideoOutputPanel.vue` embeddable so WAN-derived video surfaces can compose “Video Output” without nested card borders; Interpolation (RIFE) is now a single toggle button.
- 2025-12-29: `WanVideoOutputPanel.vue` renders the RIFE toggle inline with the other output toggles (Ping-pong/Save/Trim) for layout parity.
- 2026-01-03: Added standardized file header blocks to WAN components (doc-only change; part of rollout).
- `WanStagePanel.vue` exposes both stage sampler and stage scheduler selectors; callers pass WAN-filtered sampler/scheduler inventories into those selectors, scheduler recommendations remain hints, stage scheduler is no longer clearable to an empty/inherit value, and unsupported scheduler combinations fail loud in backend WAN route/runtime validation.
- 2026-01-27: Added a `Return frames` toggle to `WanVideoOutputPanel.vue` (default off) and an inline note when `Save output` is off (frames still returned so users can download them).
- 2026-02-20: `WanSubHeader.vue` now supports opt-in full-row toggle behavior (`clickable` + `header-click`), with built-in interactive-target exclusion and Enter/Space keyboard parity for collapsible cards.
- 2026-02-21: `WanStagePanel.vue` dropped stage-level LoRA UI; WAN LoRA insertion is prompt-level in the live video owner using prompt token chips.
- 2026-02-28: WAN prompt-level LoRA inserts maintain stage `loras[]` SHA arrays (`high/low`, dedupe-by-sha with latest weight), while these WAN components remain presentational-only.
- 2026-02-27: `WanVideoOutputPanel.vue` removed `Filename Prefix`, `Save output`, `Save metadata`, and `Trim to audio` controls; output controls now render as `Format + Pixel Format` row, `Loop Count + CRF` slider row, compact `Ping-pong + Return frames` toggle row, and one interpolation output-FPS slider (`0=Off`, active values map to backend interpolation times).
- 2026-02-27: `WanVideoOutputPanel.vue` now renders `Interpolation (RIFE)` in the same slider row as `Loop Count` and `CRF` (three compact sliders on one row).
- 2026-02-27: `WanVideoOutputPanel.vue` interpolation now represents output target FPS (not a multiplier), and `Ping-pong`/`Return frames` render stacked vertically.
- 2026-02-27: `WanVideoOutputPanel.vue` output-toggle column now uses `gc-col--presets` so the toggle stack is content-width and the three sliders consume the remaining row space.
- 2026-08-26: `WanVideoOutputPanel.vue` no longer exposes SeedVR2. The dedicated `/video-upscale` utility owns its curated controls and result flow.

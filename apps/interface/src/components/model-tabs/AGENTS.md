# apps/interface/src/components/model-tabs Overview
Date: 2026-03-13
Last Review: 2026-08-26
Status: Active

## Purpose
- Hold reference-only video bodies used by the canonical route owners under `src/views/**`.

## Files
- `apps/interface/src/components/model-tabs/WanVideoWorkspace.vue` — reference-only WAN source workspace kept as the mechanical baseline for the `VideoModelTab.vue` cutover.

## Notes
- `apps/interface/src/views/VideoModelTab.vue` is now the canonical baseline video workspace under `src/views/**`.
- `apps/interface/src/views/VideoTabRouteView.vue` is the thin route selector while current video families still branch.
- `apps/interface/src/views/video-model/**` now owns the live family-specific runtime helpers (`VideoModelTabWanRuntime.vue`, `VideoModelTabLtxRuntime.vue`).
- Keep `WanVideoWorkspace.vue` reference-only and non-live; maintenance-only compatibility/truth sync is allowed while it remains the mechanical source baseline.
- Do not invent a second generic video owner or compatibility layer here.
- Shared presentational seams belong in `apps/interface/src/components/**`; shared styling belongs in `apps/interface/src/styles/**`.
- 2026-03-21: `WanVideoWorkspace.vue` now passes the WAN `4n+1` frame contract explicitly into the shared `VideoSettingsCard.vue`; the workspace owns WAN frame alignment instead of relying on shared-component defaults.
- 2026-04-03: LTX no longer has a live body owner in this folder; the strict geometry/frame/execution-profile contract now renders through `apps/interface/src/views/VideoModelTab.vue` plus `apps/interface/src/views/video-model/VideoModelTabLtxRuntime.vue`.
- 2026-04-28: While `WanVideoWorkspace.vue` remains reference-only, its maintenance-sync prompt token counters must still use the exact `wan22_14b` token engine; do not keep generic `wan` / `wan22` token-count aliases in this baseline.
- 2026-08-26: The reference-only WAN baseline no longer carries SeedVR2 settings, snapshots, or output controls. SeedVR2 is a dedicated route-local utility, not a model-tab follower.

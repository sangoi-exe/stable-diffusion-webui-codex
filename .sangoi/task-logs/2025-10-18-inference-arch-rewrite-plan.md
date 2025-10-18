Task: Draft per-model inference engine architecture, video tasks, and UI tabs plan
Date: 2025-10-18

Summary
- Defined modular engine architecture (SD15, SDXL, FLUX, video) with registry and orchestrator.
- Spec’d new tabs `Txt2Vid` and `Img2Vid` and request types.
- Wrote docs under `.sangoi/one-trainer/architecture/` (overview, API, layout, UI plan, migration, rollout).

Files/Paths
- `.sangoi/one-trainer/architecture/*.md`
- `.sangoi/one-trainer/proposal.md` (updated)
- `.sangoi/CHANGELOG.md` (updated)

Risks
- Scope creep across engines; mitigate via phased PRs and feature flags.
- Dependency divergence for vídeo (ffmpeg, decoders). Consider process isolation for video engines.

Next
- Upon approval, scaffold `backend/core/*` and empty `backend/engines/*` with tests (no behavior change behind flags).

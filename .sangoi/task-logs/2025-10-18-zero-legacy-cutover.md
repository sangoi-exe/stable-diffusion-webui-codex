# Task: Enforce zero legacy compatibility (cutover starts with txt2img)
Date: 2025-10-18

Summary
- Updated migration/rollout docs to remove any adapters/fallbacks and to mandate direct Orchestrator usage.
- Replaced `modules/txt2img.py` JSON submit path to build a Codex `Txt2ImgRequest` and route through `InferenceOrchestrator` using the selected `codex_engine` and current checkpoint.

Files/Paths
- `.sangoi/one-trainer/architecture/migration-plan.md`
- `.sangoi/one-trainer/architecture/rollout-architecture.md`
- `.sangoi/CHANGELOG.md`
- `modules/txt2img.py`

Notes
- img2img still uses the legacy path; next step is to switch its handler similarly.

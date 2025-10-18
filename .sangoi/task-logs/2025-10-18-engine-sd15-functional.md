# Task: Implement functional SD15 engine (txt2img/img2img/inpaint/upscale)
Date: 2025-10-18

Summary
- SD15 engine now bridges the new orchestrator to existing WebUI processing (`modules.processing`) without touching UI.
- Loading uses `modules.sd_models.model_data.forge_loading_parameters` to keep compatibility with current reload logic.
- Methods implemented: `txt2img`, `img2img`, `inpaint` (same path with mask), and `upscale` (leveraging hires-fix fields).

Files/Paths
- `backend/engines/sd15/engine.py` (load + methods; emits `ProgressEvent` and `ResultEvent`)
- `backend/engines/util/adapters.py` (builders for `StableDiffusionProcessing*` from request dataclasses)

Manual Validation
- Follow `.sangoi/one-trainer/architecture/manual-validation.md` → Functional SD15 section.
- Expect events `start`→`prepare`→`end` and a `ResultEvent` with `images` and `info` JSON.

Notes/Risks
- Uses global `shared.sd_model` + Forge reload path; relies on correct `model_ref` resolution (UI title/alias/hash).
- Error handling: exceptions propagate; messages include engine/task/model; stack traces preserved in logs.

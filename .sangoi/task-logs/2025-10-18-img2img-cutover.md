# Task: Cut img2img to Orchestrator (zero legacy)
Date: 2025-10-18

Summary
- `modules/img2img.py::img2img_from_json` no longer calls legacy processing; it now constructs `Img2ImgRequest` and invokes `InferenceOrchestrator` with the selected `codex_engine` and `sd_model_checkpoint`.
- Source image/mask are derived from active UI inputs (img2img/sketch/inpaint variants); masks use alpha-channel where applicable.

Files/Paths
- `modules/img2img.py`
- `.sangoi/CHANGELOG.md`

Manual Validation
- Use UI img2img (any subtab). Ensure Engine dropdown is set; pick a checkpoint. Generate and observe events/result.
- For inpaint subtabs, verify mask application via obvious region edits.

Notes/Risks
- Batch modes and edge-case composites rely on UI-provided canvases; errors will be verbose if inputs are invalid.

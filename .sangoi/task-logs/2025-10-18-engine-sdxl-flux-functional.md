# Task: Implement functional SDXL and Flux engines
Date: 2025-10-18

Summary
- SDXL engine now mirrors SD15 bridging for `txt2img`, `img2img`, `inpaint`, and `upscale`.
- Flux engine supports `txt2img` (only); other tasks intentionally raise `UnsupportedTaskError` with detailed context.

Files/Paths
- `backend/engines/sdxl/engine.py` (methods + load)
- `backend/engines/flux/engine.py` (txt2img + load)
- `.sangoi/one-trainer/architecture/manual-validation.md` (added sections for SDXL/Flux)
- `.sangoi/CHANGELOG.md` (append engine notes)

Manual Validation
- Use the orchestrator commands from the SD15 section, switching `engine_key` and `model_ref` accordingly.

Notes/Risks
- Same global model reload path; ensure correct checkpoint resolution by UI title/hash.
- For Flux, img2img/inpaint/upscale are not wired on purpose; they must error loudly.

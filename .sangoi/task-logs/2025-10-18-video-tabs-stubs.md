# Task: Add Txt2Vid/Img2Vid tabs and register video engine stubs
Date: 2025-10-18

Summary
- Added new tabs in UI: Txt2Vid and Img2Vid, each with prompt row and basic controls (width/height, steps, fps, frames, sampler/scheduler, seed). Strict JSON submit wired to Orchestrator.
- Registered stub video engines: `svd` (img2vid) and `hunyuan_video` (txt2vid/img2vid). Currently raise UnsupportedTaskError (explicit, no silencing).

Files/Paths
- `modules/ui.py` (new Blocks for txt2vid/img2vid + submit handlers)
- `javascript/ui.js` (builders+submitters for video strict payloads)
- `modules/txt2vid.py`, `modules/img2vid.py` (Codex requests → Orchestrator)
- `backend/engines/video/{svd,hunyuan}/engine.py` (+ __init__.py)
- `backend/engines/__init__.py`, `backend/engines/registration.py`
- `.sangoi/CHANGELOG.md`

Manual Validation
- Open UI; confirm new tabs. Select Engine (`hunyuan_video` for Txt2Vid, `svd` for Img2Vid) and a checkpoint. Generate to observe explicit UnsupportedTaskError (until engines are implemented).

Next
- Implement functional video engines and optionally render previews (frames → gallery or assembled mp4) when available.

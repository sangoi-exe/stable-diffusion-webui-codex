# Task: Implement engine/mode/task preset logic (fill-only)
Date: 2025-10-18

Summary
- Added `backend/core/presets.py` to load YAML presets and apply a fill-only patch to Requests before engine validation.
- Seed presets created for sd15/sdxl/flux/hunyuan_video/wan_ti2v_5b.

Files
- `backend/core/presets.py`
- `configs/presets/*`
- Engine integrations: SD15/SDXL/Flux updated to apply preset before `parse_params`.

Manual validation
- Omit `steps` or sampler/scheduler, generate, and confirm `applied_preset_patch` in logs and no validation errors.

Notes
- No fallback across modes/engines; absence logs “Preset not found”.

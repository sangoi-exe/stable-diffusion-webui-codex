# Task: Implement per-engine parameter interfaces + parser registry
Date: 2025-10-18

Summary
- Added typed dataclasses for required/optional params per Engine × Task, plus enums and parser registry.
- Integrated engines to call the parser early for validation.

Files/Paths
- `backend/core/enums.py`
- `backend/core/params/{base,sd15,sdxl,flux,video}.py`
- `backend/core/param_registry.py`
- engines updated to call `parse_params(...)`.
- Docs: `.sangoi/one-trainer/architecture/params-interfaces.md`

Notes
- Validation failures raise `ValidationError` with explicit field/context; no silent fallbacks.

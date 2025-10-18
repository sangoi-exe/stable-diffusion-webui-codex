# Task: Scaffold SD15/SDXL/Flux engine stubs and registration wiring
Date: 2025-10-18

Summary
- Added `backend/engines/` package with base helper, registration helpers, and stub classes for SD15, SDXL, and Flux (capabilities + explicit unsupported errors).
- Exposed `register_default_engines()` to opt-in the new engines and documented manual validation commands.

Files/Paths
- `backend/engines/__init__.py`
- `backend/engines/base.py`
- `backend/engines/registration.py`
- `backend/engines/sd15/engine.py`
- `backend/engines/sd15/__init__.py`
- `backend/engines/sdxl/engine.py`
- `backend/engines/sdxl/__init__.py`
- `backend/engines/flux/engine.py`
- `backend/engines/flux/__init__.py`
- `.sangoi/one-trainer/architecture/manual-validation.md`
- `.sangoi/CHANGELOG.md`

Validation
- Manual-only plan documented; commands provided to check registry wiring and explicit `UnsupportedTaskError` messaging. Not executed yet.

Next Steps
- Implement actual execution paths for SD15/SDXL/Flux engines behind feature flags.
- Wire orchestrator and new dropdown UI once engines are functional.

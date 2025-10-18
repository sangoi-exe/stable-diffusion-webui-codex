# Task: Scaffold modular inference core package (PR1 groundwork)
Date: 2025-10-18

Summary
- Added `backend/core/` package with engine interface, request dataclasses, registry, and orchestrator skeleton.
- Created unit tests covering registry registration/lookup and orchestrator dispatch/error paths.

Files/Paths
- `backend/core/__init__.py`
- `backend/core/engine_interface.py`
- `backend/core/requests.py`
- `backend/core/registry.py`
- `backend/core/orchestrator.py`
- `backend/core/exceptions.py`
- `tests/test_backend_core_registry.py`
- `.sangoi/CHANGELOG.md`

Validation
- Intended to run `python -m pytest tests/test_backend_core_registry.py`; skipped because `pytest` is not installed in the sandbox (module import failure).

Next Steps
- Implement concrete engines (sd15/sdxl/flux) behind feature flags.
- Wire orchestrator into Gradio handlers after engine stubs are ready.

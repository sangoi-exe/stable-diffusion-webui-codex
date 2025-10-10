# Testing & Tooling

Although automated CI coverage is limited, the repository ships with scripts and conventions to validate functionality locally.

## Python Environment
- Preferred dependency lock resides in `requirements_versions.txt`; use `launch.py --reinstall-xformers` or `--reinstall-torch` to refresh critical packages.
- Conda environments (`environment-wsl2.yaml`) and one-click installers bundle vetted CUDA/PyTorch versions. Align development environments with these to reduce incompatibility reports.

## Sanity Checks
- `webui.py --skip-torch-cuda-test` can be used for smoke testing startup without GPU checks.
- `modules/call_queue.py` and `modules/api/api.py` provide endpoints for automated regression tests; see `scripts/` for batch utilities.
- Backend regression harnesses are emerging in `backend/tests/` (create as needed). Mirror legacy behaviour before asserting new expectations.

## Performance Diagnostics
- Enable verbose memory logs with `--backend-log-memory` (see `backend/args.py`). Use `backend/memory_management.py` metrics to compare VRAM usage before/after refactors.
- Profiling hooks in `backend/utils.py` and `modules/shared.py` help collect timings for sampler loops and attention layers.

## Frontend Verification
- Use `--gradio-queue` and WebSocket streaming from `backend/stream.py` to test interactive features.
- Run linting for custom JS via `node_modules/.bin/eslint` (configure rules under `package.json`).
- Localization consistency can be checked with scripts under `modules_forge/localization/` when adding keys.

## Documentation Workflow
- Whenever workflows change, update guides inside `codex/` and user-facing docs (`README.md`, `NEWS.md`).
- Record manual test matrices in merge descriptions so QA can replay them. Include GPU configuration, sampler choice, and model weights involved.

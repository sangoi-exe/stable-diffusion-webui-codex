# Testing & Tooling

Although automated CI coverage is limited, the repository ships with scripts and conventions to validate functionality locally.

## Python Environment
- Preferred dependency lock resides in `requirements_versions.txt`; use `launch.py --reinstall-xformers` or `--reinstall-torch` to refresh critical packages.
- Python 3.12 is the supported runtime; avoid mixing interpreter versions when running scripts or caching compiled assets.
- Create virtual environments with `python -m venv .venv && source .venv/bin/activate` (PowerShell: `.\.venv\Scripts\Activate.ps1`) before installing dependencies.

## Sanity Checks
- `webui.py --skip-torch-cuda-test` can be used for smoke testing startup without GPU checks.
- `modules/call_queue.py` and `modules/api/api.py` provide endpoints for automated regression tests; see `scripts/` for batch utilities.
- Backend regression harnesses are emerging in `backend/tests/` (create as needed). Mirror legacy behaviour before asserting new expectations.

## Txt2Img Baseline Capture
- Run `python scripts/capture_txt2img_baselines.py --config codex/examples/txt2img_baselines.sample.json --output-dir tests/backend/fixtures/txt2img --overwrite` from an environment where Forge has already initialised models (e.g., after `python launch.py --skip-install --exit`).
- The script bootstraps the Forge runtime, executes each scenario with a tqdm progress bar, and writes PNGs plus `metadata.json` bundles per scenario under the chosen output directory.
- Supply a bespoke JSON config when validating new samplers or checkpoints; the sample file in `codex/examples/` documents the schema (prompts, hires settings, overrides).
- For GPU-backed validation, prefer the Windows 11 / RTX 3060 (12 GB) / Ryzen 9 5950X / 64 GB / PyTorch 2.7.1 + CUDA 12.8 host; record the exact command + commit when capturing new fixtures.

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

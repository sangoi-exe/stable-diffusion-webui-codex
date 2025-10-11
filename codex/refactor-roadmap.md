Refactor Roadmap
================

Purpose: make the Forge backend simpler, more robust, and easier to debug while keeping user‑visible behaviour intact. When behaviour must change, document it and ship safe defaults.

Guiding Rules
- Preserve functional parity; fix root causes, not symptoms.
- Prefer small, proven building blocks over custom abstractions.
- Naming discipline: replace meaningless identifiers (e.g., `p`, `x2`) with descriptive names as you touch code. Do not preserve poor structure for history’s sake when tests/baselines protect behaviour.
- Avoid destructive git commands; never touch `AGENTS.md`.

Scope and Milestones
1) Backend Modernization (txt2img, base path)
   - Done: sampler prep in backend runtime (`rng.ImageRNG`, forge object restore, token‑merging, script batch hooks, extra‑network activation, post‑sample callbacks, `shared.state` updates).
   - Pending: hires/refiner orchestration (conditioning, checkpoint reloads), noise‑schedule overrides, seed/iteration bookkeeping in multi‑batch runs.

2) Logging overhaul (rich/color)
   - Done: centralized `modules/logging_config.py` with Rich/Colorama, tqdm‑aware handler, opt‑out fallback.
   - Next: unify per‑module loggers, replace ad‑hoc prints, document SD_WEBUI_LOG_LEVEL.

3) Dependencies and Tooling
   - Done: `scripts/update_requirements.py` to bump pins via PyPI while excluding PyTorch family; `--drop-excluded` for locally compiled torch.
   - Next: routine pin refresh CI step (documented command), narrow any remaining backtracking hot‑spots.
   - TODO: add a manual installer helper that installs each pinned package with `pip install --no-deps` in a stable order (reads `requirements_versions.txt`, supports `--subset` filters, prints per‑package results, and never touches torch/vision/audio). Document that resolver is bypassed by design; users must keep pins coherent.

4) Testing Strategy
   - Baselines: `scripts/capture_txt2img_baselines.py` for deterministic PNG+metadata capture (manual GPU host).
   - Day‑to‑day: manual validation through the UI (LoRA, hires, refiner, multi‑iter), with targeted unit tests only where behaviour is subtle.

Risk Register (active)
- LoRA registry bridging between UI and headless entrypoints (ensure listing before activation).
- Hires/refiner model reload timing vs. memory manager.
- Extension ordering assumptions in user environments.

Exit Criteria
- Backend runtime owns the full txt2img orchestration (including hires/refiner) with equivalent behaviour under representative prompts/baselines.
- Logs are readable by default (Rich) and controllable via env.
- Requirements can be refreshed without breaking torch pins.

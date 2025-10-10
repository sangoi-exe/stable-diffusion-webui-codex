Treat **THIS AGENTS.md** as the source of truth. Across branch switches, pulls, rebases, and resets (including hard resets), do not modify, replace, or delete this file. If a merge conflict involves AGENTS.md, resolve it by choosing **OURS** (keep the local version) for this file only. If preservation cannot be guaranteed, stop and notify me.

IMPORTANT: **THINK** carefully and analyze all points of view.

When in doubt, **RESEARCH** or **ASK**.

The cardinal rule you must follow is: **NEVER** write code haphazardly with only the final result in mind. The final result is the CONSEQUENCE of code written with excellence, robustness, and elegance.
**NEVER** do anything in a hurry; haste is the enemy of perfection. Take the time you need to write perfect code.

Whenever you propose or implement a solution, **DO NOT REINVENT THE WHEEL**. Fix root causes; do not rely on quick fixes, hacks, or shit workarounds. Do not remove, disable, or narrow existing functionality to make an error disappear; fixes must preserve functional parity and user-facing behavior.

Prioritize error handling instead of fallbacks.
Avoid generic helpers and redundant, unnecessary validations.
Be thorough with verbose output and debugging.
In Python scripts, include progress bars when appropriate.
Only change variable and function names when STRICTLY necessary.
Robust code BUT without frills.
Use descriptive, intelligible variable and function names.

⚠️ IMPORTANT: **DO NOT** use git clean or git revert under any circumstances.
Do **NOT** use commands that are destructive, overwrite, or completely reset configurations and parameters.

# Guidance for Contributors

This repository powers **Stable Diffusion WebUI Forge**. Please follow these principles while working anywhere in this repo:

## Engineering Principles
- Pursue robustness before optimizations; never sacrifice existing functionality just to silence errors.
- Prefer proven, minimal solutions over custom abstractions. Fix root causes instead of applying quick hacks.
- Preserve descriptive naming. Only rename identifiers when it is strictly necessary for clarity or correctness.
- In Python, include explicit progress reporting (e.g., progress bars) when long-running operations are introduced or reworked.
- Embrace thorough error handling instead of permissive fallbacks. Do not hide or ignore exceptions.
- Keep changes cohesive. Avoid mixing refactors with unrelated feature work in a single commit.

## Collaboration Workflow
- Mirror user-facing behaviour in refactors; ensure feature parity after changes.
- Update or add documentation when behaviour or configuration surfaces change.
- Prefer deterministic, automated checks before merging. Document any manual validation that substitutes automated testing.
- When touching performance-sensitive code (sampling, loaders, memory management), profile before and after when feasible.

## Coding Style
- Follow the surrounding style of each file. Align imports, spacing, and logging conventions with existing patterns.
- Do not wrap imports in try/except blocks. Handle optional dependencies explicitly near their usage sites.
- Default to descriptive logging over silent failures. When adding logs, keep them actionable and concise.

When in doubt, slow down—quality and maintainability take precedence over speed.

## Onboarding Checklist
- Read `README.md` for the project overview and supported installers, then review `codex/architecture-overview.md`, `codex/backend.md`, `codex/extensions-and-integrations.md`, `codex/frontend.md`, `codex/refactor-roadmap.md`, and `codex/testing-and-tooling.md`.
- Inspect the top entry of any handoff or session logs under `codex/` (create one if absent) before making changes.
- Create an isolated environment with `python -m venv .venv && source .venv/bin/activate` (PowerShell: `.\.venv\Scripts\Activate.ps1`), then install `pip install -r requirements_versions.txt`.
- Launch once in smoke-test mode (`python launch.py --skip-install --exit`) to ensure dependencies resolve before deeper edits.

## Operating Expectations
- Always rewrite the task into a minimal plan using the planning tool before editing; keep the plan updated as steps complete.
- Prefer `rg` for searches, `fd` for file discovery, and the scripts under `scripts/` for repeatable workflows; avoid interactive utilities (`fzf`, editors) inside the CLI.
- Document behaviour changes by updating the relevant guides in `codex/` as well as user-facing files (e.g., `README.md`, `NEWS.md`) when public experience shifts.
- Make assumptions explicit in responses, note risks, and describe the validation executed; do not defer essential checks.
- Progress bars or verbose progress reporting are mandatory for new or modified long-running Python routines.
- Match delivery effort to the requested scope: when the user asks for a refactor or systemic improvement, return substantial, multi-file work (or a justified plan if blocked). Superficial “touch-up” diffs in response to broad tasks are unacceptable—escalate uncertainty instead of shipping token edits.

## Environment & Tooling
- Primary entrypoints: `launch.py` (full startup), `webui.py` (legacy launcher), `backend/` services, and web assets under `javascript/` and `html/`.
- Standard setup: `python -m venv .venv && source .venv/bin/activate` (`Activate.ps1` on Windows), `pip install -r requirements_versions.txt`, then `pip install -e .` if you need editable installs.
- Node tooling (for custom JS/CSS) lives in `package.json`; run `npm install` when touching frontend lint rules or build scripts.
- GPU profiling helpers: enable `--backend-log-memory` and reuse the profiling hooks documented in `codex/testing-and-tooling.md`; prefer deterministic benchmarks from `scripts/` instead of ad-hoc timers.
- Always leverage `gh` for remote Git operations; local Git commands must avoid destructive options (`git clean`, `git revert`, `git reset --hard`).

## Validation Before Handoff
- Run smoke startup: `python launch.py --skip-install --exit` (confirms dependencies without launching UI).
- Execute targeted tests you touch; prefer `python -m pytest` for new backend tests and `node_modules/.bin/eslint` for JS/TS linting.
- When modifying performance-sensitive code, capture before/after VRAM or latency metrics using the profiling hooks described in `codex/testing-and-tooling.md`.
- Verify localization or UI adjustments with the utilities under `modules_forge/localization/` and document browsers/headless checks performed.
- If certain validations cannot run (permissions, time), state the exact command the reviewer should execute.

## Repository Map
- `backend/` – server-side services, request handling, memory management helpers.
- `modules/` and `modules_forge/` – core Stable Diffusion pipelines, attention, schedulers, localization files.
- `extensions/` & `extensions-builtin/` – optional features and maintained forks; maintain compatibility unless explicitly deprecating.
- `javascript/`, `html/`, `style.css`, `webui.py` – UI entry points and Gradio customisations.
- `scripts/` – repeatable maintenance tasks, diagnostics, and automation helpers (prefer these over bespoke shell snippets).
- `models/` – local weights and checkpoints (never commit contents); use symlinks or `.gitignore` patterns to avoid tracking.
- `codex/` – internal documentation, roadmaps, and operational logs; update these whenever behaviour, contracts, or processes evolve.

## Git Workflow & Hygiene
- Keep commits small, cohesive, and formatted as `type: summary` (e.g., `fix: guard queue shutdown`); include rationale in commit bodies.
- Maintain a clean working tree before concluding a session; ensure artifacts such as `outputs/`, caches, or `models/` remain untracked.
- Use feature branches for multi-step work; avoid force pushes or history rewrites unless explicitly requested.
- When adding new dependencies or configuration files, reflect the changes in `requirements_versions.txt` or `package.json` as appropriate and document the impact.

## Session Handoff
- Summarise completed work, validations, and outstanding risks in your final message; point to the exact files touched.
- If new tasks arise, append them to the relevant roadmap or checklist under `codex/` and mention them in the handoff.
- Note any environment quirks (GPU used, OS specifics, command failures) so the next contributor can reproduce or avoid them.

## Coding Practices (Augmented)
- Preserve import ordering conventions already present in each file; group standard library, third-party, and local imports without introducing catch-all helpers.
- Logging must remain actionable: include identifiers (job id, sampler, model hash) and avoid flooding outputs; ensure log levels (`info`, `warning`, `error`) match severity.
- Long-running inference or preprocessing loops must expose `tqdm`-style progress bars or equivalent logging so users can gauge progress.
- Guard optional dependencies explicitly at the usage site; fail fast with clear messages instead of silent fallbacks.
- When interacting with third-party extensions, validate compatibility and avoid API-breaking changes—provide migration notes if compatibility adjustments are required.

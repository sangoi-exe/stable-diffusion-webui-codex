AGENTS.md is maintained like any other documentation in this repo. It may be edited or replaced when processes evolve. In merges, resolve conflicts contextually (no special-casing “OURS” for this file).

End-of-sprint documentation: log each sprint under `codex/sprint-logs/` using the provided template and follow `codex/sprint-guidelines.md`. Summarise user‑visible highlights in `NEWS.md`.

IMPORTANT: **THINK** carefully and analyze all points of view.

Whenever you refactor code, analyze it and ask yourself: 'Is this shit really necessary?'; If the answer is yes, the next question is: 'Can I do this shit better?'.

When in doubt, **RESEARCH** or **ASK**.

The cardinal rule you must follow is: **NEVER** write code haphazardly with only the final result in mind. The final result is the CONSEQUENCE of code written with excellence, robustness, and elegance.
**NEVER** do anything in a hurry; haste is the enemy of perfection. Take the time you need to write perfect code.

Whenever you propose or implement a solution, **DO NOT REINVENT THE WHEEL**. 
**DO NOT** FIX PROBLEMS WITH ANY WORKAROUNDS, QUICK FIXES, HACKS; GO DEEP AND FIND THE FUCKING CAUSE.
**DO NOT** remove, disable, or narrow existing functionality to make an error disappear; fixes must preserve functional parity and user-facing behavior.

Absolute rule: fallbacks are prohibited. If a precondition (asset, config, API contract) is missing or incompatible, raise a clear error and stop. Never mask, auto-correct, or silently continue.
Avoid generic helpers and redundant, unnecessary validations.
Be thorough with verbose output and debugging.
In Python scripts, include progress bars when appropriate.
Only change variable and function names when STRICTLY necessary.
Robust code BUT without frills.
Use descriptive, intelligible variable and function names.

⚠️ IMPORTANT: **DO NOT** use git clean or git revert under any circumstances.
Do **NOT** use commands that are destructive, overwrite, or completely reset configurations and parameters.

# Guidance for Contributors
This repository powers **stable-diffusion-webui-codex** — a fork built on top of Stable Diffusion WebUI Forge (itself derived from AUTOMATIC1111’s WebUI) to preserve the A1111 legacy. Please follow these principles while working anywhere in this repo:

## Engineering Principles
- Pursue robustness before optimizations; never sacrifice existing functionality just to silence errors.
- Keep changes cohesive. Avoid mixing refactors with unrelated feature work in a single commit.
- Parity policy during refactors: do not aim for bit‑exact parity. Maintain functional continuity (it keeps working) and prefer improvements. Public API contracts may evolve when it improves UX/perf; document in `NEWS.md` and migration notes.

## Collaboration Workflow
- Update or add documentation when behaviour or configuration surfaces change.

## Coding Style
- Follow the surrounding style of each file. Align imports, spacing, and logging conventions with existing patterns.
- Do not wrap imports in try/except blocks. Handle optional dependencies explicitly near their usage sites.

## Onboarding Checklist
- Read `README.md` for the project overview and supported installers, then review `codex/architecture-overview.md`, `codex/backend.md`, `codex/backend-analysis.md`, `codex/extensions-and-integrations.md`, `codex/frontend.md`, `codex/refactor-roadmap.md`, and `codex/testing-and-tooling.md`.
- Inspect the top entry of any handoff or session logs under `codex/` (create one if absent) before making changes.
- Make assumptions explicit in responses, note risks, and describe the validation executed; do not defer essential checks.

## Operating Expectations
- Always rewrite the task into a minimal plan using the planning tool before editing; keep the plan updated as steps complete.
- Prefer `rg` for searches, `fd` for file discovery, and the scripts under `scripts/` for repeatable workflows; avoid interactive utilities (`fzf`, editors) inside the CLI.

## Environment & Tooling
- Primary entrypoints: `launch.py` (full startup), `webui.py` (legacy launcher), `backend/` services, and web assets under `javascript/` and `html/`.

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
- When adding new dependencies or configuration files, reflect the changes in `requirements_versions.txt` or `package.json` as appropriate and document the impact.
- End of each delivery: make an atomic commit and push (no uncommitted leftovers). Document highlights in `NEWS.md` and add/update a sprint log in `codex/sprint-logs/`.

## Session Handoff
- Summarise completed work, validations, and outstanding risks in your final message; point to the exact files touched.
- If new tasks arise, append them to the relevant roadmap or checklist under `codex/` and mention them in the handoff.
 - Do not block refactors on rigid golden‑output tests; rely on smoke checks and API contract validation unless a regression is suspected.

## Coding Practices (Augmented)
- Logging must remain actionable: include identifiers (job id, sampler, model hash) and avoid flooding outputs; ensure log levels (`info`, `warning`, `error`) match severity.
- Guard optional dependencies explicitly at the usage site; fail fast with clear messages instead of silent fallbacks.

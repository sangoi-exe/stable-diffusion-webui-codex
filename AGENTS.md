
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
Only change variable and function names when STRICTLY necessary.
Robust code BUT without frills.
Use descriptive, intelligible variable and function names.
In Python scripts, include progress bars when appropriate.

⚠️ IMPORTANT: **DO NOT** use git clean or git revert under any circumstances.

# Guidance for Contributors
End-of-sprint documentation: log each sprint under `codex/sprint-logs/` using the provided template and follow `codex/sprint-guidelines.md`. Summarise user‑visible highlights in `NEWS.md`.

## Engineering Principles
 - Parity policy during refactors: do not aim for bit‑exact parity. Maintain functional continuity (it keeps working) and prefer improvements. Public API contracts may evolve when it improves UX/perf; document in `NEWS.md` and migration notes.
 
## Collaboration Workflow
- Update or add documentation when behaviour or configuration surfaces change.

## Onboarding Checklist
- Read `README.md` for the project overview and supported installers, then review `codex/architecture-overview.md`, `codex/backend.md`, `codex/backend-analysis.md`, `codex/extensions-and-integrations.md`, `codex/frontend.md`, `codex/refactor-roadmap.md`, and `codex/testing-and-tooling.md`.
- Inspect the top entry of any handoff or session logs under `codex/` (create one if absent) before making changes.
- Make assumptions explicit in responses, note risks, and describe the validation executed; do not defer essential checks.

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

Sprint Documentation Guidelines
==============================

Purpose
- Capture an auditable summary at the end of every sprint (or sub-sprint) covering changes, risks, validations, and next steps.

Where to write
- File per sprint under `codex/sprint-logs/` named `YYYY-MM-DD_sprint-N.md`.
- Also update `NEWS.md` for user‑visible highlights.

Minimal structure (copy/paste)
- Title: `Sprint N – YYYY-MM-DD`
- Context:
  - Scope, goals, constraints (e.g., Gradio version, feature flags).
- Completed:
  - Bulleted list of user-facing and internal changes.
  - Exact files touched (paths), notable env vars, and any migrations.
- Validation:
  - Manual checks run (smoke flows), commands used, and observed results.
  - Console/network errors encountered and how resolved.
  - Screenshots or logs if helpful (do not commit artifacts, include paths/commands).
- Risks/Follow-ups:
  - Regressions to watch, extension impacts, perf/memory notes.
- Next Sprint:
  - Concrete, short list of items with expected outcome.

Conventions
- Keep bullets concise; prefer links to code paths instead of long snippets.
- When behavior or configuration surfaces change, also update any relevant docs (README, migration guides) in the same PR.
- Do not silence or hide errors; note them under Validation and Risks.

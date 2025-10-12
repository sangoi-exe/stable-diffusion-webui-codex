Python Type Safety (Pyright) – Setup & Usage
===========================================

Why Pyright here
- We want “absurdly strict” typing that screams early when values drift (e.g., str vs int for sliders), without hacks.
- Pyright runs fast and enforces strict null/unknown-type checks across the repo.

Config
- Root config: `pyrightconfig.json` (strict mode; unknown-type checks on; library types used).
- Includes: `backend/`, `modules/`, `modules_forge/`, top-level `webui.py`, `launch.py`, `spaces.py`.
- Excludes: third-party, legacy, caches, and UI bundles.

Run (global or NPX)
- Global (preferred if you installed pyright globally):
  - `pyright --stats`  # quick stats
  - `pyright --outputjson > pyright.json`  # machine-readable report
- Portable (no global install):
  - `npx -y pyright --stats`

Saved reports
- Script: `scripts/pyright-report.sh`
  - Writes to `codex/reports/pyright/pyright-YYYYMMDD-HHMMSS.{txt,json}`
  - Falls back to `npx` if `pyright` isn’t on PATH.

Interpreting failures
- “reportUnknown*”: add explicit annotations on parameters/variables and use `typing` (e.g., `int|None`, `Annotated[int, …]`).
- “reportMissingImports”: install or add stubs for the missing package; for pure runtime-only deps, consider `pyright: ignore[reportMissingImports]` on the narrow import line.
- “reportMissingTypeStubs”: prefer `pip install types-<package>` or ship a minimal `*.pyi` under a `typings/` package if needed.

Scope policy
- Do not relax checks repo-wide. If a violation is justified, use targeted `# pyright: ignore[rule-id]` with a short reason, or add a narrow override in `pyrightconfig.json` under a path-specific rule.

Next steps
- Start by cleaning hot paths used by Gradio endpoints (e.g., `modules/ui.py`, generation pipelines). Add or refine function signatures so request payloads are typed and validated before use.


Python Type Safety (Pyright) – Setup & Usage
===========================================

Why Pyright here
- We want “absurdly strict” typing that screams early when values drift (e.g., str vs int for sliders), without hacks.
- Pyright runs fast and enforces strict null/unknown-type checks across the repo.

Config
- Root config: `pyrightconfig.json` (strict mode; library types used).
- Workspace‑only mode (default here): focuses on `webui.py`, `launch.py`, `spaces.py`, and `modules/ui*.py`.
  - This avoids noisy import errors when the venv lives fora do workspace (ex.: Windows venv).
  - Rules for missing imports/type stubs are downgraded to warnings.
  - Unknown‑type diagnostics are warnings to avoid cascade quando libs faltam.
  - Core soundness (missing parameter types, incompatible overrides, call issues) permanece em erro.

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
- Não relaxe globalmente sem motivo — mas no workspace‑only mode é ok priorizar “sinal útil”:
  - Se precisar checar backend com venv disponível, ajuste `include` e promova `reportMissingImports` para `error` localmente.
  - Prefira ignores direcionados (`# pyright: ignore[rule-id]`) ao invés de afrouxar regras globais.

Next steps
- Start by cleaning hot paths used by Gradio endpoints (e.g., `modules/ui.py`, generation pipelines). Add or refine function signatures so request payloads are typed and validated before use.

Refactor Roadmap
================

Purpose: make the stable-diffusion-webui-codex backend simpler, more robust, and easier to debug while keeping user‑visible behaviour intact. When behaviour must change, document it and ship safe defaults.

Project Identity and Baseline
- `stable-diffusion-webui-codex` is a fork built on top of Forge (itself a fork of AUTOMATIC1111’s WebUI) to preserve the A1111 legacy.
- Development leverages OpenAI “codex” as a coding assistant; no LLM is embedded in runtime.
- Use `legacy/` (snapshot of Forge main) as the functional reference pipeline when refactors break behaviour; restore parity before evolving design.

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

3) Gradio 5 Migration
   - Goal: audit all JS/Gradio interactions and update to Gradio 5.49.x APIs/semantics (events, _js callbacks, components’ value models, stricter validation, tab switching, gallery extraction).
   - Tasks:
     - Replace deprecated _js handlers and wiring that changed between 4.x → 5.x (e.g., auto switch tab on “Send to …”).
     - Normalize component updates to respect min/max and choice sets (done for paste; extend across UI flows).
     - Verify queue/event lifecycles and update Blocks/Routes usage where needed.
     - Introduce TypeScript for legacy JS modules with strict null checks to prevent DOM-time errors; ship JS builds in-place for Gradio to load.
        - Scaffolding: `tsconfig.json`, `javascript-src/`, `npm run build:ts` to output into `javascript/`.
        - Convert first-wave modules: `inputAccordion`, `token-counters`, `ui` helpers.
        - Provide ambient shims for global helpers (e.g., `gradioApp`, `onUiLoaded`) in `javascript-src/shims.d.ts`.
   - Validation: manual UI tests (PNG Info → txt2img/img2img, send to upscaler, hires widgets), plus a quick smoke of API routes.

3) Dependencies and Tooling
- Done: `tools/update_requirements.py` to bump pins via PyPI while excluding PyTorch family; `--drop-excluded` for locally compiled torch.
- Next: routine pin refresh CI step (documented command), narrow any remaining backtracking hot‑spots.
- TODO: add a manual installer helper that installs each pinned package with `pip install --no-deps` in a stable order (reads `requirements_versions.txt`, supports `--subset` filters, prints per‑package results, and never touches torch/vision/audio). Document that resolver is bypassed by design; users must keep pins coherent.

References
- See `codex/gradio-migration.md` for the live Gradio 5 plan.
- See `codex/frontend-guidelines.md` for TS/JS usage discipline.
- See `codex/dependencies-policy.md` for pin/update policy.

4) Testing Strategy
   - Baselines: manual capture via WebUI using pinned seeds/configs; store outputs and metadata under `tests/backend/fixtures/txt2img/` when needed.
   - Day‑to‑day: manual validation through the UI (LoRA, hires, refiner, multi‑iter), com testes unitários apenas onde o comportamento é sutil.

Risk Register (active)
- LoRA registry bridging between UI and headless entrypoints (ensure listing before activation).
- Hires/refiner model reload timing vs. memory manager.
- Extension ordering assumptions in user environments.

Exit Criteria
- Backend runtime owns the full txt2img orchestration (including hires/refiner) with equivalent behaviour under representative prompts/baselines.
- Logs are readable by default (Rich) and controllable via env.
- Requirements can be refreshed without breaking torch pins.

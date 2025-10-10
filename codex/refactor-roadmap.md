# Refactor Roadmap

This roadmap aligns modernization work across the repository. Update it as milestones progress or priorities shift.

## 1. Backend Modernization
- [ ] Port txt2img execution path to `backend/diffusion_engine/txt2img.py` with feature parity (CFG, hires fix, refiner handoff).
- [ ] Implement shared scheduler registry bridging `modules/sd_samplers.py` and `backend/sampling/`.
- [ ] Consolidate memory reporting between `modules/shared.py` and `backend/memory_management.py` to avoid duplicate UI toggles.

## 2. Flux & Advanced Model Support
- [ ] Document Flux configuration knobs in UI and backend docs; ensure tutorials map to runtime switches.
- [ ] Expand GGUF/BnB operator coverage via `backend/operations_gguf.py` and `backend/operations_bnb.py`.
- [ ] Validate LoRA+Flux workflows and capture findings in `codex/testing-and-tooling.md`.

## 3. UI & Accessibility
- [ ] Audit tabs defined in `modules/ui.py` for redundant components and migrate shared patterns to `modules_forge/ui/`.
- [ ] Ensure localization keys exist for all Forge-only UI controls. Track missing translations in `localizations/` issues.
- [ ] Improve tablet/canvas responsiveness by profiling event handlers in `javascript/canvas/`.

## 4. Extension Ecosystem
- [ ] Review bundled extensions for compatibility with the new backend interfaces.
- [ ] Provide migration guides for popular external extensions impacted by backend refactors.
- [ ] Introduce automated smoke tests for extension loading (target: top 5 community extensions).

## 5. Tooling & Developer Experience
- [ ] Establish linting presets for Python (ruff/black) and JavaScript (eslint) aligned with Forge style.
- [ ] Automate VRAM regression tracking using hooks in `backend/memory_management.py`.
- [ ] Document manual QA matrices for release candidates inside `codex/testing-and-tooling.md`.

Revisit this roadmap after each milestone, archiving completed tasks and adding new priorities that emerge from user feedback or upstream changes.

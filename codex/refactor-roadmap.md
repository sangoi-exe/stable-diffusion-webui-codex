# Refactor Roadmap

This roadmap aligns modernization work across the repository. Update it as milestones progress or priorities shift. Beyond the baseline webui upgrade, every stream must aggressively remove overengineering and wheel-reinvention—**without** trampling the historical context or regressions called out in `codex/`. Favour the simplest fix that preserves behaviour and document any trade-offs.

> **Status Note:** See `codex/refactor-roadmap-status.md` for the latest assessment,
> blockers, and next-step recommendations for each milestone.

## 1. Backend Modernization

### Milestone 1.1 – Txt2Img runtime parity
- [ ] **Discovery:** Capture baseline behaviour by instrumenting `modules/processing.py` with golden txt2img runs (CFG, hires fix, refiner, LoRA stack). Archive reference configs + seeds.
- [ ] **Test harness:** Build deterministic fixtures and samplers in `tests/backend/test_txt2img.py`, covering base diffusion pass, cached first pass reuse, hires upscaling, and refiner scheduling.
- [ ] **Implementation:** Port orchestration into `backend/diffusion_engine/txt2img.py`, wiring sampler selection, callbacks, and hires hand-off while satisfying collected fixtures.
- [ ] **Integration:** Flip `modules/processing.py` to call the backend runtime behind a feature flag, run regression suite + manual parity verification, then remove the legacy path once stable.

### Milestone 1.2 – Scheduler registry convergence
- [ ] **Schema design:** Draft canonical scheduler metadata (id, display name, parameter schema, supported models) and review with extension maintainers.
- [ ] **Registry implementation:** Implement registry module in `backend/sampling/registry.py` with adapters consumed by `modules/sd_samplers.py`.
- [ ] **Migration:** Update UI bindings and extension hooks to reference the registry, providing shims for deprecated aliases and documenting migration steps.
- [ ] **Validation:** Extend automated tests to assert sampler equivalence across legacy and registry-backed paths, plus smoke run with popular extensions.

### Milestone 1.3 – Unified memory telemetry
- [ ] **Model definition:** Specify shared data contract for VRAM/CPU metrics (fields, units, sampling cadence) aligned with UI expectations.
- [ ] **Backend publisher:** Refactor `backend/memory_management.py` to emit the unified payload and expose subscription API.
- [ ] **UI consumers:** Update `modules/shared.py` and Forge UI toggles to consume the publisher; remove duplicate polling implementations.
- [ ] **Verification:** Add regression tests for telemetry payloads and document monitoring workflow in `codex/testing-and-tooling.md`.

## 2. Flux & Advanced Model Support

### Milestone 2.1 – Flux surface area audit
- [ ] Inventory every Flux toggle in UI tabs (`modules/ui.py`, `modules_forge/ui/`), mapping labels to runtime flags.
- [ ] Propose unified naming and expose a central config schema consumed by UI + backend.
- [ ] Update documentation in `codex/backend.md` and user-facing guides to reflect the canonical terminology.

### Milestone 2.2 – GGUF/BnB operator enablement
- [ ] Catalogue unsupported operators with frequency estimates using telemetry from representative checkpoints.
- [ ] Implement missing kernels or integrate upstream equivalents inside `backend/operations_gguf.py` and `backend/operations_bnb.py`.
- [ ] Add regression tests and benchmarking scripts to validate accuracy/perf on CUDA, ROCm, and CPU fallback paths.

### Milestone 2.3 – LoRA + Flux validation
- [ ] Script automated workflows exercising stacked LoRAs with Flux checkpoints via existing automation harnesses.
- [ ] Capture output artefacts and establish acceptance thresholds for colour, detail, and prompt adherence.
- [ ] Document recommended settings and troubleshooting matrix in `codex/testing-and-tooling.md`.

## 3. UI & Accessibility

### Milestone 3.1 – Tab consolidation
- [ ] Audit tabs/components in `modules/ui.py` and `modules_forge/ui/` for duplicated controls.
- [ ] Extract shared fragments into reusable helpers while ensuring localisation bindings remain intact.
- [ ] Smoke test with the default extension set and document migration guidelines for custom tabs.

### Milestone 3.2 – Localisation coverage
- [ ] Generate report of missing Forge-only keys across locales using localisation tooling.
- [ ] Coordinate translation updates or mark keys requiring community input.
- [ ] Add CI check (or documented manual workflow) to prevent future omissions.

### Milestone 3.3 – Canvas responsiveness
- [ ] Profile `javascript/canvas/` handlers on representative tablets to locate event-loop bottlenecks.
- [ ] Apply targeted refactors (debounce/throttle, offloading heavy work) without regressing pointer precision.
- [ ] Validate improvements through recorded metrics and update frontend changelog.

## 4. Extension Ecosystem

### Milestone 4.1 – Bundled extension audit
- [ ] Inventory bundled extensions, classify by backend touchpoints, and define minimal smoke scripts.
- [ ] Execute compatibility runs against the new backend interfaces, capturing failure logs per extension.
- [ ] File remediation tasks per issue, linking to root-cause analysis.

### Milestone 4.2 – External migration guides
- [ ] Summarise breaking backend API changes affecting extensions.
- [ ] Draft migration playbooks highlighting API replacements and testing checklists.
- [ ] Publish guides within `codex/extensions-and-integrations.md` and announce in maintainer channels.

### Milestone 4.3 – Automated extension smoke tests
- [ ] Design pytest harness capable of loading top community extensions in isolation.
- [ ] Implement deterministic environment setup (mock assets, minimal models) and run on CI runners.
- [ ] Track pass/fail metrics and gate merges on sustained regressions.

## 5. Tooling & Developer Experience

### Milestone 5.1 – Linting baseline
- [ ] Propose ruff/black + eslint/prettier configurations aligned with current style.
- [ ] Trial configs on a small subset of files, iterating until diffs are acceptable.
- [ ] Roll out via pre-commit hooks and document contribution workflow updates.

### Milestone 5.2 – VRAM regression automation
- [ ] Extend `backend/memory_management.py` with hooks for capturing allocation metrics per inference job.
- [ ] Build reporting pipeline (CI job or script) storing baselines for key workflows.
- [ ] Surface trends in dashboards or release notes, highlighting regressions.

### Milestone 5.3 – Manual QA matrices
- [ ] Enumerate manual validation scenarios not covered by automation.
- [ ] Capture environment/setup instructions, expected outcomes, and sign-off gates in `codex/testing-and-tooling.md`.
- [ ] Integrate the matrix into release checklist templates.

## 6. Runtime & Dependency Modernization

### Milestone 6.1 – Python 3.12 + CUDA 12.4 adoption
- [ ] Compile platform matrix (Windows, WSL, Linux, macOS) noting blockers per dependency.
- [ ] Update launch scripts and installers with conditional logic for the new runtime.
- [ ] Run smoke tests covering mediapipe-dependent preprocessors and document parity results.

### Milestone 6.2 – Checkpoint/embedding reload profiling
- [ ] Build benchmarking script measuring reload latency across common model directories.
- [ ] Integrate upstream weight-reload optimisations, adapting Forge hooks as needed.
- [ ] Record before/after metrics and ensure regression alarms are documented.

### Milestone 6.3 – Low-core CPU validation
- [ ] Provision or simulate low-core CPU environments.
- [ ] Validate GGUF embedding loaders and soft inpainting pipeline, adding regression scripts utilising `joblib` and `protobuf 4.x`.
- [ ] Document findings and remediation backlog for uncovered issues.

### Milestone 6.4 – Dependency audit & upstream parity
- [ ] Catalogue legacy dependency pins with historical rationale.
- [ ] Propose replacements or upgrades that unblock Python 3.12+, including testing notes.
- [ ] Track divergences from ComfyUI/SwarmUI, summarising high-impact gaps in `codex/comfyui-swarmui-postforge.md` and creating follow-up tasks.

Revisit this roadmap after each milestone, archiving completed tasks and adding new priorities that emerge from user feedback or upstream changes.

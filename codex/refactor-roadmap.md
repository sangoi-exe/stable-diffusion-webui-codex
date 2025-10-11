# Refactor Roadmap

This roadmap aligns modernization work across the repository. Update it as milestones progress or priorities shift. Beyond the baseline webui upgrade, every stream must aggressively remove overengineering and wheel-reinvention—**without** trampling the historical context or regressions called out in `codex/`. Favour the simplest fix that preserves behaviour and document any trade-offs.

> **Status Note:** See `codex/refactor-roadmap-status.md` for the latest assessment,
> blockers, and next-step recommendations for each milestone.

### Naming Discipline
- When refactoring, rename meaningless identifiers (e.g., `p`, `x2`, `foo`) to descriptive names that reflect their role. Keep renames scoped and justified; do not leave ambiguous throwaway names in migrated code.
- Do not preserve poor legacy structure for its own sake: when behaviour is covered by tests or captured baselines, prefer a clean reimplementation that expresses intent clearly over incremental patchwork.

## 1. Backend Modernization

### Milestone 1.1 – Txt2Img runtime parity
- [ ] **Discovery:** Capture baseline behaviour by instrumenting `modules/processing.py` with golden txt2img runs (CFG, hires fix, refiner, LoRA stack). Archive reference configs + seeds.
- [x] **Baseline capture script:** Landed `scripts/capture_txt2img_baselines.py` plus sample config under `codex/examples/`; generates PNG + metadata bundles in `tests/backend/fixtures/txt2img/` for regression comparisons.
- [x] **Test harness:** Extend deterministic fixtures in `tests/backend/test_txt2img.py` to cover base sampling, modified-noise overrides, hires reload sequencing, and LoRA object hand-off; expand further as refiner scheduling migrates.
- [ ] **Callback parity:** Mirror script/extension callbacks by adding adapter shims in `backend/diffusion_engine/txt2img.py` and asserting invocation order in unit tests.
- [ ] **Implementation:** Port orchestration into `backend/diffusion_engine/txt2img.py`, wiring sampler selection, callbacks, and hires hand-off while satisfying collected fixtures.
- [ ] **Integration:** Flip `modules/processing.py` to call the backend runtime behind a feature flag, run regression suite + manual parity verification, then remove the legacy path once stable.
- [ ] **Documentation:** Update `codex/backend.md` with setup instructions for capturing new baselines whenever samplers or schedulers change.

### Milestone 1.2 – Scheduler registry convergence
- [ ] **Schema design:** Draft canonical scheduler metadata (id, display name, parameter schema, supported models) and review with extension maintainers.
- [ ] **Legacy mapping audit:** Enumerate scheduler identifiers currently declared in `modules/sd_samplers.py` and bundled extensions, capturing edge-case parameters.
- [ ] **Registry implementation:** Implement registry module in `backend/sampling/registry.py` with adapters consumed by `modules/sd_samplers.py`.
- [ ] **Extension shims:** Add compatibility layer exporting legacy names + parameter coercion, documenting deprecation windows in `codex/extensions-and-integrations.md`.
- [ ] **Migration:** Update UI bindings and extension hooks to reference the registry, providing shims for deprecated aliases and documenting migration steps.
- [ ] **Validation:** Extend automated tests to assert sampler equivalence across legacy and registry-backed paths, plus smoke run with popular extensions.
- [ ] **CI enforcement:** Introduce lint/test guard that fails when a sampler bypasses the registry, ensuring future additions stay centralized.

### Milestone 1.3 – Unified memory telemetry
- [ ] **Model definition:** Specify shared data contract for VRAM/CPU metrics (fields, units, sampling cadence) aligned with UI expectations.
- [ ] **Sampling cadence prototype:** Implement a proof-of-concept polling loop under `backend/memory_management.py` that logs payload schema to confirm completeness before UI wiring.
- [ ] **Backend publisher:** Refactor `backend/memory_management.py` to emit the unified payload and expose subscription API.
- [ ] **UI consumers:** Update `modules/shared.py` and Forge UI toggles to consume the publisher; remove duplicate polling implementations.
- [ ] **Telemetry persistence:** Store snapshots from regression runs under `tests/backend/fixtures/memory_telemetry/` to detect breaking changes.
- [ ] **Verification:** Add regression tests for telemetry payloads and document monitoring workflow in `codex/testing-and-tooling.md`.
- [ ] **Alerting hooks:** Expose optional callbacks so extensions can subscribe to telemetry events without duplicating sampling logic.

## 2. Flux & Advanced Model Support

### Milestone 2.1 – Flux surface area audit
- [ ] Inventory every Flux toggle in UI tabs (`modules/ui.py`, `modules_forge/ui/`), mapping labels to runtime flags.
- [ ] Trace runtime usage in `backend/diffusion_engine` and `modules_forge/main_entry.py`, noting conflicts or redundant switches.
- [ ] Propose unified naming and expose a central config schema consumed by UI + backend.
- [ ] Update documentation in `codex/backend.md` and user-facing guides to reflect the canonical terminology.
- [ ] Publish change log entry summarising renamed toggles to steer extension maintainers.

### Milestone 2.2 – GGUF/BnB operator enablement
- [ ] Catalogue unsupported operators with frequency estimates using telemetry from representative checkpoints.
- [ ] Prioritise operator backlog by referencing failure logs captured in `logs/gguf_compatibility/`.
- [ ] Implement missing kernels or integrate upstream equivalents inside `backend/operations_gguf.py` and `backend/operations_bnb.py`.
- [ ] Add regression tests and benchmarking scripts to validate accuracy/perf on CUDA, ROCm, and CPU fallback paths.
- [ ] Document device-specific limitations and mitigation steps in `codex/backend.md`.

### Milestone 2.3 – LoRA + Flux validation
- [ ] Script automated workflows exercising stacked LoRAs with Flux checkpoints via existing automation harnesses.
- [ ] Capture output artefacts and establish acceptance thresholds for colour, detail, and prompt adherence.
- [ ] Validate metadata (CFG, sampler, LoRA weights) is preserved in png info blocks to aid troubleshooting.
- [ ] Document recommended settings and troubleshooting matrix in `codex/testing-and-tooling.md`.
- [ ] Publish sample gallery in `codex/frontend.md` to illustrate expected quality bounds.

## 3. UI & Accessibility

### Milestone 3.1 – Tab consolidation
- [ ] Audit tabs/components in `modules/ui.py` and `modules_forge/ui/` for duplicated controls.
- [ ] Catalogue per-tab dependency on `modules/scripts.py` hooks to avoid regressions when consolidating components.
- [ ] Extract shared fragments into reusable helpers while ensuring localisation bindings remain intact.
- [ ] Smoke test with the default extension set and document migration guidelines for custom tabs.
- [ ] Produce UX notes describing consolidated layout and update screenshots where applicable.

### Milestone 3.2 – Localisation coverage
- [ ] Generate report of missing Forge-only keys across locales using localisation tooling.
- [ ] Automate diff of `modules_forge/localization` vs upstream vanilla WebUI keys to detect drift.
- [ ] Coordinate translation updates or mark keys requiring community input.
- [ ] Add CI check (or documented manual workflow) to prevent future omissions.
- [ ] Update localisation contributor guide under `codex/frontend.md` with submission workflow.

### Milestone 3.3 – Canvas responsiveness
- [ ] Profile `javascript/canvas/` handlers on representative tablets to locate event-loop bottlenecks.
- [ ] Capture baseline frame-time traces and commit them to `codex/testing-and-tooling.md` for future comparisons.
- [ ] Apply targeted refactors (debounce/throttle, offloading heavy work) without regressing pointer precision.
- [ ] Validate improvements through recorded metrics and update frontend changelog.
- [ ] Communicate device/browser compatibility notes in `NEWS.md` once optimisations land.

## 4. Extension Ecosystem

### Milestone 4.1 – Bundled extension audit
- [ ] Inventory bundled extensions, classify by backend touchpoints, and define minimal smoke scripts.
- [ ] Tag each extension with maintainer contact/last update date in a shared spreadsheet referenced from `codex/extensions-and-integrations.md`.
- [ ] Execute compatibility runs against the new backend interfaces, capturing failure logs per extension.
- [ ] File remediation tasks per issue, linking to root-cause analysis.
- [ ] Track remediation status in `codex/refactor-roadmap-status.md` to keep follow-ups visible.

### Milestone 4.2 – External migration guides
- [ ] Summarise breaking backend API changes affecting extensions.
- [ ] Collect extension maintainer feedback via issue templates or surveys to validate pain points before finalising guidance.
- [ ] Draft migration playbooks highlighting API replacements and testing checklists.
- [ ] Publish guides within `codex/extensions-and-integrations.md` and announce in maintainer channels.
- [ ] Schedule follow-up review to update guides as backend milestones complete.

### Milestone 4.3 – Automated extension smoke tests
- [ ] Design pytest harness capable of loading top community extensions in isolation.
- [ ] Integrate harness with `scripts/ci_extension_matrix.py` (create if missing) to orchestrate install/load sequences.
- [ ] Implement deterministic environment setup (mock assets, minimal models) and run on CI runners.
- [ ] Track pass/fail metrics and gate merges on sustained regressions.
- [ ] Publish troubleshooting doc for flaky extensions, including sample log output.

## 5. Tooling & Developer Experience

### Milestone 5.1 – Linting baseline
- [ ] Propose ruff/black + eslint/prettier configurations aligned with current style.
- [ ] Trial configs on a small subset of files, iterating until diffs are acceptable.
- [ ] Capture before/after examples in `codex/testing-and-tooling.md` to set contributor expectations.
- [ ] Roll out via pre-commit hooks and document contribution workflow updates.
- [ ] Add CI status badge or doc callout once linting enforces across the repo.

### Milestone 5.2 – VRAM regression automation
- [ ] Extend `backend/memory_management.py` with hooks for capturing allocation metrics per inference job.
- [ ] Build reporting pipeline (CI job or script) storing baselines for key workflows.
- [ ] Surface trends in dashboards or release notes, highlighting regressions.
- [ ] Archive baseline metrics per release under `codex/testing-and-tooling.md` for historical comparison.
- [ ] Provide contributor instructions on generating ad-hoc VRAM reports locally.

### Milestone 5.3 – Manual QA matrices
- [ ] Enumerate manual validation scenarios not covered by automation.
- [ ] Capture environment/setup instructions, expected outcomes, and sign-off gates in `codex/testing-and-tooling.md`.
- [ ] Integrate the matrix into release checklist templates.
- [ ] Include traceability fields (tester, date, build hash) to align with QA audits.
- [ ] Link manual scenarios to automated counterparts where available to highlight coverage gaps.

## 6. Runtime & Dependency Modernization

### Milestone 6.1 – Python 3.12 + CUDA 12.4 adoption
- [ ] Compile platform matrix (Windows, WSL, Linux, macOS) noting blockers per dependency.
- [ ] Draft phased rollout timeline with contingency for extensions lacking wheels.
- [ ] Update launch scripts and installers with conditional logic for the new runtime.
- [ ] Run smoke tests covering mediapipe-dependent preprocessors and document parity results.
- [ ] Prepare community announcement outlining upgrade path and rollback instructions.

### Milestone 6.2 – Checkpoint/embedding reload profiling
- [ ] Build benchmarking script measuring reload latency across common model directories.
- [ ] Capture baseline metrics for SD1.x, SDXL, Flux, and LoRA-heavy workflows to inform success criteria.
- [ ] Integrate upstream weight-reload optimisations, adapting Forge hooks as needed.
- [ ] Record before/after metrics and ensure regression alarms are documented.
- [ ] Share profiling methodology in `codex/backend.md` for reproducibility.

### Milestone 6.3 – Low-core CPU validation
- [ ] Provision or simulate low-core CPU environments.
- [ ] Validate GGUF embedding loaders and soft inpainting pipeline, adding regression scripts utilising `joblib` and `protobuf 4.x`.
- [ ] Document findings and remediation backlog for uncovered issues.
- [ ] Establish minimum hardware recommendations and note any unsupported features in `README.md`.
- [ ] Capture CPU utilisation/latency charts in `codex/testing-and-tooling.md` for historical tracking.

### Milestone 6.4 – Dependency audit & upstream parity
- [ ] Catalogue legacy dependency pins with historical rationale.
- [ ] Propose replacements or upgrades that unblock Python 3.12+, including testing notes.
- [ ] Track divergences from ComfyUI/SwarmUI, summarising high-impact gaps in `codex/comfyui-swarmui-postforge.md` and creating follow-up tasks.
- [ ] Produce dependency upgrade playbook covering verification steps and rollback plan per package.
- [ ] Schedule quarterly review to keep the audit current as upstream releases ship.

Revisit this roadmap after each milestone, archiving completed tasks and adding new priorities that emerge from user feedback or upstream changes.

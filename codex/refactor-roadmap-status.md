# Refactor Roadmap – Current Status & Blockers

This document captures the present status of each roadmap initiative and lists the
identified blockers that prevented completion during this iteration. Use it as the
starting point for the next work session so that progress can resume without
re-evaluating the fundamentals.

## 1. Backend Modernization

### Port txt2img execution path to `backend/diffusion_engine/txt2img.py`
- **Status:** Blocked
- **Dependencies:** Tight coupling between `modules/processing.py` and UI script hooks;
  missing unit coverage to ensure parity for CFG, hires fix, refiner hand-off, and
  LoRA interactions.
- **Next Steps:** Extract deterministic fixtures for the txt2img sampler path, then
  migrate noise preparation, script callbacks, and refiner scheduling into
  `backend/diffusion_engine/txt2img.py` with regression tests around CFG + hires.

### Implement shared scheduler registry bridging `modules/sd_samplers.py` and `backend/sampling/`
- **Status:** Blocked
- **Dependencies:** Conflicting sampler naming between legacy modules and backend
  adapters. No consolidated source-of-truth for scheduler configuration.
- **Next Steps:** Design a schema describing sampler metadata, then refactor both
  modules to consume the registry through adapter classes while keeping extension
  overrides intact.

### Consolidate memory reporting between `modules/shared.py` and
`backend/memory_management.py`
- **Status:** Blocked
- **Dependencies:** Divergent telemetry paths (UI toggles vs. backend hooks) and lack
  of integration tests for VRAM/CPU reporting.
- **Next Steps:** Define unified data model for memory statistics, expose a single
  publisher in the backend, and update UI bindings accordingly.

## 2. Flux & Advanced Model Support

### Document Flux configuration knobs in UI and backend docs
- **Status:** In Progress
- **Progress:** Created a shared schema (`backend/diffusion_engine/flux_config.py`) that
  enumerates every Flux preset field and drives both the UI defaults and sampler
  dropdown registrations. Updated `codex/backend.md`, `codex/frontend.md`, and the new
  `codex/flux-config.md` with the inventory so documentation and code reference the same
  table of toggles.
- **Next Steps:** Wire additional Flux-specific runtime flags (e.g., refiner settings,
  LoRA interactions) into the schema as they are implemented and extend automated tests
  to assert the OptionInfo mappings remain in sync with backend expectations.

### Expand GGUF/BnB operator coverage via backend implementations
- **Status:** Blocked
- **Dependencies:** Need profiling data to validate kernels and confirm device
  compatibility (CUDA, ROCm, CPU fallback).
- **Next Steps:** Catalogue missing operations, prioritise by usage frequency, and
  implement with regression tests on representative checkpoints.

### Validate LoRA + Flux workflows and document findings
- **Status:** Blocked
- **Dependencies:** Lacking automated scenarios combining LoRA stacks with Flux
  models; manual validation is time-intensive.
- **Next Steps:** Build scripted workflows leveraging existing automation harnesses,
  capture expected outputs, and document recommended settings in
  `codex/testing-and-tooling.md`.

## 3. UI & Accessibility

### Audit tabs in `modules/ui.py` and migrate shared patterns
- **Status:** Blocked
- **Dependencies:** Requires coordinated update to template fragments in
  `modules_forge/ui/` and localization keys. Needs snapshot of active extensions to
  avoid regressions.
- **Next Steps:** Map duplicate UI fragments, introduce shared components, and verify
  localisation coverage before removal.

### Ensure localisation keys for Forge-only controls
- **Status:** Blocked
- **Dependencies:** Translation files incomplete; automation scripts for consistency
  checks not yet in place.
- **Next Steps:** Generate report of missing keys, triage by locale, and coordinate
  translation updates.

### Improve tablet/canvas responsiveness in `javascript/canvas/`
- **Status:** Blocked
- **Dependencies:** Need performance traces from representative devices; lacking
  profiling harness.
- **Next Steps:** Capture event-loop metrics, identify hotspots, and refactor handlers
  with debounce/throttle strategies where necessary.

## 4. Extension Ecosystem

### Review bundled extensions for backend compatibility
- **Status:** Blocked
- **Dependencies:** Comprehensive compatibility matrix absent; each extension requires
  targeted smoke tests with the new backend interfaces.
- **Next Steps:** Assemble extension inventory, create automated loading scripts, and
  capture failure logs for remediation.

### Provide migration guides for external extensions
- **Status:** Blocked
- **Dependencies:** Depends on findings from the compatibility review; guidance cannot
  be written until breaking changes are catalogued.
- **Next Steps:** Draft guides after completing compatibility analysis, ensuring
  actionable upgrade paths for maintainers.

### Introduce automated smoke tests for extension loading
- **Status:** Blocked
- **Dependencies:** Testing harness not yet defined; requires sandbox environment and
  deterministic extension list.
- **Next Steps:** Implement pytest-based loader harness targeting the top five
  community extensions and integrate into CI.

## 5. Tooling & Developer Experience

### Establish linting presets for Python and JavaScript
- **Status:** Blocked
- **Dependencies:** Need consensus on formatting rules and integration with existing
  CI to avoid conflicts.
- **Next Steps:** Propose ruff/black configuration, align eslint rules with project
  style, and stage incremental rollout with pre-commit hooks.

### Automate VRAM regression tracking via backend hooks
- **Status:** Blocked
- **Dependencies:** Requires instrumentation pipeline and storage for baseline metrics.
- **Next Steps:** Extend `backend/memory_management.py` to expose hook-friendly API,
  capture metrics in CI, and visualise trends.

### Document manual QA matrices in `codex/testing-and-tooling.md`
- **Status:** Blocked
- **Dependencies:** Matrices depend on completion of automation tasks above; coverage
  gaps remain.
- **Next Steps:** Once automation is defined, document manual fallback scenarios and
  sign-off checklist.

## 6. Runtime & Dependency Modernization

### Finalise Python 3.12 + CUDA 12.4 adoption
- **Status:** Blocked
- **Dependencies:** Need cross-platform smoke test results, validation of mediapipe
  preprocessors, and installer updates.
- **Next Steps:** Collect platform-specific reports, update launch scripts, and revise
  docs once parity is confirmed.

### Profile checkpoint/embedding reload latency
- **Status:** Blocked
- **Dependencies:** Benchmark harness pending; upstream weight reload fix requires
  adaptation for Forge hooks.
- **Next Steps:** Build measurement scripts, compare against Issue #3048 baselines, and
  integrate upstream patches after verifying device transfer support.

### Validate GGUF embedding loaders & soft inpainting pipeline on low-core CPUs
- **Status:** Blocked
- **Dependencies:** Need access to low-core hardware or representative simulation and
  new dependencies (`joblib`, `protobuf 4.x`) configured.
- **Next Steps:** Create smoke scripts leveraging those dependencies, gather telemetry,
  and codify regression checks.

### Audit legacy dependency pins blocking Python 3.12+
- **Status:** Blocked
- **Dependencies:** Inventory of preprocessors and historical pin rationale incomplete.
- **Next Steps:** Catalogue dependencies, propose migration/replacement strategy, and
  update requirement files with documented justification.

### Track divergence from ComfyUI and SwarmUI
- **Status:** Blocked
- **Dependencies:** Requires comparative study with up-to-date upstream snapshots.
- **Next Steps:** Review upstream changes, document gaps in
  `codex/comfyui-swarmui-postforge.md`, and raise follow-up issues for high-impact
  deltas.

---

**Outstanding Requirement:** Completing these tasks necessitates multi-sprint effort
and coordination across backend, UI, and documentation teams. The next contributor
should allocate dedicated cycles for each milestone rather than attempting ad-hoc
fixes.

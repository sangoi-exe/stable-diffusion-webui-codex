# Refactor Roadmap – Execution Tracker

This tracker mirrors the roadmap milestones and distils them into actionable work
items. Update the tables as tasks advance. Use statuses `Not Started`,
`In Progress`, `Blocked`, or `Complete`.

## 1. Backend Modernization

### Milestone 1.1 – Txt2Img runtime parity
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Discovery: capture golden txt2img runs (CFG, hires, refiner, LoRA) from `modules/processing.py`. | Not Started | Stable baseline presets; access to benchmark prompts. | Record seeds + configs in `codex/fixtures/txt2img/`. | Needed before regression testing can proceed. |
| Test harness: expand `tests/backend/test_txt2img.py` with deterministic fixtures covering hires/refiner flows. | In Progress | Golden outputs from discovery step. | Commit parametrised tests covering base + hires passes. | Previous attempt lacks reproducible fixtures; replace with golden data once captured. |
| Implementation: port orchestration into `backend/diffusion_engine/txt2img.py`. | Blocked | Deterministic tests; clarified callback contracts. | Update runtime to satisfy fixtures without regressing UI callbacks. | Awaiting reliable tests to validate behaviour. |
| Integration: toggle `modules/processing.py` to backend runtime and remove legacy path post-verification. | Blocked | Implementation complete; manual regression checklist. | Land feature flag + rollout plan documented in release notes. | Hold until runtime parity is proven and stakeholders sign off. |

### Milestone 1.2 – Scheduler registry convergence
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Draft canonical scheduler metadata schema. | Not Started | Inventory of schedulers across modules/extensions. | Document schema proposal in `codex/schedulers.md`. | Schedule review with extension maintainers. |
| Implement registry module in `backend/sampling/registry.py`. | Blocked | Approved schema. | Introduce registry with unit coverage. | Implementation must wait for consensus. |
| Migrate UI/extension hooks to registry. | Blocked | Registry module merged; migration guide drafted. | PR updating `modules/sd_samplers.py` + shims. | Coordinate with extension owners to avoid breaking changes. |
| Validate parity via automated + smoke tests. | Blocked | Migration complete. | CI job comparing sampler outputs and extension smoke scripts. | Relies on extension harness from Milestone 4.3. |

### Milestone 1.3 – Unified memory telemetry
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Define shared telemetry contract (fields, units, cadence). | Not Started | Current UI + backend telemetry review. | Specification added to `codex/backend.md`. | Align with product requirements before implementation. |
| Refactor `backend/memory_management.py` to publish unified payload. | Blocked | Approved contract. | Implementation PR exposing subscription API. | Needs agreement on data shape first. |
| Update `modules/shared.py` + UI toggles to consume publisher. | Blocked | Backend publisher merged. | UI patch removing duplicate polling. | Must coordinate localisation updates. |
| Add regression tests + doc updates. | Blocked | Publisher + UI integration. | Tests covering payload schema + doc updates in `codex/testing-and-tooling.md`. | Execute once runtime path is stable. |

## 2. Flux & Advanced Model Support

### Milestone 2.1 – Flux surface area audit
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Inventory Flux toggles across UI/backends. | Not Started | Access to enabled extensions for audit. | Spreadsheet or doc summarising controls vs. runtime flags. | Foundation for naming alignment. |
| Propose unified naming + config schema. | Blocked | Completed inventory. | Draft schema shared in design review. | Requires buy-in from UX + docs. |
| Update docs with canonical terminology. | Blocked | Approved schema. | PR updating `codex/backend.md` + tutorials. | Should sync with UI labelling changes. |

### Milestone 2.2 – GGUF/BnB operator enablement
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Catalogue missing operators + usage frequency. | Not Started | Telemetry/feedback from model zoo. | Tracker in `codex/gguf-operators.md`. | Prioritise high-impact ops first. |
| Implement/port required kernels. | Blocked | Catalogue prioritised; hardware availability. | Backend patches adding operators with tests. | Coordinate with runtime modernization (Milestone 6). |
| Add regression + performance tests. | Blocked | Implementations merged. | Benchmark scripts committed under `scripts/benchmarks/`. | Should feed VRAM regression automation. |

### Milestone 2.3 – LoRA + Flux validation
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Script automated LoRA+Flux workflows. | Not Started | Access to representative LoRA stacks + checkpoints. | Automation script leveraging existing harness. | Enables repeatable testing. |
| Capture artefacts & acceptance thresholds. | Blocked | Automated workflows in place. | Baseline outputs stored with metrics. | Use to detect regressions. |
| Document recommended settings. | Blocked | Artefacts reviewed + approved. | Guide in `codex/testing-and-tooling.md`. | Publish once validation complete. |

## 3. UI & Accessibility

### Milestone 3.1 – Tab consolidation
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Audit duplicated UI components. | Not Started | Snapshot of default + common extensions enabled. | Audit log enumerating overlaps. | Share with design stakeholders. |
| Extract shared fragments. | Blocked | Audit findings + design approval. | Refactor PR introducing shared helpers. | Watch for localisation coupling. |
| Smoke test + document migration guidance. | Blocked | Refactor merged. | Test report + migration appendix. | Ensure extension authors are notified. |

### Milestone 3.2 – Localisation coverage
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Generate missing Forge-only key report. | In Progress | Localisation tooling. | Report stored under `codex/localisation-gap.md`. | Initial script drafted; needs validation. |
| Coordinate translation updates/community outreach. | Blocked | Finalised gap report. | Issue tracker entries per locale. | Engage community translators. |
| Establish CI/manual checks to prevent regressions. | Blocked | Translation updates scheduled. | Documented workflow + optional CI job. | Ideally ties into linting baseline (Milestone 5.1). |

### Milestone 3.3 – Canvas responsiveness
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Profile tablet interactions in `javascript/canvas/`. | Not Started | Access to profiling hardware or telemetry. | Performance report with bottleneck list. | Use to scope refactors. |
| Apply targeted performance fixes. | Blocked | Profiling data + prioritised hotspots. | PR implementing throttling/debouncing improvements. | Validate no regression to stylus accuracy. |
| Validate improvements + update changelog. | Blocked | Fixes merged. | Metrics comparison + NEWS/README updates. | Capture before/after data for posterity. |

## 4. Extension Ecosystem

### Milestone 4.1 – Bundled extension audit
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Inventory bundled extensions + backend touchpoints. | Not Started | Extension manifest export. | Tracker in `codex/extensions-audit.md`. | Include maintenance status. |
| Execute compatibility smoke runs. | Blocked | Inventory complete; harness ready. | Report summarising pass/fail + logs. | Feed remediation backlog. |
| File remediation tasks with root-cause notes. | Blocked | Smoke report. | Issue list grouped by subsystem. | Prioritise by severity + usage. |

### Milestone 4.2 – External migration guides
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Summarise backend API changes impacting extensions. | Not Started | Inputs from Milestones 1.2 & 4.1. | Draft guide outline. | Keep examples concrete. |
| Author migration playbooks + testing checklist. | Blocked | Guide outline approved. | Completed doc in `codex/extensions-and-integrations.md`. | Include version mapping table. |
| Publish + announce to maintainers. | Blocked | Playbooks merged. | Announcement post + PR summary. | Coordinate with community managers. |

### Milestone 4.3 – Automated extension smoke tests
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Design pytest harness for extension loading. | Not Started | Agreement on extension priority list. | RFC covering harness design + required fixtures. | Align with CI constraints. |
| Implement deterministic environment + CI job. | Blocked | Harness design approved. | CI pipeline executing smoke suite. | Requires lightweight model assets. |
| Track metrics + gate merges. | Blocked | CI job operational. | Dashboard/reporting integrated into review checklist. | Enforce once failure rate stabilises. |

## 5. Tooling & Developer Experience

### Milestone 5.1 – Linting baseline
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Draft ruff/black + eslint/prettier configs. | Not Started | Style survey of existing code. | Config proposal PR. | Limit churn by sampling representative files. |
| Trial configs on sample modules. | Blocked | Draft configs. | Pilot branch with formatted subset + diff analysis. | Confirm appetite for automated formatting. |
| Roll out via hooks + documentation. | Blocked | Pilot approved. | Update contribution docs + add pre-commit. | Communicate freeze window before mass reformat. |

### Milestone 5.2 – VRAM regression automation
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Extend backend hooks to capture allocation metrics. | Not Started | Agreement on telemetry contract (Milestone 1.3). | Backend patch emitting metrics per job. | Should reuse unified telemetry payload. |
| Build reporting pipeline/CI job. | Blocked | Metrics hook merged. | Script + CI integration storing baselines. | Align with release cadence. |
| Surface trends in dashboards/release notes. | Blocked | Pipeline active. | Dashboard snapshots + release template updates. | Provide actionable alerts to maintainers. |

### Milestone 5.3 – Manual QA matrices
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Enumerate manual validation scenarios. | Not Started | Input from QA + support. | Draft matrix in `codex/testing-and-tooling.md`. | Prioritise high-risk workflows. |
| Capture environment/setup + expected outcomes. | Blocked | Matrix skeleton. | Detailed checklist with pass/fail criteria. | Keep instructions reproducible. |
| Integrate into release checklist. | Blocked | Detailed checklist approved. | Update release template + onboarding docs. | Ensure product owners sign off. |

## 6. Runtime & Dependency Modernization

### Milestone 6.1 – Python 3.12 + CUDA 12.4 adoption
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Compile platform compatibility matrix. | Not Started | Installer telemetry; community feedback. | Matrix stored in `codex/runtime-matrix.md`. | Identify blockers per OS. |
| Update launch scripts/installers. | Blocked | Compatibility matrix + resolved blockers. | PR updating `launch.py`, installers, docs. | Include rollback strategy. |
| Run smoke tests + document parity. | Blocked | Updated launchers. | Test log + sign-off summary. | Cover mediapipe preprocessors explicitly. |

### Milestone 6.2 – Checkpoint/embedding reload profiling
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Build reload benchmarking script. | Not Started | Agreement on metrics + scenarios. | Script under `scripts/benchmarks/reload.py`. | Baseline to compare optimisations. |
| Integrate upstream optimisations. | Blocked | Benchmark script + profiling data. | Backend patch applying upstream fixes. | Verify compatibility with Forge hooks. |
| Record before/after metrics + alerts. | Blocked | Optimisations merged. | Report appended to `codex/testing-and-tooling.md`. | Feed into VRAM regression automation. |

### Milestone 6.3 – Low-core CPU validation
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Provision/simulate low-core CPU environments. | Not Started | Hardware access or virtualization budget. | Documented plan for test rigs. | Could leverage CI containers with limited cores. |
| Validate GGUF embeddings + soft inpainting. | Blocked | Test environment ready. | Regression scripts + results stored in repo. | Use to identify missing dependencies. |
| Document findings + remediation backlog. | Blocked | Validation results. | Summary doc + actionable backlog items. | Share with runtime owners. |

### Milestone 6.4 – Dependency audit & upstream parity
| Task | Status | Dependencies | Next Deliverable | Notes |
| --- | --- | --- | --- | --- |
| Catalogue legacy dependency pins + rationale. | Not Started | Access to historical release notes/issues. | Audit table in `codex/dependency-audit.md`. | Include reason each pin exists. |
| Propose upgrades/replacements. | Blocked | Catalogue complete; compatibility assessments. | RFC detailing proposed changes + risk mitigation. | Review with release engineering. |
| Track divergence vs. ComfyUI/SwarmUI. | In Progress | Latest upstream snapshots. | Update `codex/comfyui-swarmui-postforge.md` with gap analysis. | Current diff needs refresh; schedule monthly updates. |

---

**Outstanding Requirement:** Each milestone demands dedicated ownership. Assign
owners per milestone before next sprint planning to prevent the backlog from
stalling. Update this tracker at the end of every work session.

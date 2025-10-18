# Task: Record manual-only testing directive and error logging requirements
Date: 2025-10-18

Summary
- Updated implementation and rollout plans to forbid automated suites (`pytest`, smoke scripts) and require manual testing walkthroughs with captured logs.
- Added guidance that errors must remain fully verbose (task, engine, modelo, preset, options, stack trace) for manual debugging.

Files/Paths
- `.sangoi/one-trainer/implementation-plan.md`
- `.sangoi/one-trainer/architecture/rollout-architecture.md`
- `.sangoi/CHANGELOG.md`

Notes
- All future PRs must include manual validation evidence (commands, outputs, screenshots/logs) instead of automated tests.

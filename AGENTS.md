# Guidance for Contributors

This repository powers **Stable Diffusion WebUI Forge**. Please follow these principles while working anywhere in this repo:

## Engineering Principles
- Pursue robustness before optimizations; never sacrifice existing functionality just to silence errors.
- Prefer proven, minimal solutions over custom abstractions. Fix root causes instead of applying quick hacks.
- Preserve descriptive naming. Only rename identifiers when it is strictly necessary for clarity or correctness.
- In Python, include explicit progress reporting (e.g., progress bars) when long-running operations are introduced or reworked.
- Embrace thorough error handling instead of permissive fallbacks. Do not hide or ignore exceptions.
- Keep changes cohesive. Avoid mixing refactors with unrelated feature work in a single commit.

## Collaboration Workflow
- Mirror user-facing behaviour in refactors; ensure feature parity after changes.
- Update or add documentation when behaviour or configuration surfaces change.
- Prefer deterministic, automated checks before merging. Document any manual validation that substitutes automated testing.
- When touching performance-sensitive code (sampling, loaders, memory management), profile before and after when feasible.

## Coding Style
- Follow the surrounding style of each file. Align imports, spacing, and logging conventions with existing patterns.
- Do not wrap imports in try/except blocks. Handle optional dependencies explicitly near their usage sites.
- Default to descriptive logging over silent failures. When adding logs, keep them actionable and concise.

When in doubt, slow down—quality and maintainability take precedence over speed.

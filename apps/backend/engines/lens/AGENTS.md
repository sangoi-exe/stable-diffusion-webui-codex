# apps/backend/engines/lens Overview
<!-- tags: backend, engines, lens -->
Date: 2026-05-24
Last Review: 2026-05-24
Status: Skeleton

## Purpose
- Host the parked Lens engine facade for the single public `lens` architecture family.
- Validate Diffusers-style Lens folder metadata without constructing upstream pipelines or loading tensor payloads.

## Key Files
- `apps/backend/engines/lens/lens.py` — manually-registerable `LensEngine` facade; metadata-load skeleton and `sample_lens_txt2img(...)` not-implemented hook.
- `apps/backend/engines/lens/factory.py` — JSON/index-only folder validator with `metadata_only` and `runtime_ready` modes.
- `apps/backend/engines/lens/spec.py` — engine load-spec assembly around the factory validator.
- `apps/backend/engines/lens/__init__.py` — package marker only.

## Notes
- `lens` is not default-registered in this tranche. Use `register_lens(...)` only for isolated/manual skeleton validation.
- `/api/engines/capabilities` must expose `lens` only through `parked_exact_engines`, not as a runnable engine row.
- Runtime txt2img remains absent; API requests must fail before task creation while this engine is parked.

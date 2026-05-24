# apps/backend/runtime/families/lens Overview
<!-- tags: backend, runtime, families, lens -->
Date: 2026-05-24
Last Review: 2026-05-24
Status: Skeleton

## Purpose
- Own the Lens architecture-family runtime contracts for the single public `lens` engine id.
- Keep `default`, `turbo`, and `base` as internal variants selected through `extras.lens.variant`; do not create `lens_turbo` or `lens_base` engines.

## Structure
- `config.py` — variant specs, defaults, architecture constants, official resolution bucket table, and not-implemented messages.
- `resolution.py` — official bucket validation and Lens image-sequence geometry helpers.
- `scheduler.py` — Lens FlowMatch scheduler metadata validation, empirical `mu`, sigma ladder, and transformer timestep scaling helpers.
- `sampler.py` — reserved Lens txt2img denoising owner; skeleton raises `NotImplementedError` until runtime lands.
- `__init__.py` — lightweight public contract exports only.

## Notes
- This folder is skeleton-only in tranche 1. It must not import upstream `lens`, construct Diffusers pipelines, load tensor payloads, or claim generation support.
- Native checkpoint keys stay native. Any future keyspace bridge belongs in `apps/backend/runtime/state_dict/**`, not in prefix strippers or eager remap helpers here.
- The reasoner is hard-deferred. Do not add prompt rewrite config, API-key state, or dormant reasoner settings under this family.

# AGENT — Model Parser
<!-- tags: runtime, model-parser -->
Status: Active
Last Review: 2026-07-30

## Mandate
- Parse checkpoint state dicts without `huggingface_guess`.
- Split, convert, and validate components using registry `ModelSignature` metadata.
- Produce structured `CodexEstimatedConfig` objects for loaders and adapters.

## Key files
- `__init__.py` — public parser entrypoint.
- `builders.py` — component registration and estimated-config assembly.
- `plan.py` — execution engine for declarative parser plans.
- `quantization.py` — quantization detection/validation helpers.
- `converters/` — shared component converters.
- `families/` — family-specific parser planners.

## Expectations
- 2026-03-20: `CodexEstimatedConfig` no longer carries an `inpaint_model()` heuristic; request/use-case ownership decides img2img masking behavior, and parser output must stay descriptive rather than pipeline-authoring.
- Keep GGUF plans aligned with canonical keyspace resolvers in `apps/backend/runtime/state_dict/**`.
- `quantization.py` must detect GGUF/NF4/FP4 and fail loud on unsupported packed artifacts.
- When parser modules change, run `uv run python -m py_compile ...` for the touched parser files and record manual validation steps.
- 2026-03-23: `families/ltx2.py` is part of the live advertised LTX2 `txt2vid` / `img2vid` slice. It strips `model.diffusion_model.` into a temporary `dit_root`, validates connector groups fail-loud across both direct aliases and wrapped `connectors.` surfaces, preserves core `transformer_blocks.*` tensors in the `transformer` component, and emits explicit `transformer` + `connectors` components plus `vae`, `audio_vae`, and `vocoder`. Standalone `transformer_1d_blocks.*` remains supported as connector evidence but is not treated as a mandatory invariant.
- 2026-07-30: `families/qwen_image.py` is the Edit-2511 core-only GGUF planner. It consumes the already-resolved exact native transformer view, emits one `transformer` component, and performs no key conversion.

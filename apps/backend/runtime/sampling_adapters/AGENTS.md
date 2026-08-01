# apps/backend/runtime/sampling_adapters Overview
<!-- tags: runtime, sampling, adapters, prediction -->
Date: 2025-11-10
Last Review: 2026-08-01
Status: Active

## Purpose
- Thin sampling adapter wrappers and shared planning/math helpers used by samplers/patchers (native-only; no external sampler deps).

## Notes
- Keep this folder small and focused; consider consolidating these adapters into the sampling stack when call sites are fully native.
- 2025-11-28: `prediction_from_diffusers_scheduler` preserves scheduler `sigma_data`/`prediction_type` when constructing `Prediction`, avoiding silent scaling drift for v-pred SDXL variants.
- 2026-01-02: Added standardized file header docstrings to adapter modules (doc-only change; part of rollout).
- 2026-02-08: Flow predictors now expose explicit SIMPLE schedule mode constants (`SIMPLE_SCHEDULE_MODE_TAIL_DOWNSAMPLE_SIGMAS`, `SIMPLE_SCHEDULE_MODE_FLOWMATCH_SHIFTED_LINSPACE`) and validate `simple_schedule_mode` against `FLOW_SIMPLE_SCHEDULE_MODES` (fail loud on unknown values).
- 2026-03-08: `extra.py` now exposes a pure `build_restart_step_plan(...)` seam shared by the legacy `restart_sampler(...)` helper and the native `driver.py` restart lane; deterministic runtime renoise stays owned by the driver via `ImageRNG`.
- 2026-08-01: `SamplerModel.apply_model(...)` forwards native denoiser output directly into unchanged predictor math; `memory_required(...)` retains its existing numerical FP32 floor for predictor-result pressure.

## Invariants (SamplerModel.apply_model)
- `c_crossattn` deve ser Tensor 3D (B,S,C); erro explícito se inválido.
- Quando `codex_config.context_dim` estiver presente:
  - Se int → `C` deve ser exatamente igual a este valor.
  - Se sequência → `C` deve pertencer ao conjunto informado.
- Se `diffusion_model.num_classes` não é `None`, `y` deve estar presente (Tensor 2D) — sem fallback.
- Se `adm_in_channels` (em `codex_config`) estiver definido (>0), `y.shape[1]` deve ser igual a este valor.

## Logging
- Logger `backend.runtime.sampling_adapters.sampler_model` em nível DEBUG registra shapes de `x`, `t`, `context` e `y` antes do forward do UNet.
- 2025-12-12: `SamplerModel.apply_model` deep logs are owned by the shared sampler-model seam; use `CODEX_SAMPLER_MODEL_DEBUG=1` or `CODEX_SAMPLER_MODEL_DEBUG_APPLY_MODEL=1` for tensor stats + forwarded extra cond keys in callers such as Z Image.

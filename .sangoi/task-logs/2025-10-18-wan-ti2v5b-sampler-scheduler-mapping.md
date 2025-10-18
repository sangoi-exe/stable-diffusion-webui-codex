Task: Map sampler/scheduler to Diffusers and limit per engine (WAN TI2V‑5B)
Date: 2025-10-18

Changes
- Added `backend/engines/video/wan/schedulers.py`:
  - `allowed_samplers_for_engine(engine)` returns WAN subset: Euler a, Euler, DDIM, DPM++ 2M, DPM++ 2M SDE, PLMS.
  - `apply_sampler_scheduler(pipe, sampler, scheduler)` swaps `pipe.scheduler` via Diffusers classes and flags (`use_karras_sigmas`, `timestep_spacing='trailing'`, `use_exponential_sigmas`) with warnings.
- Engine uses mapping and annotates `info` with `sampler_in/scheduler_in` e `sampler_effective/scheduler_effective`.

Rationale
- Enforce compatibility and predictability per engine; prevent silent mismatches.

Next
- Validate per-resolution behavior and defaults; extend mapping to Hunyuan/SVD após WAN.


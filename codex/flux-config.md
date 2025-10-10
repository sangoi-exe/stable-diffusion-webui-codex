# Flux Configuration Inventory

This document inventories every Flux-specific toggle surfaced in the UI and maps
it to the backend schema introduced in `backend/diffusion_engine/flux_config.py`.
Use it as the canonical reference when auditing presets, wiring new controls, or
validating extension compatibility.

## Schema Overview

| Key | Label | Scope | Control | Default | Notes |
| --- | ----- | ----- | ------- | ------- | ----- |
| `flux_t2i_width` | txt2img width | txt2img | Slider (64–2048, step 8) | `896` | Base canvas width for Flux txt2img runs. |
| `flux_t2i_height` | txt2img height | txt2img | Slider (64–2048, step 8) | `1152` | Base canvas height for Flux txt2img runs. |
| `flux_t2i_cfg` | txt2img CFG | txt2img | Slider (1–30, step 0.1) | `1.0` | Default classifier-free guidance. |
| `flux_t2i_hr_cfg` | txt2img HiRes CFG | txt2img | Slider (1–30, step 0.1) | `1.0` | CFG applied during hires fix. |
| `flux_t2i_d_cfg` | txt2img Distilled CFG | txt2img | Slider (0–30, step 0.1) | `3.5` | CFG for distilled checkpoints. |
| `flux_t2i_hr_d_cfg` | txt2img Distilled HiRes CFG | txt2img | Slider (0–30, step 0.1) | `3.5` | Distilled hires CFG override. |
| `flux_i2i_width` | img2img width | img2img | Slider (64–2048, step 8) | `1024` | Base width when img2img preset is Flux. |
| `flux_i2i_height` | img2img height | img2img | Slider (64–2048, step 8) | `1024` | Base height for img2img Flux preset. |
| `flux_i2i_cfg` | img2img CFG | img2img | Slider (1–30, step 0.1) | `1.0` | Default CFG for img2img. |
| `flux_i2i_d_cfg` | img2img Distilled CFG | img2img | Slider (0–30, step 0.1) | `3.5` | Distilled CFG for img2img. |
| `flux_GPU_MB` | GPU Weights (MB) | shared | Slider (0–`total_vram`, step 1) | `total_vram - 1024` | Leaves 1 GB headroom for activations and refiner swaps. |
| `flux_t2i_sampler` | txt2img sampler | txt2img | Dropdown (`visible_samplers()`) | `Euler` | Choices populated from sampler registry. |
| `flux_t2i_scheduler` | txt2img scheduler | txt2img | Dropdown (`sd_schedulers.schedulers`) | `Simple` | Uses scheduler labels exposed to UI. |
| `flux_i2i_sampler` | img2img sampler | img2img | Dropdown (`visible_samplers()`) | `Euler` | Shares sampler list with txt2img. |
| `flux_i2i_scheduler` | img2img scheduler | img2img | Dropdown (`sd_schedulers.schedulers`) | `Simple` | Shares scheduler list with txt2img. |

## Runtime Touchpoints

- **UI presets:** `modules_forge/main_entry.py` invokes `build_flux_option_info` to
  hydrate slider defaults (including GPU VRAM budgets) when the Flux preset is
  selected.
- **Sampler script:** `modules/processing_scripts/sampler.py` relies on the same
  helper to register dropdown defaults using the live sampler/scheduler lists.
- **JavaScript filters:** `javascript/extraNetworks.js` uses `opts.forge_preset`
  to filter cards by model family, ensuring Flux-only assets stay scoped.

## Extension Guidance

- Add new Flux controls by extending `FLUX_FIELD_SPECS`. The helper will expose
  them automatically once wired into the relevant UI modules.
- Keep dropdown choices dynamic. Pass runtime lists through the helper's
  `dynamic_choices` parameter so extensions injecting custom samplers inherit the
  new defaults.
- When capturing presets or exporting UI snapshots, reference this inventory to
  ensure metadata round-trips (e.g., PNG info blocks) include every Flux field.

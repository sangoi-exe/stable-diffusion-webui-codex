Refactor Status – 2025-10-11
============================

1) Txt2Img Backend Runtime
- Status: In progress (near‑complete for base pass)
- Completed
  - RNG initialization in runtime (latent shape aware)
  - Forge object restore (original and LoRA‑applied) + token merging
  - Script hooks: before_process_batch, process_batch, post_sample
  - Extra network activation; job/progress updates via shared.state
- Outstanding
  - LoRA registry initialization order for all entrypoints (UI vs. script)
  - Hires/refiner orchestration (conditioning, checkpoint hot‑swap, timing)
  - Noise‑schedule overrides and multi‑iter seed bookkeeping

2) Logging
- Status: Adopted
- Completed: Rich/Colorama console, tqdm‑aware handler, fallback to std logging
- Follow‑ups: Replace scattered prints with logger calls; doc env toggles

3) Dependencies
- Status: Adopted
- Completed: requirements updater script; drop‑excluded torch family; resolved open‑clip/protobuf conflict via pin bump

Next Steps
- Harden LoRA registry bridging and remove temporary diagnostics
- Port hires/refiner path; add targeted tests or baseline runs to lock behaviour
- Sweep modules for logger usage and update docs


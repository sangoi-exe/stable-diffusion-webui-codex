# apps/backend/video/upscaling Overview
Date: 2026-02-27
Last Review: 2026-08-31
Status: Active

## Purpose
- Own the fail-loud SeedVR2 child runner for dedicated video-upscale tasks.

## Key Files
- `seedvr2.py` — SeedVR2 child runner (exact upstream geometry/padding, phase-specific host/CUDA/MPS live-set admission including MPS Phase 4 source reconstruction, MPS unified-memory admission, source video -> upstream CLI -> numerically validated PNG output paths).
- `__init__.py` — Package marker (no facade exports).

## Notes
- Do not import code from `.refs/**` inside `apps/**`.
- Runner executes the checked-out upstream CLI (`inference_cli.py`) in a task-scoped child process. It must fail loud on checkout/bootstrap errors, missing Python dependencies, model download failures, GPU admission failure, child execution failure, malformed/duplicate/gapped frame identities, and frame count or size mismatches.
- `streaming=true` selects bounded streaming immediately. `smart_fallback=true` first admits whole-video accelerator execution from measured CUDA/MPS capacity and, on MPS, the shared physical-memory budget; it then selects one bounded streaming route on capacity rejection or direct-child OOM. If cancellation becomes active after a direct child has already exited OOM, the runner preserves that typed OOM and does not start streaming. The runner has no CPU fallback.
- An immediate task cancellation callback terminates the complete child process group, including descendants that outlive the leader. The runner does not materialize source or output video tensors in the backend process.
- 2026-08-31: Runner metadata records the selected direct or streaming plan, measured accelerator-memory budget, MPS unified-memory requirement, concurrent source/target/latent/output per-frame estimates, exact source/target/padded geometry, uniform- and `4n+1`-padding construction peaks, validated numeric output dimensions, and any one smart-fallback selection or direct-child OOM retry. Host admission includes the complete float32 result, float32 PNG multiplication temporary, uint8 PNG array, and source-resolution temporal-padding temporaries. MPS admission includes complete input, retained latent collections, final output, color-correction-aware Phase 4 source reconstruction while complete input/output remain resident, and shared host/device phase peaks. Child diagnostics retain a bounded tail plus cumulative OOM evidence, reach EOF before classification, and keep the inference gate occupied until the process group is physically cleaned.
- Runtime directories:
  - Repo: `CODEX_SEEDVR2_REPO_DIR` (default `CODEX_ROOT/.uv/xdg-data/seedvr2/repo`)
    - Default-path bootstrap only: clone from `CODEX_SEEDVR2_REPO_URL` (default `https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git`) when missing, then always enforce `CODEX_SEEDVR2_REPO_REF` (default `4490bd1f482e026674543386bb2a4d176da245b9`) on each run (checkout; fetch+checkout fallback when ref is not local).
    - Default-path bootstrap/ref enforcement is serialized with an interprocess lock file under `CODEX_ROOT/.uv/xdg-data/seedvr2/`.
  - Model cache: `CODEX_SEEDVR2_MODEL_DIR` (default `CODEX_ROOT/.uv/xdg-data/seedvr2`)
  - CUDA override: `CODEX_SEEDVR2_CUDA_DEVICE` sets the explicit visible CUDA device index used by SeedVR2 and has priority over component device-derived mapping.
  - Task output work dir: caller-owned `~/.cache/codex/seedvr2-dedicated-video-upscale/run-<id>/...`

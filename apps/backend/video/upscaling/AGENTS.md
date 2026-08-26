# apps/backend/video/upscaling Overview
Date: 2026-02-27
Last Review: 2026-08-26
Status: Active

## Purpose
- Own the fail-loud SeedVR2 frame runner for dedicated video-upscale tasks.

## Key Files
- `seedvr2.py` — SeedVR2 native in-process runner (PIL frames -> tensor batch -> SeedVR2 runtime module -> validated output frames).
- `__init__.py` — Package marker (no facade exports).

## Notes
- Do not import code from `.refs/**` inside `apps/**`.
- Runner executes SeedVR2 in-process via the checked-out runtime module (`inference_cli.py`) and must fail loud on missing runtime symbols, missing Python dependencies, model download failures, runtime execution failures, and frame count/size mismatches.
- 2026-03-01: SeedVR2 runner metadata/logging now surfaces `attention_mode` + `sdpa_flash_runtime` when available; this explicitly disambiguates PyTorch SDPA flash-kernel runtime from SeedVR2's own Flash-Attention package availability check output.
- Runtime directories:
  - Repo: `CODEX_SEEDVR2_REPO_DIR` (default `CODEX_ROOT/.uv/xdg-data/seedvr2/repo`)
    - Default-path bootstrap only: clone from `CODEX_SEEDVR2_REPO_URL` (default `https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git`) when missing, then always enforce `CODEX_SEEDVR2_REPO_REF` (default `4490bd1f482e026674543386bb2a4d176da245b9`) on each run (checkout; fetch+checkout fallback when ref is not local).
    - Default-path bootstrap/ref enforcement is serialized with an interprocess lock file under `CODEX_ROOT/.uv/xdg-data/seedvr2/`.
  - Model cache: `CODEX_SEEDVR2_MODEL_DIR` (default `CODEX_ROOT/.uv/xdg-data/seedvr2`)
  - CUDA override: `CODEX_SEEDVR2_CUDA_DEVICE` sets the explicit visible CUDA device index used by SeedVR2 and has priority over component device-derived mapping.
  - Work dir: `CODEX_ROOT/.tmp/seedvr2/<run_id>/...`

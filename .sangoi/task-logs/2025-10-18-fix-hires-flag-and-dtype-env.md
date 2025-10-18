Date: 2025-10-18
Task: Fix hires enable flag and add dtype env overrides

Fixes
- builders: `StableDiffusionProcessingTxt2Img` no longer treats a non-empty `highres_fix` dict as `enable_hr=True`. It now respects `highres_fix.enable` explicitly. Prevents unintended HR with scale=1 and steps=0.
- args: add environment overrides to force dtypes without CLI flags:
  - `CODEX_UNET_DTYPE=bf16|fp16|fp32|fp8_e4m3fn|fp8_e5m2`
  - `CODEX_VAE_DTYPE=bf16|fp16|fp32`
  - `CODEX_VAE_IN_CPU=1` and `CODEX_ALL_IN_FP32=1` supported.

Files
- backend/engines/util/adapters.py
- backend/args.py
- webui.settings.bat.example (documented examples)

Notes
- Default heuristics still apply when env vars are unset. Env takes priority.

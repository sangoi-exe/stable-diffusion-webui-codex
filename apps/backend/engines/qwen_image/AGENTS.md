# apps/backend/engines/qwen_image
Date: 2026-05-17
Last Review: 2026-07-31
Status: Active

## Purpose
- Host the Qwen Image Edit-2511 engine facade for the single `qwen_image` architecture family.
- Keep the public task set img2img-only and the internal asset identity fixed to `edit_2511`.

## Key Files
- `apps/backend/engines/qwen_image/qwen_image.py` — engine registration target; validates mandatory blocked Core streaming plus the exact core-only transformer/TEnc/VAE bundle, owns patcher-backed native runtime assembly, and exposes the loaded family runtime to the canonical img2img use-case.
- `apps/backend/engines/qwen_image/__init__.py` — package marker only.

## Notes
- Do not introduce checkpoint-generation-specific Qwen engine ids, aliases, path roots, or families.
- Qwen Image execution requires one Qwen2.5-VL-7B text encoder and one Qwen Image VAE selected through canonical sha-backed asset contracts; do not reuse Anima Qwen3-0.6B or WanVAE contracts.
- The engine delegates mode orchestration to canonical `run_img2img`; Qwen tensor math and component lifecycle remain under `runtime/families/qwen_image`.
- `load()` rejects disabled/missing Core streaming and non-blocked swap modes before transformer, text-encoder, or VAE assembly begins.
- `qwen_image_runtime` is available only after `load()` and fails loud through the shared runtime lifecycle guard otherwise.

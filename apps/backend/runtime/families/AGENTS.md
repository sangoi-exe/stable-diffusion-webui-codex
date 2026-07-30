<!-- tags: backend, runtime, families, layout -->

# apps/backend/runtime/families Overview
Date: 2026-01-17
Last Review: 2026-05-24
Status: Active

## Purpose
- Host model-family runtime code (WAN22/Flux/SD/ZImage/Z-Image L2P/Chroma/LTX2/Netflix VOID/Qwen Image/Lens) under a single `families/` root so `apps/backend/runtime/` stays reserved for generic, cross-family runtime modules (models/loaders, ops, sampling, memory, vision, etc.).

## Structure
- `wan22/`, `flux/`, `sd/`, `zimage/`, `zimage_l2p/`, `chroma/` — family runtimes (implementation-specific).
- `zimage_l2p/` — public `zhen-nan/L2P` pixel-space no-VAE runtime owner; separate from latent Z-Image Turbo/Base.
- `ltx2/` — native-only LTX2 runtime seam (video + audio + external Gemma3 asset contract) with family-owned model, scheduler, and execution code under `apps/**`.
- `netflix_void/` — native-only Netflix VOID vid2vid scaffold (explicit base bundle + literal Pass 1/Pass 2 overlays; execution still fail-loud until the repo-owned runtime port lands).
- `qwen_image/` — Qwen Image Edit-2511 native single-image img2img runtime contracts.
- `lens/` — Microsoft Lens architecture-family skeleton contracts for internal `default|turbo|base` variants, official buckets, schedule helpers, and the fail-loud sampler seam; real runtime is not implemented.

## Notes
- Avoid facade-first imports; prefer importing the defining module within each family runtime.
- See the plan: `.sangoi/plans/2026-01-17-backend-runtime-families-layout.md`.
- 2026-05-24: Lens skeleton is contract-only. Do not import upstream `lens`, construct `LensPipeline`, load tensor payloads, or add reasoner/prompt-rewrite state in `runtime/families/lens` until a separate runtime plan lands.

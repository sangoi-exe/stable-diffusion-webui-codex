# apps/backend/engines Overview
<!-- tags: backend, engines, registry, lazy-imports -->
Date: 2025-12-05
Last Review: 2026-05-24
Status: Active

## Purpose
- Implements model-specific execution logic (Diffusion engines, WAN22, Flux, FLUX.2, SD, Chroma, Qwen Image, Lens, Netflix VOID), plus shared engine utilities and registration.

## Subdirectories
- `common/` — Shared base classes/helpers used by multiple engines.
- `sd/`, `flux/`, `flux2/`, `ltx2/`, `netflix_void/`, `wan22/`, `zimage/`, `anima/`, `qwen_image/`, `lens/` — Model-specific engine implementations and components (Flux family includes Chroma + Kontext variants; `flux2/` is the separate Klein 4B/base-4B seam; `qwen_image/` is the single Qwen Image architecture family facade; `lens/` is a parked metadata-validation skeleton).
- `util/` — Utility helpers for schedulers, attention mapping, etc.

## Key Files
- `__init__.py` — Exposes engine registration hooks and lazy engine-class exports (import-light public surface for external consumers).
- `registration.py` — Canonical registry of available engines.

## Notes
- New engines should live under their own subdirectory (mirroring existing patterns) and register via `registration.py`.
- Task orchestration lives under `apps/backend/use_cases/`; keep engines model-specific and keep shared logic under `apps/backend/runtime/`.
- Factory standard: for each model family, prefer a `spec.py` + `factory.py` seam under `apps/backend/engines/<family>/` so assembly stays consistent and engines keep only model-specific logic (plan: `.sangoi/plans/2026-01-03-engine-factory-standard-v1.md`).
- Diffusion engines now share a `BaseInferenceEngine` lifecycle; instantiate via the registry and let `load()` pull `DiffusionModelBundle` instances instead of invoking loaders manually.
- 2025-11-03: SDXL engine reads `debug_conditioning` from backend config (no direct env lookup) to log conditioning norms when requested.
- 2025-11-30: `apps.backend.engines.__init__` now lazily resolves WAN22 engine classes; importing the package no longer pulls Hugging Face assets or torch unless the engines are requested.
- 2025-12-05: `common.base.BaseInferenceEngine.load()` now accepts an optional `text_encoder_override` option (family + `<family>/<path>` label from paths.json [+ optional components]) and forwards it to `runtime.models.resolve_diffusion_bundle`, so text encoder overrides are applied centrally by the loader instead of por-engine shims.
- 2026-01-02: Added standardized file header docstrings to engine facade/registration modules (doc-only change; part of rollout).
- 2026-01-04: Flux.1-family engine keys are `flux1` / `flux1_kontext` / `flux1_chroma` (no legacy aliases); FLUX.2 now uses the separate `flux2` engine key.
- 2026-01-06: `common.base` VAE overrides (`vae_path`) now unwrap wrapper VAEs via `first_stage_model` before applying state dicts.
- 2026-01-18: `register_default_engines(...)` now registers `flux1_chroma` alongside `flux1`/`flux1_kontext` so the canonical Chroma engine key is available to API callers without manual registration.
- 2026-04-05: `netflix_void` remains a dedicated engine package, but it is now an explicit parked placeholder: `register_default_engines(...)` no longer registers it, `registration.register_netflix_void(...)` raises `NotImplementedError`, and the engine class itself stays as a fail-loud stub until the native vid2vid runtime is implemented.
- 2026-05-17: `register_default_engines(...)` treats grouped helpers as atomic default-registration units. The SDXL base/refiner group must be all present or all absent before default registration; partial caller-provided registries fail with `EngineRegistrationError` instead of silently leaving a missing sibling.
- 2026-01-31: Image-mode wrappers are now fully owned by `apps/backend/use_cases/` (txt2img + img2img); engines delegate via `CodexDiffusionEngine`. The common base also provides default first-stage VAE encode/decode for image engines (WAN/video keep explicit overrides).
- 2026-01-31: Engine-common helpers expanded to reduce drift:
  - Generic conditioning cache helpers (overrideable per call) + shared tensor move helpers for CPU↔device caching.
  - Runtime guard helper (`require_runtime`) for consistent “call load() first” errors across engines.
- 2026-02-05: Added the `anima` engine key.
- 2026-07-30: `qwen_image` is an Edit-2511-only img2img engine. Its live asset set is one exact core transformer GGUF, one external Qwen2.5-VL GGUF, and one external Qwen VAE SafeTensors file; txt2img is not a supported task.
- 2026-05-24: Added the `lens/` parked skeleton facade and `registration.register_lens(...)` for isolated/manual validation only. `lens` is not default-registered; capability truth stays under `parked_exact_engines` until real Lens txt2img runtime lands.
- 2026-02-08: Engine adapters now map swap-model pointer semantics using `switch_at_step` (`RefinerConfig.swap_at_step`) for both global and hires nested refiner config.
- 2026-02-28: `AnimaEngine` is an implemented runtime-backed engine (`apps/backend/engines/anima/anima.py`) and is registered by default (`registration.register_anima` / `register_default_engines`); it is no longer a stub facade.
- 2026-03-06: Added the dedicated `flux2/` engine package plus default registration key `flux2`. The FLUX.2 seam is separate from Flux.1/Chroma/Kontext, uses one Qwen3-4B text encoder, owns its AutoencoderKLFlux2 latent encode/decode contract locally, and now exposes txt2img plus dedicated image-conditioned img2img on the registered engine path: masked img2img/inpaint remains active, partial-denoise img2img is wired, unmasked hires img2img is wired, and masked hires still fails loud.
- 2026-03-11: Added the dedicated `ltx2/` engine package plus default registration key `ltx2`. `Ltx2Engine(BaseVideoEngine)` carries the loader-produced typed LTX2 bundle contract, advertises `txt2vid` / `img2vid`, and assembles the native-only runtime through the family-owned LTX2 seam; canonical video use-cases still own telemetry/postprocess/export and consume `Ltx2RunResult`.
- 2026-02-17: WAN22 canonical registration now maps `wan22_14b` to a dedicated GGUF 14B lane (`Wan2214BEngine`) with no inheritance from `wan22_5b`, preventing 14B dispatch from collapsing into 5B behavior.
- 2026-02-20: WAN22 14B lane is now single-key only (`wan22_14b`); explicit `wan22_14b_native` registration was removed.
- 2026-02-20: WAN22 animate lane key renamed to `wan22_14b_animate` (old `wan22_animate_14b` removed).

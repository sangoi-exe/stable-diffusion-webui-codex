# apps/backend/core/contracts Overview
<!-- tags: backend, core, contracts, assets, drift -->
Date: 2026-01-18
Last Review: 2026-05-30
Status: Active

## Purpose
- Owns backend “contract as code” modules that describe request invariants shared across UI ↔ API ↔ runtime (e.g. per-engine asset requirements).

## Key Files
- `apps/backend/core/contracts/asset_requirements.py` — Canonical per-engine asset requirements (VAE/text encoder) used by the API and exposed to the UI.
- `apps/backend/core/contracts/text_encoder_slots.py` — Header-only text encoder slot classifier used by the API to map sha-selected encoders into contract slots (order-independent).

## Notes
- Contracts here must be deterministic and fail loudly when an engine key is missing (prevents drift).
- Keep these modules lightweight: no heavy model imports at module import time.
- 2026-02-05: Added Anima engine asset contract (`anima`) and Qwen3-0.6B text encoder slot (`qwen3_06b`) for sha-selected TE resolution.
- 2026-05-17: Added the `qwen_image` asset contract with required Qwen Image VAE plus exactly one `qwen2_5_vl_7b` multimodal text-encoder slot; safetensors slot classification requires both embed dim 3584 and `visual.*` tower evidence.
- 2026-05-30: `EngineAssetContract.uses_vae` is the runtime VAE usage bit that controls explicit `vae_source` propagation; `requires_vae` means an external VAE asset is mandatory and must imply `uses_vae`.
- 2026-05-17: Explicit contract-ownership maps (`engine_id -> owner`, `semantic_engine -> owner`) cover executable and parked asset-contract ids only. Capability-only exact ids such as `flux1_fill` are intentionally absent so dependency/asset lookup fails loud instead of treating them as runtime aliases.
- 2026-02-16: WAN22 video engines now have explicit per-engine owners/contracts (`wan22_5b`, `wan22_14b`, `wan22_14b_animate`) with no owner alias fallback across model variants.
- 2026-02-20: Removed `wan22_14b_native` owner alias; stale engine id usage now fails loud at contract lookup.
- 2026-02-21: Semantic `wan22` contract owner baseline is now `wan22_14b` (instead of `wan22_5b`) to avoid stale 5B-default drift in semantic-only capability/contract surfaces.
- 2026-03-11: Added backend-only LTX2 asset contract owner `ltx2` with a required external `gemma3_12b` text encoder slot; header-only slot classification now recognizes Gemma3-12B safetensors/GGUF assets by embed-dim/arch metadata and stays fail-loud for unknown LLM layouts.
- 2026-03-12: `ltx2` now distinguishes non-core-only vs core-only truth: the current distilled GGUF pack is the core-only path (external video VAE + Gemma3-12B, connectors/audio bundle internal to the LTX2 family lane, no `mmproj` slot in this pass), while the non-core-only contract stays separate for self-contained checkpoints. Header-only GGUF slot classification now recognizes Gemma3-12B side assets from both llama.cpp-style GGUF headers (`gemma3.embedding_length=3840`) and Codex-converted GGUF headers (`model.architecture=gemma3`, `model.embedding_length=3840`), and still fails loud on `mmproj` projector GGUF files.
- 2026-05-23: Z-Image L2P Qwen3-4B GGUF text encoders classify into `qwen3_4b` only from dedicated Codex profile metadata (`zimage_l2p_tenc`, component `tenc`, family `zimage_l2p`, slot `qwen3_4b`); generic `llama` architecture metadata is not enough for the L2P TEnc path.

# apps/backend/infra/config Overview
Date: 2026-02-18
Last Review: 2026-07-30
Status: Active

## Purpose
- Houses backend configuration primitives (CLI parsing, env/bootstrap flags, repo/path resolution, runtime feature toggles).
- Keeps startup/runtime configuration contracts centralized and fail-loud.

## Key files
- `args.py` — Runtime CLI/env parser and `RuntimeMemoryConfig` builder.
- `bootstrap_env.py` — Bootstrap env overlay publication without mutating global `os.environ`.
- `env_flags.py` — Canonical env flag parsers shared by runtime diagnostics/features.
- `lora_apply_mode.py` — Strict enum/parser/reader for LoRA application mode; unset config resolves to `online`.
- `lora_merge_mode.py` — Strict enum/parser/reader for LoRA merge math mode (`fast|precise`).
- `lora_refresh_signature.py` — Strict enum/parser/reader for LoRA refresh signature mode (`structural|content_sha256`).
- `paths.py` — Paths.json/settings discovery and model root provisioning helpers.
- `repo_root.py` — Repo root resolution (honors `CODEX_ROOT` launcher contract).

## Notes
- 2026-02-18: Interactive device prompts in `args.py` now route explicit stdout writes through `apps.backend.infra.stdio` to keep primitive stream emission centralized while preserving CLI prompt behavior.
- 2026-02-18: Added LoRA loader runtime toggles `CODEX_LORA_MERGE_MODE` (`fast|precise`) and `CODEX_LORA_REFRESH_SIGNATURE` (`structural|content_sha256`) with strict parsing and CLI wiring (`--lora-merge-mode`, `--lora-refresh-signature`).
- 2026-02-20: `paths.py` now enforces fail-loud config semantics: invalid `apps/paths.json` parse/type errors raise, repo-relative entries are containment-checked against `CODEX_ROOT` (parent/symlink escapes rejected), and `_ensure_model_dirs` no longer swallows directory-provisioning failures.
- 2026-02-22: GGUF dequant-forward run cache flags were retired: `--gguf-dequant-cache=lvl1|lvl2` now fails loud with removal guidance, and tuning flags (`--gguf-dequant-cache-limit-mb`, `--gguf-dequant-cache-ratio`) are rejected as unsupported.
- 2026-02-23: `args.py` now supports `--main-device` and enforces a global main-device invariant (core/TE/VAE locked to one value), with fallback to `cuda` when available (else `cpu`) when not explicitly provided.
- 2026-07-30: Matching core/TE/VAE device policies collapse to `AUTO` only when main and mount resolve to the same backend; when mount differs, the explicit main backend remains the role owner so `AUTO -> mount` manager semantics stay unchanged.
- 2026-02-23: `args.py` now defaults offload authority to CPU when `--offload-device`/`CODEX_OFFLOAD_DEVICE` is unset or `auto`; invalid unresolved offload backend states now fail loud in `build_runtime_memory_config(...)` instead of silently mirroring main-device backend.
- 2026-02-23: `args.py` now validates `--cuda-malloc` against allocator env contract: strict mode fails loud unless `PYTORCH_CUDA_ALLOC_CONF` resolves to `backend:cudaMallocAsync` (including malformed/multiple backend entry detection).
- 2026-03-05: `paths.py` model-root provisioning now includes Flux.2 keys (`flux2_ckpt`, `flux2_tenc`, `flux2_vae`, `flux2_loras`) alongside existing families.
- 2026-03-12: `paths.py` model-root provisioning now includes LTX2 keys (`ltx2_ckpt`, `ltx2_tenc`, `ltx2_vae`, `ltx2_connectors`, `ltx2_loras`) so repo-relative `models/ltx2*` folders are created with the same startup path-seam used by other families.
- 2026-05-17: `paths.py` model-root provisioning includes Qwen Image split-asset roots (`qwen_image_ckpt`, `qwen_image_tenc`, `qwen_image_vae`) for the single `qwen_image` architecture family.
- 2026-05-23: `paths.py` model-root provisioning includes the Z-Image L2P denoiser root (`zimage_l2p_ckpt`) while reusing `zimage_tenc` for the required Qwen3-4B text encoder.
- Keep this folder focused on config/bootstrap contracts; runtime execution logic belongs outside `infra/config`.
- 2026-03-31: Bootstrap env naming must follow the real owner seam: runtime-global/bootstrap keys live here, family-prefixed keys stay in family-owned code, and shared runtime feature toggles must not be introduced under a model-family prefix.

Backend Deep Dive
=================

Scope
- Inventory core modules and flows; document flags, memory model, loader behavior and planned service endpoints.

Key Modules
- `backend/args.py`
  - Precision toggles (`--all-in-fp32/fp16`, `--unet-in-*`, `--vae-in-*`, `--clip-in-*`).
  - Attention (`--attention-split/quad/pytorch`, upcast toggles).
  - VRAM modes (`--always-*vram`), offload/stream toggles (`--always-offload-from-vram`, `--cuda-stream`, `--pin-shared-memory`).
  - Offline strict (`--disable-online-tokenizer`).

- `backend/memory_management.py`
  - VRAM/CPU detection (NVIDIA, Intel XPU, MPS), device resolution and dtype policies.
  - UNet/VAE/CLIP memory requirement estimation; offload/pin strategies.
  - Integrates with `backend/stream.py` when `--cuda-stream` is on.

- `backend/stream.py`
  - Safe creation/validation of CUDA/XPU streams; `should_use_stream()` gating.
  - Disabled on Windows; falls back to synchronous path on errors.

- `backend/loader.py`
  - Splits state dicts; loads components (diffusers/transformers) under `using_forge_operations(...)` with correct dtypes/devices.
  - Minimal HuggingFace assets: `config.json`/`model_index.json` and tokenizers (`tokenizer/`, `tokenizer_2/`).
  - Online (default): downloads only `*.json/*.txt` when missing. Offline strict: `--disable-online-tokenizer` fails fast with instructions.

- `backend/sampling/sampling_function.py`
  - Computes inference memory for UNet (+ControlNet +LoRA), warns on low VRAM, and calls GPU load/offload helpers.

- `backend/patcher/*`, `backend/operations*.py`, `backend/nn/*`
  - Patchers for LoRA; low-bit ops (bnb/gguf); integrated UNet/VAE/CLIP.

- `backend/services/*`
  - `image_service.py`, `media_service.py`, `options_service.py`, `sampler_service.py` centralize common endpoints utilities.

Recent Changes
- Loader strict offline: fail-fast when tokenizer/config assets are missing and `--disable-online-tokenizer` is set.
- UI SSR auto‑detection: SSR only when Node >= 20 is present (reduces launch latency and avoids manifest/queue 404s when SSR is unsupported).

Planned Backend Enhancements
- `/internal/memory` endpoint: JSON with VRAM/RAM free/total, vram_state, async loading toggle, pin_shared_memory, streams in use.
- Structured VRAM warnings: include recommended actions (reduce GPU Weight; turn off streams; switch to low‑vram mode) in API responses.
- Robust downloads (online mode): retry/backoff & short timeouts for `*.json/*.txt` when fetching from HF.

Validation Runbook
- Offline strict: remove tokenizer folders from a known repo; run with `--disable-online-tokenizer`; verify loader aborts with explicit instructions.
- Online: same scenario sem a flag; verificar que apenas `*.json/*.txt` são baixados.
- Low VRAM: force low‑vram; gerar imagem alta resolução; observar warnings + tempo.
- Streams: ligar `--cuda-stream`; monitorar latências e ausência de exceções.


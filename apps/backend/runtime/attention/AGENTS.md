# apps/backend/runtime/attention Overview
Date: 2025-10-28
Last Review: 2026-08-01
Status: Active

## Purpose
- Centralizes attention backend selection and related helpers (PyTorch SDPA, optional alt implementations).

## Notes
- Keep new attention kernels registered here so engines/runtime modules can reference a single entrypoint.
- 2026-03-19: Added the generic SRAM/shared-memory attention runtime bridge under `apps/backend/runtime/attention/sram/` and re-exported its contract helpers from `apps/backend/runtime/attention/__init__.py`. The first live consumer is WAN22 self-attention; this slice does not yet expose a global `AttentionBackend` selector for SRAM.
- 2026-01-24: Attention dispatch is now runtime-config-driven (no import-time backend binding). xFormers is imported lazily when selected and errors fail loud when unavailable/disabled.
- 2026-02-20: Added `attention_function_pre_shaped(...)` and causal-aware dispatcher signatures so pre-shaped `[B,H,S,D]` callsites can route through the central runtime backend selector without bypassing dispatcher contracts.
- 2026-02-20: PyTorch SDPA paths now support strict per-call policy forwarding (`auto|flash|mem_efficient|math`) through `attention_function(...)` / `attention_function_pre_shaped(...)`; non-PyTorch backend + policy combinations fail loud.
- 2026-02-21: Flash-only SDPA requests now warn and fallback deterministically (`mem_efficient` then `math`) when flash kernels are unavailable at runtime, instead of silently degrading.
- 2026-02-21: Flash policy now performs a precheck before attempting flash kernels (including head-dim constraint `D <= 256`); ineligible calls skip direct flash attempt and enter deterministic fallback immediately with explicit reason.
- 2026-05-24: PyTorch SDPA `auto` stays neutral and lets PyTorch choose; only explicit `flash` requests use the local flash precheck. That precheck checks non-null masks before device/dtype/head-dim constraints, and SDPA calls log requested/effective policy, mask descriptor, enabled CUDA SDP state, and q/k/v shapes once per request/shape/policy.
- 2026-05-24: `attention_pytorch(skip_reshape=True)` validates pre-shaped `[B,H,S,D]` inputs fail-loud, allows `q_len != k_len` when `k_len == v_len`, makes Q/K/V contiguous before PyTorch SDPA, and still returns flattened `[B,S,H*D]`; `attention_function_pre_shaped(...)` remains the 4D-facing owner.
- 2026-08-01: `_attention_pytorch_pre_shaped(...)` is the validated native `[B,H,S,D]` PyTorch SDPA owner. `attention_function_pre_shaped(...)` returns it directly for PyTorch, flattened callers convert once, and non-PyTorch adapters retain their prior path.

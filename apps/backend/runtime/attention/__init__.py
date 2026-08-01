"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Attention backend implementations (basic / chunked / xFormers / PyTorch SDPA) + diffusers processor adapter.
Provides memory/performance variants, precision-upcast helpers, single-head spatial legacy paths, neutral SDPA `auto` dispatch,
mask-aware explicit flash fallback/logging, and direct pre-shaped `[B,H,S,D]` PyTorch SDPA output without a flatten/reconstruct roundtrip.

Symbols (top-level; keep in sync; no ghosts):
- `get_attn_precision` (function): Resolves attention precision policy (handles global upcast and disable flags).
- `exists` (function): Small helper (`val is not None`) used across attention functions.
- `attention_basic` (function): Baseline attention (einsum) with optional mask handling and precision upcast.
- `attention_sub_quad` (function): Memory-saving attention variant (sub-quadratic/chunked) for long sequences.
- `attention_split` (function): Splits attention computation into chunks to reduce peak memory.
- `attention_xformers` (function): xFormers attention path (when available and not broken).
- `attention_pytorch` (function): PyTorch SDPA attention path with optional per-call SDPA policy (`auto|flash|mem_efficient|math`) and flash fallback warning behavior.
- `_attention_pytorch_pre_shaped` (function): Validated PyTorch SDPA owner that returns native `[B,H,S,D]` output.
- `_resolve_sdpa_policy_for_inputs` (function): Preserves neutral `auto` dispatch and routes explicit flash requests away from known-ineligible inputs.
- `_flash_sdpa_ineligibility_reason` (function): Validates hard flash-kernel constraints (mask/shape/head-dim/device/dtype) so known-ineligible flash calls skip direct flash attempts and enter deterministic fallback with explicit reason.
- `_log_sdpa_call_once` (function): Emits bounded PyTorch SDPA call diagnostics with backend, requested/effective policy, mask, enabled kernels, and q/k/v shapes.
- `attention_function` (function): Runtime-selected cross-attention dispatcher (driven by `memory_management.manager.config.attention.backend`) with optional SDPA policy forwarding for PyTorch backend.
- `attention_function_pre_shaped` (function): Dispatcher wrapper for pre-shaped Q/K/V tensors (`[B,H,S,D]` -> `[B,H,S,D]`), including optional SDPA policy forwarding.
- `attention_function_single_head_spatial` (function): Runtime-selected single-head spatial attention dispatcher (VAE; driven by runtime config).
- `slice_attention_single_head_spatial` (function): Single-head spatial attention variant using slicing/chunking.
- `normal_attention_single_head_spatial` (function): Baseline single-head spatial attention.
- `xformers_attention_single_head_spatial` (function): xFormers-backed single-head spatial attention.
- `pytorch_attention_single_head_spatial` (function): PyTorch SDPA-backed single-head spatial attention.
- `SramAttentionMode` (enum): Generic SRAM/shared-memory attention mode (`off|auto|force`) re-exported from the versioned runtime bridge.
- `SramAttentionContractError` (class): Fail-loud generic SRAM attention contract error.
- `SramAttentionAttemptResult` (dataclass): Generic SRAM attention attempt result for pre-shaped Q/K/V dispatch.
- `SramAttentionWarmupStatus` (dataclass): Generic SRAM attention warmup status with truthful `loaded` vs `ready`.
- `parse_sram_attention_mode` (function): Parses and validates the generic SRAM attention mode string.
- `resolve_effective_sram_attention_mode` (function): Resolves generic SRAM attention mode from override/env.
- `warmup_sram_attention_extension_for_load` (function): Warms up the generic SRAM attention extension and performs the narrow readiness smoke call.
- `is_sram_attention_extension_available` (function): Returns whether generic SRAM attention ops are loaded and registered.
- `last_sram_attention_extension_error` (function): Returns the last generic SRAM attention extension load/build error details.
- `AttentionProcessorCodex` (class): Diffusers-style attention processor adapter that dispatches to the selected attention backend.
"""

import math
from contextvars import ContextVar
from contextlib import nullcontext
from typing import Literal
from apps.backend.runtime.logging import get_backend_logger

import einops
import torch

from apps.backend.infra.config.args import args
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import AttentionBackend
from apps.backend.runtime.misc.sub_quadratic_attention import efficient_dot_product_attention
from apps.backend.runtime.attention.sram import (
    SramAttentionAttemptResult,
    SramAttentionContractError,
    SramAttentionMode,
    SramAttentionWarmupStatus,
    is_extension_available as is_sram_attention_extension_available,
    last_extension_error as last_sram_attention_extension_error,
    parse_sram_attention_mode,
    resolve_effective_sram_attention_mode,
    warmup_extension_for_load as warmup_sram_attention_extension_for_load,
)

_LOGGER = get_backend_logger("backend.attention")

# Avoid importing via backend facade during runtime package init to prevent cycles


_XFORMERS_OPS = None
_XFORMERS_BROKEN = None
_XFORMERS_VERSION = None
_XFORMERS_IMPORT_ERROR: Exception | None = None
_SDPA_CALL_LOGGED: set[tuple[str, str, str, str, str, str, str, str, str, str]] = set()
_SDPA_FLASH_FALLBACK_LOGGED: set[tuple[str, str, str, str, str, str]] = set()
_SDPA_NON_FLASH_FALLBACK_LOGGED: set[tuple[str, str, str, str, str, str, str]] = set()
_ATTENTION_REQUEST_ID_CTX: ContextVar[str] = ContextVar("attention_request_id", default="-")


def _require_xformers_ops():
    global _XFORMERS_OPS, _XFORMERS_BROKEN, _XFORMERS_VERSION, _XFORMERS_IMPORT_ERROR

    if _XFORMERS_OPS is not None:
        return _XFORMERS_OPS, bool(_XFORMERS_BROKEN), str(_XFORMERS_VERSION or "")

    if _XFORMERS_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(f"xformers is not available: {_XFORMERS_IMPORT_ERROR}") from _XFORMERS_IMPORT_ERROR

    try:
        import xformers  # type: ignore
        import xformers.ops  # type: ignore
    except Exception as exc:
        _XFORMERS_IMPORT_ERROR = exc
        raise ModuleNotFoundError(f"xformers is not available: {exc}") from exc

    version = str(getattr(xformers, "__version__", "") or "")
    _XFORMERS_VERSION = version
    _XFORMERS_BROKEN = version.startswith("0.0.2") and not version.startswith("0.0.20")
    _XFORMERS_OPS = xformers.ops

    return _XFORMERS_OPS, bool(_XFORMERS_BROKEN), version


def set_attention_request_id(request_id: str | None) -> str:
    normalized = str(request_id or "").strip() or "-"
    _ATTENTION_REQUEST_ID_CTX.set(normalized)
    return normalized


def get_attention_request_id() -> str:
    rid = _ATTENTION_REQUEST_ID_CTX.get()
    return rid if rid else "-"


__all__ = [name for name in globals() if not name.startswith("_")]


_SDPA_POLICY_VALUES = frozenset({"auto", "flash", "mem_efficient", "math"})
_SDPAPolicy = Literal["auto", "flash", "mem_efficient", "math"]


def _resolve_default_sdpa_policy() -> _SDPAPolicy:
    try:
        cfg = memory_management.manager.config.attention
    except Exception as exc:
        raise RuntimeError("Failed to resolve attention config while deriving default SDPA policy.") from exc
    if cfg.backend != AttentionBackend.PYTORCH:
        return "auto"
    if cfg.enable_flash and cfg.enable_mem_efficient:
        return "auto"
    if cfg.enable_flash:
        return "flash"
    if cfg.enable_mem_efficient:
        return "mem_efficient"
    return "math"


def _normalize_sdpa_policy(sdpa_policy: str | None) -> _SDPAPolicy:
    if sdpa_policy is None:
        return _resolve_default_sdpa_policy()
    if not isinstance(sdpa_policy, str):
        raise TypeError(
            "attention SDPA policy must be a string when provided "
            f"(got {type(sdpa_policy).__name__})."
        )
    normalized = sdpa_policy.strip().lower()
    if normalized not in _SDPA_POLICY_VALUES:
        allowed = ", ".join(sorted(_SDPA_POLICY_VALUES))
        raise RuntimeError(
            f"Unsupported attention SDPA policy {sdpa_policy!r}. "
            f"Allowed: {allowed}."
        )
    return normalized  # type: ignore[return-value]


def _sdpa_context(*, sdpa_policy: _SDPAPolicy, device: torch.device):
    if device.type != "cuda" or sdpa_policy == "auto":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend  # type: ignore[attr-defined]
        from torch.nn.attention import sdpa_kernel  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(
            "Per-call SDPA policy selection requires torch.nn.attention.sdpa_kernel support."
        ) from exc
    policy_to_backend = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    backend = policy_to_backend.get(sdpa_policy)
    if backend is None:
        raise RuntimeError(f"Unsupported SDPA policy {sdpa_policy!r} for torch.nn.attention.sdpa_kernel.")
    return sdpa_kernel(backend)


def _cuda_sdp_enabled_state() -> dict[str, object]:
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        return {}

    def _call_bool(name: str) -> object:
        enabled_fn = getattr(cuda_backend, name, None)
        if enabled_fn is None:
            return "missing"
        try:
            return bool(enabled_fn())
        except Exception as exc:
            return f"error:{type(exc).__name__}"

    return {
        "flash": _call_bool("flash_sdp_enabled"),
        "mem_efficient": _call_bool("mem_efficient_sdp_enabled"),
        "math": _call_bool("math_sdp_enabled"),
        "cudnn": _call_bool("cudnn_sdp_enabled"),
    }


def _sdpa_mask_description(mask: torch.Tensor | None) -> str:
    if mask is None:
        return "None"
    if not isinstance(mask, torch.Tensor):
        return f"{type(mask).__name__}"
    return f"{tuple(mask.shape)} {mask.dtype} {mask.device}"


def _log_sdpa_call_once(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    requested_policy: _SDPAPolicy,
    effective_policy: _SDPAPolicy,
    is_causal: bool,
) -> None:
    request_id = get_attention_request_id()
    mask_description = _sdpa_mask_description(mask)
    key = (
        request_id,
        str(q.device),
        str(q.dtype),
        str(tuple(q.shape)),
        str(tuple(k.shape)),
        str(tuple(v.shape)),
        mask_description,
        str(is_causal),
        requested_policy,
        effective_policy,
    )
    if key in _SDPA_CALL_LOGGED:
        return
    _SDPA_CALL_LOGGED.add(key)
    _LOGGER.info(
        "[attention][req=%s] sdpa call backend=pytorch_sdpa q=%s k=%s v=%s "
        "dtype=%s device=%s mask=%s is_causal=%s requested_policy=%s "
        "effective_policy=%s enabled=%s",
        request_id,
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        str(q.dtype),
        str(q.device),
        mask_description,
        bool(is_causal),
        requested_policy,
        effective_policy,
        _cuda_sdp_enabled_state(),
    )


def _log_sdpa_flash_fallback_once(
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_policy: _SDPAPolicy,
    fallback_policy: _SDPAPolicy,
    reason: str,
) -> None:
    request_id = get_attention_request_id()
    key = (request_id, str(device), str(dtype), requested_policy, fallback_policy, reason)
    if key in _SDPA_FLASH_FALLBACK_LOGGED:
        return
    _SDPA_FLASH_FALLBACK_LOGGED.add(key)
    _LOGGER.warning(
        "[attention][req=%s] flash SDPA route unavailable; requested_policy=%s "
        "falling back to policy=%s device=%s dtype=%s reason=%s",
        request_id,
        requested_policy,
        fallback_policy,
        str(device),
        str(dtype),
        reason,
    )


def _log_sdpa_non_flash_fallback_once(
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested_policy: _SDPAPolicy,
    primary_policy: _SDPAPolicy,
    fallback_policy: _SDPAPolicy,
    reason: str,
) -> None:
    request_id = get_attention_request_id()
    key = (request_id, str(device), str(dtype), requested_policy, primary_policy, fallback_policy, reason)
    if key in _SDPA_NON_FLASH_FALLBACK_LOGGED:
        return
    _SDPA_NON_FLASH_FALLBACK_LOGGED.add(key)
    _LOGGER.warning(
        "[attention][req=%s] non-flash SDPA route fallback: requested_policy=%s "
        "primary_policy=%s fallback_policy=%s device=%s dtype=%s reason=%s",
        request_id,
        requested_policy,
        primary_policy,
        fallback_policy,
        str(device),
        str(dtype),
        reason,
    )


def _default_non_flash_sdpa_policy(*, device: torch.device) -> _SDPAPolicy:
    # Deterministic non-flash default route:
    # - CUDA prefers mem_efficient first.
    # - Non-CUDA uses math.
    if device.type == "cuda":
        return "mem_efficient"
    return "math"


def _resolve_sdpa_policy_for_inputs(
    *,
    requested_policy: _SDPAPolicy,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> _SDPAPolicy:
    if requested_policy != "flash":
        return requested_policy
    if _flash_sdpa_ineligibility_reason(q, k, v, mask=mask) is None:
        return "flash"
    return _default_non_flash_sdpa_policy(device=q.device)


def _flash_sdpa_ineligibility_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None,
) -> str | None:
    if mask is not None:
        return (
            "flash SDPA does not support non-null attn_mask in this runtime path; "
            f"got mask={_sdpa_mask_description(mask)}."
        )
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return (
            "flash precheck expects q/k/v tensors with shape [B,H,S,D]; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    q_dim = int(q.shape[-1])
    k_dim = int(k.shape[-1])
    v_dim = int(v.shape[-1])
    if q_dim != k_dim or q_dim != v_dim:
        return (
            "flash requires q/k/v to share the same last dimension; "
            f"got q={q_dim}, k={k_dim}, v={v_dim}."
        )
    if q_dim > 256:
        return (
            "flash requires head_dim <= 256; "
            f"got head_dim={q_dim}."
        )
    if q.device.type != "cuda":
        return f"flash requires CUDA tensors (got device={q.device})."
    if q.dtype not in {torch.float16, torch.bfloat16}:
        return f"flash requires fp16/bf16 dtype (got dtype={q.dtype})."
    return None


def _run_pytorch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    is_causal: bool,
    sdpa_policy: str | None,
) -> torch.Tensor:
    requested_policy = _normalize_sdpa_policy(sdpa_policy)
    normalized_policy = _resolve_sdpa_policy_for_inputs(
        requested_policy=requested_policy,
        q=q,
        k=k,
        v=v,
        mask=mask,
    )

    def _run_once(policy: _SDPAPolicy) -> torch.Tensor:
        _log_sdpa_call_once(
            q=q,
            k=k,
            v=v,
            mask=mask,
            requested_policy=requested_policy,
            effective_policy=policy,
            is_causal=is_causal,
        )
        with _sdpa_context(sdpa_policy=policy, device=q.device):
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )

    if normalized_policy != "flash":
        if requested_policy == "flash":
            precheck_failure = _flash_sdpa_ineligibility_reason(q, k, v, mask=mask)
            if precheck_failure is None:
                precheck_failure = "flash SDPA route resolved to a non-flash fallback policy."
            try:
                _log_sdpa_flash_fallback_once(
                    device=q.device,
                    dtype=q.dtype,
                    requested_policy=requested_policy,
                    fallback_policy=normalized_policy,
                    reason=precheck_failure,
                )
                return _run_once(normalized_policy)
            except Exception as primary_exc:
                if normalized_policy == "math":
                    raise RuntimeError(
                        "SDPA policy flash precheck failed and fallback policy math failed. "
                        f"Reason: {precheck_failure}"
                    ) from primary_exc
                _log_sdpa_non_flash_fallback_once(
                    device=q.device,
                    dtype=q.dtype,
                    requested_policy=requested_policy,
                    primary_policy=normalized_policy,
                    fallback_policy="math",
                    reason=str(primary_exc),
                )
                return _run_once("math")
        return _run_once(normalized_policy)

    precheck_failure = _flash_sdpa_ineligibility_reason(q, k, v, mask=mask)
    if precheck_failure is not None:
        raise RuntimeError(
            "Internal SDPA policy error: flash remained effective after its precheck failed. "
            f"Reason: {precheck_failure}"
        )

    try:
        return _run_once("flash")
    except Exception as flash_exc:
        fallback_policies: list[_SDPAPolicy] = [_default_non_flash_sdpa_policy(device=q.device)]
        if "math" not in fallback_policies:
            fallback_policies.append("math")
        for fallback_policy in fallback_policies:
            try:
                _log_sdpa_flash_fallback_once(
                    device=q.device,
                    dtype=q.dtype,
                    requested_policy=requested_policy,
                    fallback_policy=fallback_policy,
                    reason=str(flash_exc),
                )
                return _run_once(fallback_policy)
            except Exception:
                continue
        raise RuntimeError(
            "SDPA policy flash failed and no fallback policy succeeded."
        ) from flash_exc


def get_attn_precision(attn_precision=torch.float32):
    if args.disable_attention_upcast:
        return None
    forced = memory_management.manager.force_upcast_attention_dtype()
    if forced is not None:
        return forced
    return attn_precision


def exists(val):
    return val is not None


def attention_basic(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, is_causal=False):
    if is_causal:
        raise RuntimeError("attention_basic does not support is_causal=True")
    attn_precision = get_attn_precision(attn_precision)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    scale = dim_head ** -0.5

    h = heads
    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

    if attn_precision == torch.float32:
        sim = torch.einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if exists(mask):
        if mask.dtype == torch.bool:
            mask = einops.rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = einops.repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        else:
            if len(mask.shape) == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
            sim.add_(mask)

    sim = sim.softmax(dim=-1)
    out = torch.einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return out


def attention_sub_quad(query, key, value, heads, mask=None, attn_precision=None, skip_reshape=False, is_causal=False):
    if is_causal:
        raise RuntimeError("attention_sub_quad does not support is_causal=True")
    attn_precision = get_attn_precision(attn_precision)

    if skip_reshape:
        b, _, _, dim_head = query.shape
    else:
        b, _, dim_head = query.shape
        dim_head //= heads

    if skip_reshape:
        query = query.reshape(b * heads, -1, dim_head)
        value = value.reshape(b * heads, -1, dim_head)
        key = key.reshape(b * heads, -1, dim_head).movedim(1, 2)
    else:
        query = query.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
        value = value.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
        key = key.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 3, 1).reshape(b * heads, dim_head, -1)

    dtype = query.dtype
    upcast_attention = attn_precision == torch.float32 and query.dtype != torch.float32
    if upcast_attention:
        bytes_per_token = torch.finfo(torch.float32).bits // 8
    else:
        bytes_per_token = torch.finfo(query.dtype).bits // 8
    batch_x_heads, q_tokens, _ = query.shape
    _, _, k_tokens = key.shape

    mem_free_total, mem_free_torch = memory_management.manager.get_free_memory(query.device, return_torch_stats=True)

    kv_chunk_size_min = None
    kv_chunk_size = None
    query_chunk_size = None

    for x in [4096, 2048, 1024, 512, 256]:
        count = mem_free_total / (batch_x_heads * bytes_per_token * x * 4.0)
        if count >= k_tokens:
            kv_chunk_size = k_tokens
            query_chunk_size = x
            break

    if query_chunk_size is None:
        query_chunk_size = 512

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

    hidden_states = efficient_dot_product_attention(
        query,
        key,
        value,
        query_chunk_size=query_chunk_size,
        kv_chunk_size=kv_chunk_size,
        kv_chunk_size_min=kv_chunk_size_min,
        use_checkpoint=False,
        upcast_attention=upcast_attention,
        mask=mask,
    )

    hidden_states = hidden_states.to(dtype)

    hidden_states = hidden_states.unflatten(0, (-1, heads)).transpose(1, 2).flatten(start_dim=2)
    return hidden_states


def attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, is_causal=False):
    if is_causal:
        raise RuntimeError("attention_split does not support is_causal=True")
    attn_precision = get_attn_precision(attn_precision)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    scale = dim_head ** -0.5
    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

    mem_free_total = memory_management.manager.get_free_memory(q.device)
    if mem_free_total <= 0:
        raise RuntimeError(
            "Not enough memory for attention: free memory estimate is non-positive "
            f"(have={mem_free_total} bytes, device={q.device})."
        )

    if attn_precision == torch.float32:
        element_size = 4
        upcast = True
    else:
        element_size = q.element_size()
        upcast = False

    gb = 1024 ** 3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * element_size
    modifier = 3
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
        # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
        #      f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

    if steps > 64:
        max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
        raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                           f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

    slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        if upcast:
            with torch.autocast(enabled=False, device_type='cuda'):
                s1 = torch.einsum('b i d, b j d -> b i j', q[:, i:end].float(), k.float()) * scale
        else:
            s1 = torch.einsum('b i d, b j d -> b i j', q[:, i:end], k) * scale

        if mask is not None:
            if len(mask.shape) == 2:
                s1 += mask[i:end]
            else:
                s1 += mask[:, i:end]

        s2 = s1.softmax(dim=-1).to(v.dtype)
        del s1

        r1[:, i:end] = torch.einsum('b i j, b j d -> b i d', s2, v)
        del s2

    del q, k, v

    r1 = (
        r1.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return r1


def attention_xformers(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, is_causal=False):
    if is_causal:
        raise RuntimeError("xformers attention path does not support is_causal=True")
    if not memory_management.manager.xformers_enabled():
        raise RuntimeError(
            "xformers attention was requested, but the active runtime is not configured for xformers. "
            "Set attention backend to xformers and ensure xformers is installed (and not disabled)."
        )

    xops, broken, _version = _require_xformers_ops()
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    if broken and b * heads > 65535:
        raise RuntimeError("xformers is broken for this batch*heads size; refusing to fall back to PyTorch attention")

    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.reshape(b, -1, heads, dim_head),
            (q, k, v),
        )

    if mask is not None:
        pad = 8 - q.shape[1] % 8
        mask_out = torch.empty([q.shape[0], q.shape[1], q.shape[1] + pad], dtype=q.dtype, device=q.device)
        mask_out[:, :, :mask.shape[-1]] = mask
        mask = mask_out[:, :, :mask.shape[-1]]

    out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)

    if skip_reshape:
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
    else:
        out = (
            out.reshape(b, -1, heads * dim_head)
        )

    return out


def _attention_pytorch_pre_shaped(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    is_causal: bool,
    sdpa_policy: str | None,
    expected_heads: int | None = None,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError(
            "_attention_pytorch_pre_shaped expects q/k/v with shape [B,H,S,D]; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    if tuple(q.shape[:2]) != tuple(k.shape[:2]) or tuple(q.shape[:2]) != tuple(v.shape[:2]):
        raise RuntimeError(
            "_attention_pytorch_pre_shaped expects matching [B,H] dimensions across q/k/v; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    if expected_heads is not None and int(q.shape[1]) != int(expected_heads):
        raise RuntimeError(
            "_attention_pytorch_pre_shaped heads mismatch: "
            f"q has {int(q.shape[1])}, expected {int(expected_heads)}."
        )
    if int(q.shape[-1]) != int(k.shape[-1]) or int(q.shape[-1]) != int(v.shape[-1]):
        raise RuntimeError(
            "_attention_pytorch_pre_shaped expects matching head dims across q/k/v; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    if int(k.shape[2]) != int(v.shape[2]):
        raise RuntimeError(
            "_attention_pytorch_pre_shaped expects matching K/V sequence lengths; "
            f"got k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    return _run_pytorch_sdpa(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        mask=mask,
        is_causal=is_causal,
        sdpa_policy=sdpa_policy,
    )


def attention_pytorch(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    is_causal=False,
    sdpa_policy: str | None = None,
):
    heads = int(heads)
    if heads <= 0:
        raise RuntimeError(f"attention_pytorch requires positive heads; got {heads}.")
    if skip_reshape:
        out = _attention_pytorch_pre_shaped(
            q,
            k,
            v,
            mask=mask,
            is_causal=is_causal,
            sdpa_policy=sdpa_policy,
            expected_heads=heads,
        )
        b = int(q.shape[0])
        q_tokens = int(q.shape[2])
        dim_head = int(q.shape[3])
    else:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise RuntimeError(
                "attention_pytorch(skip_reshape=False) expects q/k/v with shape [B,S,C]; "
                f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
            )
        b, _, dim_head = q.shape
        if int(k.shape[0]) != int(b) or int(v.shape[0]) != int(b):
            raise RuntimeError(
                "attention_pytorch batch mismatch for q/k/v: "
                f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
            )
        if int(k.shape[1]) != int(v.shape[1]):
            raise RuntimeError(
                "attention_pytorch K/V sequence length mismatch: "
                f"k={tuple(k.shape)} v={tuple(v.shape)}."
            )
        if int(k.shape[-1]) != int(dim_head) or int(v.shape[-1]) != int(dim_head):
            raise RuntimeError(
                "attention_pytorch inner-dim mismatch for q/k/v: "
                f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
            )
        if int(dim_head) % heads != 0:
            raise RuntimeError(f"attention_pytorch inner_dim={int(dim_head)} is not divisible by heads={heads}.")
        q_tokens = int(q.shape[1])
        dim_head //= heads
        q, k, v = map(
            lambda t: t.reshape(b, -1, heads, dim_head).transpose(1, 2).contiguous(),
            (q, k, v),
        )
        out = _attention_pytorch_pre_shaped(
            q,
            k,
            v,
            mask=mask,
            is_causal=is_causal,
            sdpa_policy=sdpa_policy,
            expected_heads=heads,
        )

    return out.transpose(1, 2).reshape(b, q_tokens, heads * dim_head)


def slice_attention_single_head_spatial(q, k, v):
    r1 = torch.zeros_like(k, device=q.device)
    scale = (int(q.shape[-1]) ** (-0.5))

    mem_free_total = memory_management.manager.get_free_memory(q.device)
    if mem_free_total <= 0:
        raise RuntimeError(
            "Not enough memory for spatial attention: free memory estimate is non-positive "
            f"(have={mem_free_total} bytes, device={q.device})."
        )

    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))

    while True:
        try:
            slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                s1 = torch.bmm(q[:, i:end], k) * scale

                s2 = torch.nn.functional.softmax(s1, dim=2).permute(0, 2, 1)
                del s1

                r1[:, :, i:end] = torch.bmm(v, s2)
                del s2
            break
        except memory_management.manager.oom_exception as e:
            memory_management.manager.soft_empty_cache(force=True)
            steps *= 2
            if steps > 128:
                raise e
            _LOGGER.warning("OOM during attention, increasing steps to %d", steps)

    return r1


def normal_attention_single_head_spatial(q, k, v):
    # compute attention
    b, c, h, w = q.shape

    q = q.reshape(b, c, h * w)
    q = q.permute(0, 2, 1)  # b,hw,c
    k = k.reshape(b, c, h * w)  # b,c,hw
    v = v.reshape(b, c, h * w)

    r1 = slice_attention_single_head_spatial(q, k, v)
    h_ = r1.reshape(b, c, h, w)
    del r1
    return h_


def xformers_attention_single_head_spatial(q, k, v):
    if not memory_management.manager.xformers_enabled_vae():
        raise RuntimeError(
            "xformers VAE attention was requested, but the active runtime is not configured for xformers VAE attention. "
            "Ensure xformers is installed, the attention backend is set to xformers, and VAE xformers is enabled."
        )
    xops, _broken, _version = _require_xformers_ops()

    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )

    out = xops.memory_efficient_attention(q, k, v, attn_bias=None)
    out = out.transpose(1, 2).reshape(B, C, H, W)
    return out


def pytorch_attention_single_head_spatial(q, k, v, *, sdpa_policy: str | None = None):
    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )

    out = _run_pytorch_sdpa(
        q,
        k,
        v,
        mask=None,
        is_causal=False,
        sdpa_policy=sdpa_policy,
    )
    out = out.transpose(2, 3).reshape(B, C, H, W)
    return out

def _selected_backend(*, backend_override: AttentionBackend | None = None) -> AttentionBackend:
    if backend_override is not None:
        return backend_override
    try:
        backend = memory_management.manager.config.attention.backend
    except Exception as exc:
        raise RuntimeError("Failed to resolve runtime attention backend from memory manager config.") from exc
    return backend


def attention_function(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    is_causal=False,
    backend: AttentionBackend | None = None,
    sdpa_policy: str | None = None,
):
    backend_selected = _selected_backend(backend_override=backend)
    normalized_sdpa_policy = _normalize_sdpa_policy(sdpa_policy) if sdpa_policy is not None else None
    if normalized_sdpa_policy is not None and backend_selected != AttentionBackend.PYTORCH:
        raise RuntimeError(
            "attention_function(sdpa_policy=...) is supported only for backend='pytorch'. "
            f"Got backend={backend_selected.value!r} policy={normalized_sdpa_policy!r}."
        )
    if skip_reshape:
        if q.ndim != 4:
            raise RuntimeError(
                "attention_function(skip_reshape=True) requires pre-shaped q with shape [B,H,S,D]; "
                f"got q={tuple(q.shape)}."
            )
        if int(q.shape[1]) != int(heads):
            raise RuntimeError(
                "attention_function(skip_reshape=True) requires `heads` to match q.shape[1] "
                f"(heads={int(heads)} q.shape[1]={int(q.shape[1])})."
            )
    if backend_selected == AttentionBackend.XFORMERS:
        return attention_xformers(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            is_causal=is_causal,
        )
    if backend_selected == AttentionBackend.PYTORCH:
        return attention_pytorch(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            is_causal=is_causal,
            sdpa_policy=normalized_sdpa_policy,
        )
    if backend_selected == AttentionBackend.SPLIT:
        return attention_split(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            is_causal=is_causal,
        )
    if backend_selected == AttentionBackend.QUAD:
        return attention_sub_quad(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            is_causal=is_causal,
        )
    raise RuntimeError(
        f"Unsupported attention backend {backend_selected!r} in attention_function; "
        f"expected one of {[backend.value for backend in AttentionBackend]}."
    )


def attention_function_pre_shaped(
    q,
    k,
    v,
    *,
    mask=None,
    is_causal=False,
    backend: AttentionBackend | None = None,
    sdpa_policy: str | None = None,
):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError(
            "attention_function_pre_shaped expects q/k/v with shape [B,H,S,D]; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    if tuple(q.shape[:2]) != tuple(k.shape[:2]) or tuple(q.shape[:2]) != tuple(v.shape[:2]):
        raise RuntimeError(
            "attention_function_pre_shaped expects matching [B,H] dimensions across q/k/v; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    if int(q.shape[-1]) != int(k.shape[-1]) or int(q.shape[-1]) != int(v.shape[-1]):
        raise RuntimeError(
            "attention_function_pre_shaped expects matching head dims across q/k/v; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    if int(k.shape[2]) != int(v.shape[2]):
        raise RuntimeError(
            "attention_function_pre_shaped expects matching K/V sequence lengths; "
            f"got k={tuple(k.shape)} v={tuple(v.shape)}."
        )
    batch = int(q.shape[0])
    heads = int(q.shape[1])
    q_tokens = int(q.shape[2])
    head_dim = int(q.shape[3])
    backend_selected = _selected_backend(backend_override=backend)
    normalized_sdpa_policy = _normalize_sdpa_policy(sdpa_policy) if sdpa_policy is not None else None
    if normalized_sdpa_policy is not None and backend_selected != AttentionBackend.PYTORCH:
        raise RuntimeError(
            "attention_function_pre_shaped(sdpa_policy=...) is supported only for backend='pytorch'. "
            f"Got backend={backend_selected.value!r} policy={normalized_sdpa_policy!r}."
        )
    if backend_selected == AttentionBackend.PYTORCH:
        return _attention_pytorch_pre_shaped(
            q,
            k,
            v,
            mask=mask,
            is_causal=is_causal,
            sdpa_policy=normalized_sdpa_policy,
            expected_heads=heads,
        )
    out = attention_function(
        q,
        k,
        v,
        heads=heads,
        mask=mask,
        skip_reshape=True,
        is_causal=is_causal,
        backend=backend_selected,
        sdpa_policy=None,
    )
    expected = heads * head_dim
    if out.ndim != 3 or int(out.shape[0]) != batch or int(out.shape[1]) != q_tokens or int(out.shape[2]) != expected:
        raise RuntimeError(
            "attention_function_pre_shaped expected flattened output [B,S,H*D]; "
            f"got {tuple(out.shape)} for q={tuple(q.shape)}."
        )
    return out.reshape(batch, q_tokens, heads, head_dim).transpose(1, 2).contiguous()


def attention_function_single_head_spatial(q, k, v):
    backend = _selected_backend()
    if backend == AttentionBackend.XFORMERS:
        return xformers_attention_single_head_spatial(q, k, v)
    if backend == AttentionBackend.PYTORCH:
        return pytorch_attention_single_head_spatial(q, k, v, sdpa_policy=None)
    return normal_attention_single_head_spatial(q, k, v)


class AttentionProcessorCodex:
    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        hidden_states = attention_function(query, key, value, heads=attn.heads, mask=attention_mask)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

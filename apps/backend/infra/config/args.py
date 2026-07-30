"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend CLI argument parsing and runtime memory config bootstrap.
Builds the argparse schema for runtime flags (devices/dtypes/attention/swap/smart offload) and turns argv/env into a `RuntimeMemoryConfig`.
Supports separate storage vs compute dtype overrides for core/text encoder/VAE (e.g., `--core-dtype` vs `--core-compute-dtype`) for stability and tuning.
Also parses diagnostics bootstrap toggles (`--trace-contract`, `--trace-profiler`) for runtime trace/profiler activation.
LoRA apply-mode bootstrap resolves unset config to `online` while preserving explicit `merge` overrides.
Also parses strict LoRA loader policy toggles (`--lora-merge-mode`, `--lora-refresh-signature`) for merge/signature behavior.
Main-device invariant support enforces a single runtime device authority: `--main-device` (launcher-provided) governs
core/TE/VAE and falls back to CUDA when available (else CPU) when omitted.
Mount/offload device invariants add explicit lifecycle control (`--mount-device`, `--offload-device`) with fail-loud
normalization; mount defaults to the resolved main device, offload defaults to CPU when unset/auto, and matching
component roles retain the explicit main backend whenever mount differs from main.

Symbols (top-level; keep in sync; no ghosts):
- `_build_parser` (function): Defines the argparse schema for backend runtime flags (devices/dtypes/attention/swap/etc).
- `_truthy` (function): Parses a string env/arg into a boolean (truthy/falsey).
- `_has_value` (function): Checks whether a parsed CLI option has a meaningful value (vs unset/default).
- `_apply_source_overrides` (function): Applies overrides from a source mapping onto the argparse namespace.
- `_validate_runtime_flags` (function): Validates behavior-changing runtime flag combinations (online LoRA math, runtime cache/allocator constraints).
- `_parse_pytorch_cuda_alloc_conf` (function): Parses `PYTORCH_CUDA_ALLOC_CONF` entries into validated `key:value` pairs.
- `_allocator_backend_from_cuda_env` (function): Extracts and validates allocator backend from `PYTORCH_CUDA_ALLOC_CONF`.
- `_validate_required_devices` (function): Ensures resolved device values obey the main-device invariant.
- `_normalize_device_choice` (function): Normalizes device choice strings (e.g., cpu/cuda/directml) into canonical form.
- `_device_backend_for_choice` (function): Maps normalized device choices into `DeviceBackend` values.
- `_cuda_available_for_main_device` (function): Detects CUDA availability for main-device fallback resolution.
- `_default_main_device_choice` (function): Resolves default main-device fallback (`cuda` when available, else `cpu`).
- `_resolve_main_device_choice` (function): Resolves authoritative main device from args/env/legacy component settings.
- `_resolve_aux_device_choice` (function): Resolves mount/offload device choices from args/env with explicit fallback policy.
- `_default_offload_device_choice` (function): Resolves default offload target when offload is unset/auto (CPU by default).
- `_validate_mount_offload_device_choices` (function): Enforces fail-loud mount/offload de-residency invariants (no CPU->accelerator unload target, no non-CPU offload==mount).
- `_normalize_dtype_choice` (function): Normalizes dtype choice strings (fp32/fp16/bf16/fp8) into canonical form.
- `_torch_dtype_for_choice` (function): Maps a dtype choice string to a torch dtype name (string form used across config objects).
- `_apply_component_device_overrides` (function): Applies per-component device overrides (core/vae/text encoders) to `RuntimeMemoryConfig`.
- `_apply_env_overrides` (function): Applies environment-variable overrides onto parsed args.
- `_resolve_attention_backend` (function): Resolves attention backend selection into `AttentionBackend`.
- `_resolve_attention_sdpa_policy` (function): Resolves SDPA policy selection (`auto|flash|mem_efficient|math`) for PyTorch attention backend.
- `build_runtime_memory_config` (function): Builds a `RuntimeMemoryConfig` from parsed args (includes validation + defaults).
- `initialize` (function): Entry-point helper; parses argv/env and returns the built runtime config (used by launchers).
"""

import argparse
import logging
import os
import sys
from typing import Mapping, MutableMapping, Sequence
from apps.backend.runtime.logging import get_backend_logger

from .lora_apply_mode import DEFAULT_LORA_APPLY_MODE, ENV_LORA_APPLY_MODE, LoraApplyMode, parse_lora_apply_mode
from .lora_online_math import (
    DEFAULT_LORA_ONLINE_MATH,
    ENV_LORA_ONLINE_MATH,
    LoraOnlineMath,
    parse_lora_online_math,
)
from .lora_merge_mode import (
    DEFAULT_LORA_MERGE_MODE,
    ENV_LORA_MERGE_MODE,
    LoraMergeMode,
    parse_lora_merge_mode,
)
from .lora_refresh_signature import (
    DEFAULT_LORA_REFRESH_SIGNATURE_MODE,
    ENV_LORA_REFRESH_SIGNATURE,
    LoraRefreshSignatureMode,
    parse_lora_refresh_signature_mode,
)

from apps.backend.runtime.memory.config import (
    AttentionBackend,
    AttentionConfig,
    DeviceBackend,
    DeviceRole,
    MemoryBudgets,
    PrecisionFlags,
    RuntimeMemoryConfig,
    SwapConfig,
    SwapMethod,
    SwapPolicy,
)

_LOG = get_backend_logger("backend.infra.config.args")
TRACE_DEBUG_DEFAULT = 10
_DEVICE_CHOICES: tuple[str, ...] = ("auto", "cuda", "cpu", "mps", "xpu", "directml")
_DEVICE_CHOICE_TO_BACKEND: dict[str, DeviceBackend] = {
    "auto": DeviceBackend.AUTO,
    "cuda": DeviceBackend.CUDA,
    "cpu": DeviceBackend.CPU,
    "mps": DeviceBackend.MPS,
    "xpu": DeviceBackend.XPU,
    "directml": DeviceBackend.DIRECTML,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID")

    fp_group = parser.add_mutually_exclusive_group()
    fp_group.add_argument("--all-in-fp16", action="store_true")

    fpcore_group = parser.add_mutually_exclusive_group()
    fpcore_group.add_argument("--core-in-bf16", action="store_true")
    fpcore_group.add_argument("--core-in-fp16", action="store_true")
    fpcore_group.add_argument("--core-in-fp8-e4m3fn", action="store_true")
    fpcore_group.add_argument("--core-in-fp8-e5m2", action="store_true")

    fpvae_group = parser.add_mutually_exclusive_group()
    fpvae_group.add_argument("--vae-in-fp16", action="store_true")
    fpvae_group.add_argument("--vae-in-fp32", action="store_true")
    fpvae_group.add_argument("--vae-in-bf16", action="store_true")

    parser.add_argument("--vae-in-cpu", action="store_true")

    fpte_group = parser.add_mutually_exclusive_group()
    fpte_group.add_argument("--clip-in-fp8-e4m3fn", action="store_true")
    fpte_group.add_argument("--clip-in-fp8-e5m2", action="store_true")
    fpte_group.add_argument("--clip-in-fp16", action="store_true")
    fpte_group.add_argument("--clip-in-fp32", action="store_true")

    attn_group = parser.add_mutually_exclusive_group()
    attn_group.add_argument("--attention-split", action="store_true")
    attn_group.add_argument("--attention-quad", action="store_true")
    attn_group.add_argument("--attention-pytorch", action="store_true")
    attn_group.add_argument(
        "--attention-backend",
        choices=["pytorch", "xformers", "split", "quad"],
        default=None,
        help=(
            "Attention backend selection (restart required). "
            "Use 'pytorch' for Torch SDPA, 'xformers' for xFormers attention, "
            "'split' for chunked attention, or 'quad' for sub-quadratic attention."
        ),
    )
    parser.add_argument(
        "--attention-sdpa-policy",
        choices=["auto", "flash", "mem_efficient", "math"],
        default=None,
        help=(
            "SDPA policy for PyTorch attention backend. "
            "Explicit policies ('flash', 'mem_efficient', 'math') require '--attention-backend=pytorch'; "
            "'auto' is treated as neutral for non-pytorch backends. "
            "'auto' lets PyTorch choose; 'flash' forces flash; "
            "'mem_efficient' forces efficient attention; 'math' forces math kernel."
        ),
    )

    upcast = parser.add_mutually_exclusive_group()
    upcast.add_argument("--force-upcast-attention", action="store_true")
    upcast.add_argument("--disable-attention-upcast", action="store_true")

    parser.add_argument("--disable-xformers", action="store_true")
    parser.add_argument(
        "--smart-offload",
        action="store_true",
        help="Load TE/UNet/VAE to GPU only for the active stage, offloading between steps.",
    )

    parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1)
    parser.add_argument("--disable-ipex-hijack", action="store_true")

    vram_group = parser.add_mutually_exclusive_group()
    vram_group.add_argument("--always-gpu", action="store_true")
    vram_group.add_argument("--always-high-vram", action="store_true")
    vram_group.add_argument("--always-normal-vram", action="store_true")
    vram_group.add_argument("--always-low-vram", action="store_true")
    vram_group.add_argument("--always-no-vram", action="store_true")
    vram_group.add_argument("--always-cpu", action="store_true")

    parser.add_argument("--always-offload-from-vram", action="store_true")
    parser.add_argument("--pytorch-deterministic", action="store_true")

    parser.add_argument("--cuda-malloc", action="store_true")
    parser.add_argument("--cuda-stream", action="store_true")
    parser.add_argument("--pin-shared-memory", action="store_true")

    parser.add_argument("--disable-gpu-warning", action="store_true")

    parser.add_argument("--disable-online-tokenizer", action="store_true")

    parser.add_argument(
        "--gguf-dequant-cache",
        choices=["off", "lvl1", "lvl2"],
        default=None,
        help=(
            "Removed feature flag for GGUF dequant-forward run cache. "
            "Only 'off' is supported in this build; selecting 'lvl1'/'lvl2' fails loud."
        ),
    )
    parser.add_argument(
        "--gguf-dequant-cache-limit-mb",
        type=int,
        default=None,
        metavar="MB",
        help=(
            "Removed feature flag for GGUF dequant-forward run cache. Must be unset."
        ),
    )
    parser.add_argument(
        "--gguf-dequant-cache-ratio",
        type=float,
        default=None,
        metavar="RATIO",
        help=(
            "Removed feature flag for GGUF dequant-forward run cache. Must be unset."
        ),
    )

    parser.add_argument(
        "--lora-apply-mode",
        choices=[m.value for m in LoraApplyMode],
        default=None,
        help=(
            "Global LoRA application mode: "
            "'online' applies patches on-the-fly during forward (default), "
            "'merge' rewrites weights once at apply-time. "
            "Changing this requires restarting the backend process."
        ),
    )

    parser.add_argument(
        "--lora-online-math",
        choices=[m.value for m in LoraOnlineMath],
        default=None,
        help=(
            "Online LoRA math mode when '--lora-apply-mode online' is active: "
            "'weight_merge' (default) materializes patched weights per forward; "
            "'activation' applies LoRA as an activation-side delta (reserved for packed GGUF kernels)."
        ),
    )
    parser.add_argument(
        "--lora-merge-mode",
        choices=[m.value for m in LoraMergeMode],
        default=None,
        help=(
            "Offline LoRA merge math mode: "
            "'fast' uses float32 accumulation (default), "
            "'precise' uses float64 accumulation to reduce repeated-merge numeric drift."
        ),
    )
    parser.add_argument(
        "--lora-refresh-signature",
        choices=[m.value for m in LoraRefreshSignatureMode],
        default=None,
        help=(
            "LoRA refresh signature mode: "
            "'structural' hashes bundle structure only, "
            "'content_sha256' hashes patch tensor contents to force refresh when values change (default)."
        ),
    )

    parser.add_argument(
        "--debug-conditioning",
        action="store_true",
        help="Emit verbose conditioning diagnostics during diffusion runs.",
    )

    parser.add_argument(
        "--debug-preview-factors",
        action="store_true",
        help="Emit a best-fit latent→RGB preview matrix (for building Approx-cheap live previews).",
    )

    parser.add_argument(
        "--trace-debug",
        action="store_true",
        help="Enable global call-trace debug (logger.debug for every Python call).",
    )
    parser.add_argument(
        "--trace-call-debug",
        dest="trace_debug",
        action="store_true",
        help="Alias for --trace-debug (global function-call trace).",
    )
    parser.add_argument(
        "--trace-debug-max-per-func",
        type=int,
        default=TRACE_DEBUG_DEFAULT,
        metavar="N",
        help="Maximum call logs per function when trace debug is enabled (<=0 disables limit).",
    )
    parser.add_argument(
        "--trace-call-debug-max-per-func",
        dest="trace_debug_max_per_func",
        type=int,
        default=TRACE_DEBUG_DEFAULT,
        metavar="N",
        help="Alias for --trace-debug-max-per-func.",
    )
    parser.add_argument(
        "--trace-contract",
        action="store_true",
        help="Enable contract-trace logging (JSONL events under logs/contract-trace).",
    )
    parser.add_argument(
        "--trace-profiler",
        action="store_true",
        help="Enable runtime torch profiler tracing for generation runs.",
    )

    parser.add_argument(
        "--swap-policy",
        choices=[p.value for p in SwapPolicy],
        default=SwapPolicy.CPU.value,
        help="Offload policy when VRAM is insufficient.",
    )
    parser.add_argument(
        "--swap-method",
        choices=[m.value for m in SwapMethod],
        default=SwapMethod.BLOCKED.value,
        help="Data transfer mode for swap operations (`blocked`, `async`, `block_swap_experimental`).",
    )
    parser.add_argument(
        "--gpu-prefer-construct",
        action="store_true",
        help="Prefer constructing models directly on GPU (no implicit fallback).",
    )

    parser.add_argument(
        "--main-device",
        choices=_DEVICE_CHOICES,
        default=None,
        help=(
            "Global runtime device authority. When set, core/text-encoder/VAE are forced to this device. "
            "When unset, backend resolves to CUDA if available, otherwise CPU."
        ),
    )
    parser.add_argument(
        "--mount-device",
        choices=_DEVICE_CHOICES,
        default=None,
        help=(
            "Model mount/load device authority. Defaults to resolved main device when unset/auto. "
            "Must be launcher-provided when non-default behavior is needed."
        ),
    )
    parser.add_argument(
        "--offload-device",
        choices=_DEVICE_CHOICES,
        default=None,
        help=(
            "Model offload target authority. Defaults to CPU when unset/auto. "
            "Use with swap policy controls for explicit residency behavior."
        ),
    )
    parser.add_argument(
        "--core-device",
        choices=_DEVICE_CHOICES,
        default=None,
        help="Explicit device for diffusion core (overrides saved WebUI settings).",
    )
    parser.add_argument(
        "--te-device",
        choices=_DEVICE_CHOICES,
        default=None,
        help="Explicit device for text encoder (overrides saved WebUI settings).",
    )
    parser.add_argument(
        "--vae-device",
        choices=_DEVICE_CHOICES,
        default=None,
        help="Explicit device for VAE (overrides saved WebUI settings).",
    )

    dtype_choices = ["auto", "fp16", "bf16", "fp32", "fp8_e4m3fn", "fp8_e5m2"]
    parser.add_argument(
        "--core-dtype",
        choices=dtype_choices,
        default=None,
        help="Preferred dtype for diffusion core (overrides saved WebUI settings).",
    )
    parser.add_argument(
        "--core-compute-dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default=None,
        help=(
            "Compute dtype for diffusion core activations (distinct from --core-dtype storage). "
            "Default is fp32 for stability."
        ),
    )
    parser.add_argument(
        "--te-dtype",
        choices=dtype_choices,
        default=None,
        help="Preferred dtype for text encoder (overrides saved WebUI settings).",
    )
    parser.add_argument(
        "--te-compute-dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default=None,
        help=(
            "Compute dtype for text encoder activations (distinct from --te-dtype storage). "
            "Default is fp32 for stability."
        ),
    )
    parser.add_argument(
        "--vae-dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default=None,
        help="Preferred dtype for VAE (overrides saved WebUI settings).",
    )
    parser.add_argument(
        "--vae-compute-dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default=None,
        help=(
            "Compute dtype for VAE activations (distinct from --vae-dtype storage). "
            "Default is fp32 for stability."
        ),
    )

    return parser


_DEVICE_DIRECTIVES = (
    ("main_device", "codex_main_device"),
    ("mount_device", "codex_mount_device"),
    ("offload_device", "codex_offload_device"),
    ("core_device", "codex_core_device"),
    ("te_device", "codex_te_device"),
    ("vae_device", "codex_vae_device"),
)

_DTYPE_DIRECTIVES = (
    ("core_dtype", "codex_core_dtype"),
    ("core_compute_dtype", "codex_core_compute_dtype"),
    ("te_dtype", "codex_te_dtype"),
    ("te_compute_dtype", "codex_te_compute_dtype"),
    ("vae_dtype", "codex_vae_dtype"),
    ("vae_compute_dtype", "codex_vae_compute_dtype"),
)


def _truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _apply_source_overrides(
    ns: argparse.Namespace,
    env_map: MutableMapping[str, str],
    settings: Mapping[str, object] | None,
) -> None:
    settings = settings or {}

    def _setting_value(key: str) -> str | None:
        if key not in settings:
            return None
        value = settings[key]
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    for flag_attr, settings_key in _DEVICE_DIRECTIVES + _DTYPE_DIRECTIVES:
        raw = getattr(ns, flag_attr, None)
        if raw is not None:
            text = str(raw).strip().lower()
            setattr(ns, settings_key, text or None)
            continue

        setting_val = _setting_value(settings_key)
        if setting_val:
            setattr(ns, settings_key, setting_val)

    # Attention backend is a runtime-wide policy (not per-request). Allow saved settings
    # to seed the initial runtime config when no CLI override was provided.
    attention_cli_override = any(
        bool(getattr(ns, name, False))
        for name in (
            "attention_split",
            "attention_quad",
            "attention_pytorch",
        )
    ) or _has_value(getattr(ns, "attention_backend", None))
    if not attention_cli_override:
        raw_backend = _setting_value("codex_attention_backend")
        if raw_backend:
            mapped = raw_backend.strip().lower()
            if mapped not in {"pytorch", "xformers", "split", "quad"}:
                raise RuntimeError(
                    "Invalid saved setting codex_attention_backend="
                    f"'{raw_backend}'. Allowed: pytorch, xformers, split, quad.",
                )
            ns.attention_backend = mapped

    if getattr(ns, "debug_conditioning", False):
        env_map["CODEX_DEBUG_COND"] = "1"

    if getattr(ns, "debug_preview_factors", False):
        env_map["CODEX_DEBUG_PREVIEW_FACTORS"] = "1"

    _ = env_map  # env_map only carries debug/log vars now (settings are payload/options-driven)


def _parse_pytorch_cuda_alloc_conf(raw_conf: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw_entry in str(raw_conf or "").split(","):
        token = raw_entry.strip()
        if not token:
            continue
        if ":" not in token:
            raise RuntimeError(
                "Invalid PYTORCH_CUDA_ALLOC_CONF entry "
                f"{token!r}: expected 'key:value' format."
            )
        key, value = token.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise RuntimeError(
                "Invalid PYTORCH_CUDA_ALLOC_CONF entry "
                f"{token!r}: expected non-empty 'key:value' parts."
            )
        entries.append((key, value))
    return entries


def _allocator_backend_from_cuda_env(env: Mapping[str, str]) -> str | None:
    raw_conf = str(env.get("PYTORCH_CUDA_ALLOC_CONF", "") or "").strip()
    if not raw_conf:
        return None
    backend: str | None = None
    for key, value in _parse_pytorch_cuda_alloc_conf(raw_conf):
        if key.lower() == "backend":
            if backend is not None:
                raise RuntimeError(
                    "Invalid PYTORCH_CUDA_ALLOC_CONF: multiple 'backend' entries found. "
                    "Use exactly one backend directive."
                )
            backend = value
    return backend


def _validate_runtime_flags(ns: argparse.Namespace, env: Mapping[str, str]) -> None:
    gguf_dequant_cache = str(getattr(ns, "gguf_dequant_cache", "off") or "off").strip().lower()
    gguf_dequant_cache_limit_mb = getattr(ns, "gguf_dequant_cache_limit_mb", None)
    gguf_dequant_cache_ratio = getattr(ns, "gguf_dequant_cache_ratio", None)
    lora_apply_mode = str(getattr(ns, "lora_apply_mode", DEFAULT_LORA_APPLY_MODE.value))
    lora_online_math = str(getattr(ns, "lora_online_math", DEFAULT_LORA_ONLINE_MATH.value))

    if lora_online_math == LoraOnlineMath.ACTIVATION.value:
        if lora_apply_mode != LoraApplyMode.ONLINE.value:
            raise RuntimeError("--lora-online-math=activation requires '--lora-apply-mode online'.")
        raise RuntimeError(
            "--lora-online-math=activation is reserved and not implemented yet in this build.",
        )

    if gguf_dequant_cache != "off":
        raise RuntimeError(
            "GGUF dequant-forward run cache (lvl1/lvl2) was removed. "
            "Set '--gguf-dequant-cache=off' and remove dequant-cache tuning flags."
        )
    if gguf_dequant_cache_limit_mb is not None:
        raise RuntimeError(
            "--gguf-dequant-cache-limit-mb is no longer supported because GGUF dequant-forward run cache was removed."
        )
    if gguf_dequant_cache_ratio is not None:
        raise RuntimeError(
            "--gguf-dequant-cache-ratio is no longer supported because GGUF dequant-forward run cache was removed."
        )

    if getattr(ns, "cuda_malloc", False):
        allocator_backend = _allocator_backend_from_cuda_env(env)
        if allocator_backend is None:
            raise RuntimeError(
                "--cuda-malloc requires PYTORCH_CUDA_ALLOC_CONF to include "
                "'backend:cudaMallocAsync'."
            )
        if allocator_backend.replace(" ", "").lower() != "cudamallocasync":
            raise RuntimeError(
                "--cuda-malloc requires PYTORCH_CUDA_ALLOC_CONF backend:cudaMallocAsync, "
                f"but found backend:{allocator_backend}."
            )

    attn_backend = _resolve_attention_backend(ns)
    _resolve_attention_sdpa_policy(ns, backend=attn_backend)
    if attn_backend == AttentionBackend.XFORMERS:
        if getattr(ns, "disable_xformers", False):
            raise RuntimeError("xformers attention backend is incompatible with --disable-xformers.")
        try:
            import xformers  # type: ignore # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"xformers attention backend requested, but xformers is not available: {exc}") from exc


def _normalize_device_choice(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if not v:
        return None
    alias_map = {
        "gpu": "cuda",
        "dml": "directml",
    }
    normalized = alias_map.get(v, v)
    if normalized in _DEVICE_CHOICES:
        return normalized
    raise ValueError(
        f"Unsupported device option '{value}'. Allowed: {', '.join(_DEVICE_CHOICES)}"
    )


def _device_backend_for_choice(choice: str | None) -> DeviceBackend | None:
    normalized = _normalize_device_choice(choice)
    if normalized is None:
        return None
    return _DEVICE_CHOICE_TO_BACKEND[normalized]


def _cuda_available_for_main_device() -> bool:
    try:
        import torch  # type: ignore

        return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        return False


def _default_main_device_choice() -> str:
    return "cuda" if _cuda_available_for_main_device() else "cpu"


def _default_offload_device_choice(_main_device_choice: str) -> str:
    return "cpu"


def _resolve_main_device_choice(ns: argparse.Namespace, env: Mapping[str, str]) -> str:
    explicit_main = _normalize_device_choice(getattr(ns, "codex_main_device", None))
    if explicit_main is None:
        env_main = _normalize_device_choice(str(env.get("CODEX_MAIN_DEVICE", "") or "").strip() or None)
        explicit_main = env_main

    if explicit_main is not None:
        return _default_main_device_choice() if explicit_main == "auto" else explicit_main

    component_choices = {
        "core": _normalize_device_choice(getattr(ns, "codex_core_device", None)),
        "te": _normalize_device_choice(getattr(ns, "codex_te_device", None)),
        "vae": _normalize_device_choice(getattr(ns, "codex_vae_device", None)),
    }

    non_auto = {
        value for value in component_choices.values() if value not in (None, "auto")
    }
    if len(non_auto) > 1:
        joined = ", ".join(f"{name}={value}" for name, value in component_choices.items() if value not in (None, "auto"))
        raise RuntimeError(
            "Component device divergence is not allowed without --main-device. "
            f"Found: {joined}. Set CODEX_MAIN_DEVICE/--main-device to a single value."
        )
    if len(non_auto) == 1:
        return next(iter(non_auto))
    return _default_main_device_choice()


def _resolve_aux_device_choice(
    ns: argparse.Namespace,
    env: Mapping[str, str],
    *,
    namespace_attr: str,
    env_key: str,
    fallback_choice: str,
) -> str:
    explicit = _normalize_device_choice(getattr(ns, namespace_attr, None))
    if explicit is None:
        env_raw = str(env.get(env_key, "") or "").strip() or None
        explicit = _normalize_device_choice(env_raw)
    if explicit in (None, "auto"):
        return fallback_choice
    return explicit


def _validate_mount_offload_device_choices(
    *,
    main_device_choice: str | None,
    mount_device_choice: str | None,
    offload_device_choice: str | None,
) -> None:
    main_device = _normalize_device_choice(main_device_choice)
    mount_device = _normalize_device_choice(mount_device_choice)
    offload_device = _normalize_device_choice(offload_device_choice)

    if main_device is None:
        raise RuntimeError("Offload-device invariant violated: missing resolved codex_main_device.")
    if mount_device is None:
        raise RuntimeError("Offload-device invariant violated: missing resolved codex_mount_device.")
    if offload_device is None:
        raise RuntimeError("Offload-device invariant violated: missing resolved codex_offload_device.")

    if main_device == "cpu" and offload_device != "cpu":
        raise RuntimeError(
            "Offload-device invariant violated: main_device=cpu requires offload_device=cpu. "
            f"Got offload_device={offload_device!r}."
        )

    if mount_device == "cpu" and offload_device != "cpu":
        raise RuntimeError(
            "Offload-device invariant violated: mount_device=cpu requires offload_device=cpu. "
            f"Got offload_device={offload_device!r}."
        )

    if mount_device != "cpu" and offload_device == mount_device:
        raise RuntimeError(
            "Offload-device invariant violated: offload_device matches mount_device for non-CPU unload "
            f"(mount={mount_device!r}, offload={offload_device!r}). "
            "Contract R requires real de-residency."
        )


def _normalize_dtype_choice(value: str | None, *, allow_fp8: bool = False) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if not v or v == "auto":
        return None
    mapping = {
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp32": "fp32",
        "float32": "fp32",
        "float": "fp32",
        "single": "fp32",
    }
    if allow_fp8:
        mapping.update(
            {
                "fp8_e4m3fn": "fp8_e4m3fn",
                "fp8-e4m3fn": "fp8_e4m3fn",
                "fp8_e4": "fp8_e4m3fn",
                "fp8_e5m2": "fp8_e5m2",
                "fp8-e5m2": "fp8_e5m2",
                "fp8_e5": "fp8_e5m2",
            }
        )
    result = mapping.get(v)
    if result is None:
        allowed = ", ".join(sorted(set(mapping.values())))
        raise ValueError(f"Unsupported dtype option '{value}'. Allowed: {allowed}")
    return result


def _validate_required_devices(ns: argparse.Namespace) -> None:
    """Ensure resolved runtime devices obey the global main-device invariant."""

    main_device = _normalize_device_choice(getattr(ns, "codex_main_device", None))
    if main_device is None:
        raise RuntimeError("Main-device invariant violated: codex_main_device was not resolved.")

    for attr in ("codex_core_device", "codex_te_device", "codex_vae_device"):
        value = _normalize_device_choice(getattr(ns, attr, None))
        if value is None:
            raise RuntimeError(f"Main-device invariant violated: missing {attr}.")
        if value != main_device:
            raise RuntimeError(
                "Main-device invariant violated: "
                f"{attr}={value!r} diverges from codex_main_device={main_device!r}."
            )

    for attr in ("codex_mount_device", "codex_offload_device"):
        value = _normalize_device_choice(getattr(ns, attr, None))
        if value is None:
            raise RuntimeError(f"Device invariant violated: missing {attr}.")

    _validate_mount_offload_device_choices(
        main_device_choice=main_device,
        mount_device_choice=getattr(ns, "codex_mount_device", None),
        offload_device_choice=getattr(ns, "codex_offload_device", None),
    )


def _torch_dtype_for_choice(choice: str | None) -> str | None:
    if choice is None:
        return None
    mapping = {
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
        "fp8_e4m3fn": "float16",
        "fp8_e5m2": "float16",
    }
    return mapping.get(choice)


def _apply_component_device_overrides(config: RuntimeMemoryConfig, ns: argparse.Namespace) -> None:
    main_backend = _device_backend_for_choice(getattr(ns, "codex_main_device", None))
    mount_backend = _device_backend_for_choice(getattr(ns, "codex_mount_device", None))
    role_choices = (
        (DeviceRole.CORE, getattr(ns, "codex_core_device", None), getattr(ns, "codex_core_dtype", None)),
        (DeviceRole.TEXT_ENCODER, getattr(ns, "codex_te_device", None), getattr(ns, "codex_te_dtype", None)),
        (DeviceRole.VAE, getattr(ns, "codex_vae_device", None), getattr(ns, "codex_vae_dtype", None)),
    )
    for role, device_choice, dtype_choice in role_choices:
        policy = config.component_policy(role)
        resolved_backend = _device_backend_for_choice(device_choice)
        if resolved_backend is not None:
            if (
                main_backend is not None
                and mount_backend is not None
                and resolved_backend == main_backend
                and resolved_backend == mount_backend
            ):
                policy.preferred_backend = DeviceBackend.AUTO
            else:
                policy.preferred_backend = resolved_backend

        forced = _torch_dtype_for_choice(dtype_choice)
        if resolved_backend == DeviceBackend.CPU:
            forced = "float32"
        if forced:
            policy.forced_dtype = forced

        compute_key = {
            DeviceRole.CORE: "codex_core_compute_dtype",
            DeviceRole.TEXT_ENCODER: "codex_te_compute_dtype",
            DeviceRole.VAE: "codex_vae_compute_dtype",
        }.get(role)
        if compute_key:
            compute_choice = getattr(ns, compute_key, None)
            forced_compute = _torch_dtype_for_choice(compute_choice)
            if resolved_backend == DeviceBackend.CPU and forced_compute and forced_compute != "float32":
                forced_compute = "float32"
            if forced_compute:
                policy.forced_compute_dtype = forced_compute


def _apply_env_overrides(ns: argparse.Namespace, env: Mapping[str, str]) -> None:
    if not _has_value(getattr(ns, "attention_backend", None)):
        raw_attention_backend = str(env.get("CODEX_ATTENTION_BACKEND", "") or "").strip().lower()
        if raw_attention_backend:
            ns.attention_backend = raw_attention_backend
    if not _has_value(getattr(ns, "attention_sdpa_policy", None)):
        raw_attention_sdpa_policy = str(env.get("CODEX_ATTENTION_SDPA_POLICY", "") or "").strip().lower()
        if raw_attention_sdpa_policy:
            ns.attention_sdpa_policy = raw_attention_sdpa_policy
    if not _has_value(getattr(ns, "codex_main_device", None)):
        raw_main_device = str(env.get("CODEX_MAIN_DEVICE", "") or "").strip().lower()
        if raw_main_device:
            ns.codex_main_device = raw_main_device
    if not _has_value(getattr(ns, "codex_mount_device", None)):
        raw_mount_device = str(env.get("CODEX_MOUNT_DEVICE", "") or "").strip().lower()
        if raw_mount_device:
            ns.codex_mount_device = raw_mount_device
    if not _has_value(getattr(ns, "codex_offload_device", None)):
        raw_offload_device = str(env.get("CODEX_OFFLOAD_DEVICE", "") or "").strip().lower()
        if raw_offload_device:
            ns.codex_offload_device = raw_offload_device

    def _set_core_dtype(val: str | None) -> None:
        ns.core_in_fp16 = False
        ns.core_in_bf16 = False
        ns.core_in_fp8_e4m3fn = False
        ns.core_in_fp8_e5m2 = False
        if not val:
            return
        v = val.strip().lower()
        ns.core_in_bf16 = False
        ns.core_in_fp16 = False
        ns.core_in_fp8_e4m3fn = False
        ns.core_in_fp8_e5m2 = False
        if v in {"bf16", "bfloat16"}:
            ns.core_in_bf16 = True
        elif v in {"fp16", "half"}:
            ns.core_in_fp16 = True
        elif v in {"fp8_e4m3fn", "fp8-e4m3fn", "fp8_e4"}:
            ns.core_in_fp8_e4m3fn = True
        elif v in {"fp8_e5m2", "fp8-e5m2", "fp8_e5"}:
            ns.core_in_fp8_e5m2 = True

    def _set_vae_dtype(val: str | None) -> None:
        if not val:
            return
        v = val.strip().lower()
        if v in {"bf16", "bfloat16"}:
            ns.vae_in_bf16 = True
            ns.vae_in_fp16 = False
            ns.vae_in_fp32 = False
        elif v in {"fp16", "half"}:
            ns.vae_in_bf16 = False
            ns.vae_in_fp16 = True
            ns.vae_in_fp32 = False
        elif v in {"fp32", "float", "single"}:
            ns.vae_in_bf16 = False
            ns.vae_in_fp16 = False
            ns.vae_in_fp32 = True

    def _set_te_dtype(val: str | None) -> None:
        for attr in ("clip_in_fp16", "clip_in_fp32", "clip_in_fp8_e4m3fn", "clip_in_fp8_e5m2", "clip_in_bf16"):
            setattr(ns, attr, False)
        if not val:
            return
        v = val.strip().lower()
        if v in {"fp16", "half", "float16"}:
            ns.clip_in_fp16 = True
        elif v in {"fp32", "float32", "float", "single"}:
            ns.clip_in_fp32 = True
        elif v in {"bf16", "bfloat16"}:
            ns.clip_in_bf16 = True
        elif v in {"fp8_e4m3fn", "fp8-e4m3fn", "fp8_e4"}:
            ns.clip_in_fp8_e4m3fn = True
        elif v in {"fp8_e5m2", "fp8-e5m2", "fp8_e5"}:
            ns.clip_in_fp8_e5m2 = True

    resolved_main_device = _resolve_main_device_choice(ns, env)
    ns.codex_main_device = resolved_main_device
    ns.codex_mount_device = _resolve_aux_device_choice(
        ns,
        env,
        namespace_attr="codex_mount_device",
        env_key="CODEX_MOUNT_DEVICE",
        fallback_choice=resolved_main_device,
    )
    ns.codex_offload_device = _resolve_aux_device_choice(
        ns,
        env,
        namespace_attr="codex_offload_device",
        env_key="CODEX_OFFLOAD_DEVICE",
        fallback_choice=_default_offload_device_choice(resolved_main_device),
    )
    _validate_mount_offload_device_choices(
        main_device_choice=resolved_main_device,
        mount_device_choice=ns.codex_mount_device,
        offload_device_choice=ns.codex_offload_device,
    )
    ns.codex_core_device = resolved_main_device
    ns.codex_te_device = resolved_main_device
    ns.codex_vae_device = resolved_main_device

    core_device_choice = resolved_main_device
    core_device_backend = _device_backend_for_choice(core_device_choice)
    core_dtype_raw = getattr(ns, "codex_core_dtype", None)
    core_dtype_choice = _normalize_dtype_choice(core_dtype_raw, allow_fp8=True)
    if core_device_backend == DeviceBackend.CPU and core_dtype_choice not in (None, "fp32"):
        core_dtype_choice = "fp32"
    _set_core_dtype(core_dtype_choice or core_dtype_raw)
    ns.codex_core_device = core_device_choice
    ns.codex_core_dtype = core_dtype_choice

    core_compute_dtype_raw = getattr(ns, "codex_core_compute_dtype", None)
    core_compute_dtype_choice = _normalize_dtype_choice(core_compute_dtype_raw, allow_fp8=False)
    if core_device_backend == DeviceBackend.CPU and core_compute_dtype_choice not in (None, "fp32"):
        core_compute_dtype_choice = "fp32"
    ns.codex_core_compute_dtype = core_compute_dtype_choice

    vae_device_choice = resolved_main_device
    vae_device_backend = _device_backend_for_choice(vae_device_choice)
    vae_dtype_raw = getattr(ns, "codex_vae_dtype", None)
    vae_dtype_choice = _normalize_dtype_choice(vae_dtype_raw, allow_fp8=False)
    if vae_device_backend == DeviceBackend.CPU:
        ns.vae_in_cpu = True
        if vae_dtype_choice not in (None, "fp32"):
            vae_dtype_choice = "fp32"
    _set_vae_dtype(vae_dtype_choice or vae_dtype_raw)
    ns.codex_vae_device = vae_device_choice
    ns.codex_vae_dtype = vae_dtype_choice

    vae_compute_dtype_raw = getattr(ns, "codex_vae_compute_dtype", None)
    vae_compute_dtype_choice = _normalize_dtype_choice(vae_compute_dtype_raw, allow_fp8=False)
    if vae_device_backend == DeviceBackend.CPU and vae_compute_dtype_choice not in (None, "fp32"):
        vae_compute_dtype_choice = "fp32"
    ns.codex_vae_compute_dtype = vae_compute_dtype_choice

    te_device_choice = resolved_main_device
    te_device_backend = _device_backend_for_choice(te_device_choice)
    te_dtype_raw = getattr(ns, "codex_te_dtype", None)
    te_dtype_choice = _normalize_dtype_choice(te_dtype_raw, allow_fp8=True)
    if te_device_backend == DeviceBackend.CPU and te_dtype_choice not in (None, "fp32"):
        te_dtype_choice = "fp32"
    _set_te_dtype(te_dtype_choice or te_dtype_raw)
    ns.codex_te_device = te_device_choice
    ns.codex_te_dtype = te_dtype_choice

    te_compute_dtype_raw = getattr(ns, "codex_te_compute_dtype", None)
    te_compute_dtype_choice = _normalize_dtype_choice(te_compute_dtype_raw, allow_fp8=False)
    if te_device_backend == DeviceBackend.CPU and te_compute_dtype_choice not in (None, "fp32"):
        te_compute_dtype_choice = "fp32"
    ns.codex_te_compute_dtype = te_compute_dtype_choice

    if te_device_backend == DeviceBackend.CPU and not getattr(ns, "clip_in_fp32", False):
        ns.clip_in_fp32 = True

    if te_device_backend == DeviceBackend.CPU and getattr(ns, "clip_in_fp16", False):
        ns.clip_in_fp16 = False

    if _truthy(env.get("CODEX_DEBUG_COND")):
        ns.debug_conditioning = True

    if _truthy(env.get("CODEX_DEBUG_PREVIEW_FACTORS")):
        ns.debug_preview_factors = True

    # Global call tracing (function-level). This toggles a runtime hook in
    # the API entrypoint; we keep the flag for visibility in the parsed args.
    if _truthy(env.get("CODEX_TRACE_CALL_DEBUG")):
        ns.trace_debug = True
    # Honour max-calls-per-func trace limit from env (used by BIOS/TUI)
    raw_trace_max = env.get("CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC")
    if raw_trace_max is not None:
        try:
            ns.trace_debug_max_per_func = max(0, int(raw_trace_max))
        except Exception:
            ns.trace_debug_max_per_func = TRACE_DEBUG_DEFAULT

    if _truthy(env.get("CODEX_TRACE_CONTRACT")):
        ns.trace_contract = True

    if _truthy(env.get("CODEX_TRACE_PROFILER")) or _truthy(env.get("CODEX_PROFILE")):
        ns.trace_profiler = True

    # LoRA apply mode (global): honour env only when CLI arg is unset.
    raw_lora_mode = env.get(ENV_LORA_APPLY_MODE)
    if raw_lora_mode is not None and not _has_value(getattr(ns, "lora_apply_mode", None)):
        try:
            ns.lora_apply_mode = parse_lora_apply_mode(raw_lora_mode).value
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    # LoRA online math mode: honour env only when CLI arg is unset.
    raw_lora_online_math = env.get(ENV_LORA_ONLINE_MATH)
    if raw_lora_online_math is not None and not _has_value(getattr(ns, "lora_online_math", None)):
        try:
            ns.lora_online_math = parse_lora_online_math(raw_lora_online_math).value
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    # LoRA merge mode (global): honour env only when CLI arg is unset.
    raw_lora_merge_mode = env.get(ENV_LORA_MERGE_MODE)
    if raw_lora_merge_mode is not None and not _has_value(getattr(ns, "lora_merge_mode", None)):
        try:
            ns.lora_merge_mode = parse_lora_merge_mode(raw_lora_merge_mode).value
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    # LoRA refresh signature mode (global): honour env only when CLI arg is unset.
    raw_lora_refresh_signature = env.get(ENV_LORA_REFRESH_SIGNATURE)
    if raw_lora_refresh_signature is not None and not _has_value(getattr(ns, "lora_refresh_signature", None)):
        try:
            ns.lora_refresh_signature = parse_lora_refresh_signature_mode(raw_lora_refresh_signature).value
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc


def _resolve_attention_backend(ns: argparse.Namespace) -> AttentionBackend:
    explicit = getattr(ns, "attention_backend", None)
    if isinstance(explicit, str) and explicit.strip():
        normalized = explicit.strip().lower()
        mapping = {
            "pytorch": AttentionBackend.PYTORCH,
            "xformers": AttentionBackend.XFORMERS,
            "split": AttentionBackend.SPLIT,
            "quad": AttentionBackend.QUAD,
        }
        resolved = mapping.get(normalized)
        if resolved is not None:
            return resolved
        allowed = ", ".join(sorted(mapping))
        raise RuntimeError(f"Unsupported attention backend '{explicit}'. Allowed: {allowed}.")

    if ns.attention_split:
        return AttentionBackend.SPLIT
    if ns.attention_quad:
        return AttentionBackend.QUAD
    if ns.attention_pytorch:
        return AttentionBackend.PYTORCH
    return AttentionBackend.PYTORCH


def _resolve_attention_sdpa_policy(
    ns: argparse.Namespace,
    *,
    backend: AttentionBackend,
) -> str:
    explicit = getattr(ns, "attention_sdpa_policy", None)
    if not isinstance(explicit, str) or not explicit.strip():
        return "auto"
    normalized = explicit.strip().lower()
    if normalized not in {"auto", "flash", "mem_efficient", "math"}:
        allowed = "auto, flash, mem_efficient, math"
        raise RuntimeError(f"Unsupported attention SDPA policy '{explicit}'. Allowed: {allowed}.")
    if backend != AttentionBackend.PYTORCH and normalized != "auto":
        raise RuntimeError("--attention-sdpa-policy requires '--attention-backend=pytorch'.")
    return normalized


def build_runtime_memory_config(ns: argparse.Namespace) -> RuntimeMemoryConfig:
    precision = PrecisionFlags(
        all_fp16=ns.all_in_fp16,
        core_fp16=ns.core_in_fp16,
        core_bf16=ns.core_in_bf16,
        core_fp8_e4m3fn=ns.core_in_fp8_e4m3fn,
        core_fp8_e5m2=ns.core_in_fp8_e5m2,
        vae_fp16=ns.vae_in_fp16,
        vae_fp32=ns.vae_in_fp32,
        vae_bf16=ns.vae_in_bf16,
        vae_in_cpu=ns.vae_in_cpu,
        clip_fp16=getattr(ns, "clip_in_fp16", False),
        clip_fp32=getattr(ns, "clip_in_fp32", False),
        clip_bf16=getattr(ns, "clip_in_bf16", False),
        clip_fp8_e4m3fn=getattr(ns, "clip_in_fp8_e4m3fn", False),
        clip_fp8_e5m2=getattr(ns, "clip_in_fp8_e5m2", False),
    )

    attention_backend = _resolve_attention_backend(ns)
    sdpa_policy = _resolve_attention_sdpa_policy(ns, backend=attention_backend)
    force_upcast = bool(ns.force_upcast_attention)
    if getattr(ns, "disable_attention_upcast", False):
        force_upcast = False

    attention = AttentionConfig(
        backend=attention_backend,
        enable_flash=attention_backend == AttentionBackend.PYTORCH and sdpa_policy in {"auto", "flash"},
        enable_mem_efficient=attention_backend == AttentionBackend.PYTORCH and sdpa_policy in {"auto", "mem_efficient"},
        force_upcast=force_upcast,
        allow_split_fallback=True,
        allow_quad_fallback=True,
    )

    swap = SwapConfig(
        policy=SwapPolicy(ns.swap_policy),
        method=SwapMethod(ns.swap_method),
        always_offload=ns.always_offload_from_vram,
        pin_shared_memory=ns.pin_shared_memory,
    )

    config = RuntimeMemoryConfig(
        device_backend=DeviceBackend.AUTO,
        mount_device_backend=DeviceBackend.AUTO,
        offload_device_backend=DeviceBackend.AUTO,
        gpu_device_id=ns.gpu_device_id,
        gpu_prefer_construct=ns.gpu_prefer_construct,
        precision=precision,
        swap=swap,
        attention=attention,
        budgets=MemoryBudgets(),
        deterministic_algorithms=ns.pytorch_deterministic,
        disable_xformers=ns.disable_xformers,
        enable_xformers_vae=not ns.disable_xformers,
    )

    if ns.always_gpu:
        config.device_backend = DeviceBackend.CUDA
    elif ns.always_cpu:
        config.device_backend = DeviceBackend.CPU
    elif ns.directml is not None:
        config.device_backend = DeviceBackend.DIRECTML
        config.allow_directml = True

    if ns.vae_in_cpu:
        config.component_policy(DeviceRole.VAE).preferred_backend = DeviceBackend.CPU

    _apply_component_device_overrides(config, ns)

    resolved_main_backend = _device_backend_for_choice(getattr(ns, "codex_main_device", None))
    if resolved_main_backend is not None and resolved_main_backend != DeviceBackend.AUTO:
        config.device_backend = resolved_main_backend

    resolved_mount_backend = _device_backend_for_choice(getattr(ns, "codex_mount_device", None))
    if resolved_mount_backend in (None, DeviceBackend.AUTO):
        resolved_mount_backend = config.device_backend
    config.mount_device_backend = resolved_mount_backend

    resolved_offload_backend = _device_backend_for_choice(getattr(ns, "codex_offload_device", None))
    if resolved_offload_backend in (None, DeviceBackend.AUTO):
        raise RuntimeError(
            "Offload-device invariant violated: codex_offload_device resolved to an invalid backend."
        )
    config.offload_device_backend = resolved_offload_backend

    if DeviceBackend.DIRECTML in {
        config.device_backend,
        config.mount_device_backend,
        config.offload_device_backend,
    }:
        config.allow_directml = True

    return config


_PARSER = _build_parser()
_args: argparse.Namespace | None = None
_memory_config: RuntimeMemoryConfig | None = None
_UNKNOWN: list[str] = []

def initialize(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    settings: Mapping[str, object] | None = None,
    *,
    strict: bool = True,
) -> tuple[argparse.Namespace, RuntimeMemoryConfig]:
    """Parse runtime arguments applying CLI/env/settings precedence.

    Returns the parsed namespace and freshly built RuntimeMemoryConfig.
    When ``strict`` is True, raises RuntimeError for unknown CLI arguments and
    if required device flags are missing after applying overrides.
    """
    global _args, _memory_config, _UNKNOWN

    argv_list = list(argv) if argv is not None else sys.argv[1:]
    namespace, unknown = _PARSER.parse_known_args(argv_list)
    _UNKNOWN = list(unknown)

    deprecated = [arg for arg in unknown if arg.startswith("--unet-in-")]
    if deprecated:
        raise RuntimeError(
            "Deprecated precision flag(s) detected: "
            + ", ".join(deprecated)
            + ". Use '--core-in-*' variants instead."
        )
    if strict and unknown:
        raise RuntimeError(
            "Unknown CLI argument(s): "
            + ", ".join(unknown)
            + ". Remove unsupported flags from the runtime bootstrap command."
        )

    source_env = env if env is not None else os.environ
    env_map: MutableMapping[str, str] = {}
    for key, value in source_env.items():
        if value is None:
            continue
        env_map[key] = str(value)

    _apply_source_overrides(namespace, env_map, settings)

    if getattr(namespace, "trace_debug_max_per_func", None) is None:
        namespace.trace_debug_max_per_func = TRACE_DEBUG_DEFAULT
    elif namespace.trace_debug_max_per_func < 0:
        namespace.trace_debug_max_per_func = 0
    _apply_env_overrides(namespace, env_map)
    if getattr(namespace, "lora_apply_mode", None) is None:
        namespace.lora_apply_mode = DEFAULT_LORA_APPLY_MODE.value
    if getattr(namespace, "gguf_dequant_cache", None) is None:
        namespace.gguf_dequant_cache = "off"
    if getattr(namespace, "lora_online_math", None) is None:
        namespace.lora_online_math = DEFAULT_LORA_ONLINE_MATH.value
    if getattr(namespace, "lora_merge_mode", None) is None:
        namespace.lora_merge_mode = DEFAULT_LORA_MERGE_MODE.value
    if getattr(namespace, "lora_refresh_signature", None) is None:
        namespace.lora_refresh_signature = DEFAULT_LORA_REFRESH_SIGNATURE_MODE.value

    if strict:
        _validate_runtime_flags(namespace, env_map)
        _validate_required_devices(namespace)
    config = build_runtime_memory_config(namespace)

    _args = namespace
    _memory_config = config
    return namespace, config


# Initialise module defaults with non-strict semantics so early imports don't abort.
args, memory_config = initialize(strict=False)

dynamic_args = {
    "embedding_dir": "./embeddings",
    "emphasis_name": "original",
}


__all__ = [
    "args",
    "memory_config",
    "dynamic_args",
    "build_runtime_memory_config",
    "initialize",
]

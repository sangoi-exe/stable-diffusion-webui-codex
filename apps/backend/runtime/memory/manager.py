"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native runtime memory management service (hardware probe + precision/budget policies + full/streamed residency registry).
Provides a single manager that decides device/precision defaults, tracks fully resident and explicitly streamed components, and applies swap/offload policies during engine orchestration.
Resolves explicit main/mount/offload device backends from runtime config and keeps lifecycle device routing centralized.
Exposes per-role storage vs compute dtype selection (core/TE default to fp16 on accelerator devices, VAE defaults to fp32, CPU remains fp32 unless overridden), supports “native weights dtype” selection via `dtype_for_role(..., native_dtype=...)`, and provides `is_model_loaded(...)` for stage-scoped smart offload decisions.
Also emits canonical smart-offload INFO audit events for model load/unload transitions and unload no-op requests when smart offload is active.
Smart-offload action payloads include best-effort memory windows (`memory_before_*`/`memory_after_*`) to provide global per-action residency telemetry.
`unload_model(...)` now treats manager-level `offload_device` as authoritative and fail-loud: explicit unload must route to the configured offload target (no legacy CPU-force override path).
Generic smart-offload action emission (`load`/`unload`/`unload_noop`) is centralized here, with optional caller context fields (`source`/`stage`/`component_hint`/`event_reason`).
Per-item load telemetry keys remain exposed in `memory_snapshot()['models']`; expensive per-load counter probes are captured only when memory-debug diagnostics are enabled.
Partial-load rollback covers `BaseException`, preserves the primary failure, physically unloads every attempted record,
and unregisters only records already present in the residency registry.

Symbols (top-level; keep in sync; no ghosts):
- `_PrecisionState` (dataclass): Internal precision selection state (derived from hardware + configured flags) used to choose dtypes.
- `_detect_oom_exception` (function): Detects the appropriate OOM exception class for the active backend/runtime.
- `_normalize_device_name` (function): Normalizes device name strings for stable matching and policy decisions.
- `_device_has_native_bf16` (function): Heuristic for whether a device likely supports native BF16 (name + compute capability).
- `_probe_hardware` (function): Performs hardware probing and returns a `HardwareProbe` (raises `HardwareProbeError` on failure).
- `_LoadedModelRecord` (dataclass): Tracks one loaded model/component, its full/streamed residency contract, verified snapshot, and per-load telemetry.
- `CodexMemoryManager` (class): Main memory manager; owns runtime config, budget calculation, model registry, and policy decisions
  (contains many methods for load/unload bookkeeping, swap/offload behavior, and “best defaults” selection).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, MutableSequence, Optional, Sequence, Set, Tuple

import torch

from apps.backend.runtime.diagnostics.fallback_state import mark_fallback_used
from apps.backend.runtime.load_authority import (
    LoadAuthorityStage,
    guarded_load_entrypoint,
)

from .config import (
    AttentionBackend,
    DeviceBackend,
    DeviceRole,
    HardwareProbe,
    RuntimeMemoryConfig,
    SwapPolicy,
)
from .exceptions import HardwareProbeError, MemoryConfigurationError, MemoryLoadError
from .smart_offload import SmartOffloadAction, log_smart_offload_action, smart_offload_enabled
from .streamed_residency import (
    ResidencyMode,
    StreamedCoreRuntime,
    StreamedFootprint,
    StreamedResidencyPhase,
    StreamedResidencySnapshot,
)


logger = get_backend_logger("backend.memory.manager")


_NATIVE_BF16_NAME_FRAGMENTS: Tuple[str, ...] = (
    "geforce rtx 30",
    "geforce rtx 40",
    "geforce rtx 50",
    "nvidia a100",
    "nvidia a10",
    "nvidia a30",
    "nvidia a40",
    "nvidia l4",
    "nvidia l40",
    "nvidia l40s",
    "nvidia h100",
    "nvidia h200",
    "amd instinct mi100",
    "amd instinct mi200",
    "amd instinct mi250",
    "amd instinct mi250x",
    "amd instinct mi300",
    "amd instinct mi300a",
    "amd instinct mi300x",
    "intel arc a",
    "intel data center gpu max",
)

_PRECISION_HINTS: Dict[DeviceRole, str] = {
    DeviceRole.CORE: "--core-dtype fp32",
    DeviceRole.TEXT_ENCODER: "--te-dtype fp32",
    DeviceRole.VAE: "--vae-dtype fp32",
}


@dataclass(slots=True)
class _PrecisionState:
    role: DeviceRole
    ladder: Tuple[torch.dtype, ...]
    manual_override: bool = False
    index: int = 0

    def current(self) -> torch.dtype:
        return self.ladder[min(self.index, len(self.ladder) - 1)]

    def select(self, supported: Sequence[torch.dtype]) -> torch.dtype:
        if not supported:
            raise MemoryConfigurationError(f"No supported dtypes provided for {self.role.value}.")
        if self.manual_override:
            dtype = self.current()
            if dtype not in supported:
                return supported[-1]
            return dtype
        # Prefer staying on current index if valid, otherwise advance within ladder order
        cur = self.current()
        if cur in supported:
            return cur
        # search forward from current index
        for idx in range(self.index + 1, len(self.ladder)):
            dtype = self.ladder[idx]
            if dtype in supported:
                self.index = idx
                return dtype
        # search from start if nothing ahead is supported
        for idx in range(0, self.index):
            dtype = self.ladder[idx]
            if dtype in supported:
                self.index = idx
                return dtype
        return supported[-1]

    def advance(self) -> Optional[torch.dtype]:
        if self.manual_override:
            return None
        next_index = self.index + 1
        if next_index >= len(self.ladder):
            return None
        self.index = next_index
        return self.current()

    def allow_fallback(self) -> bool:
        return not self.manual_override and self.index < len(self.ladder) - 1


def _detect_oom_exception() -> type[BaseException]:
    try:
        return torch.cuda.OutOfMemoryError  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return RuntimeError


def _normalize_device_name(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.strip().lower()


def _device_has_native_bf16(name: Optional[str], cc_major: Optional[int]) -> bool:
    normalized = _normalize_device_name(name)
    if normalized:
        for fragment in _NATIVE_BF16_NAME_FRAGMENTS:
            if fragment in normalized:
                return True
    if cc_major is not None and cc_major >= 9:
        # Hopper/Blackwell class or newer.
        return True
    return False


def _probe_hardware() -> HardwareProbe:
    probe = HardwareProbe()
    try:
        probe.cuda_available = torch.cuda.is_available()
        probe.cuda_device_count = torch.cuda.device_count() if probe.cuda_available else 0
    except Exception:  # pragma: no cover
        probe.cuda_available = False
        probe.cuda_device_count = 0

    props = None

    if probe.cuda_available:
        try:
            current_index = torch.cuda.current_device()
        except Exception:  # pragma: no cover
            current_index = 0
        try:
            props = torch.cuda.get_device_properties(current_index)
            probe.cuda_device_name = getattr(props, "name", None)
            probe.cuda_cc_major = getattr(props, "major", None)
            probe.cuda_cc_minor = getattr(props, "minor", None)
        except Exception:  # pragma: no cover
            props = None

    try:
        probe.mps_available = bool(torch.backends.mps.is_available())
    except Exception:  # pragma: no cover
        probe.mps_available = False

    try:  # pragma: no cover
        probe.xpu_available = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())  # type: ignore[attr-defined]
    except Exception:
        probe.xpu_available = False

    probe.directml_available = bool(os.getenv("DIRECTML_PATH"))  # best effort

    try:
        mem_total = getattr(props, "total_memory", None)
        if mem_total is None and probe.cuda_available:
            fallback_props = torch.cuda.get_device_properties(0)
            mem_total = getattr(fallback_props, "total_memory", None)
        if mem_total is not None:
            probe.total_vram_mb = int(mem_total // (1024 * 1024))
    except Exception:  # pragma: no cover
        probe.total_vram_mb = None

    try:
        import psutil  # type: ignore

        probe.total_ram_mb = int(psutil.virtual_memory().total // (1024 * 1024))
    except Exception:  # pragma: no cover
        probe.total_ram_mb = None

    try:
        probe.bf16_support = bool(torch.cuda.is_bf16_supported()) if probe.cuda_available else False
    except Exception:  # pragma: no cover
        probe.bf16_support = False

    if probe.cuda_available:
        probe.native_bf16 = bool(
            probe.bf16_support and _device_has_native_bf16(probe.cuda_device_name, probe.cuda_cc_major)
        )

    probe.fp8_support = False
    if probe.cuda_available:
        try:
            cc = torch.cuda.get_device_properties(torch.cuda.current_device()).major
            probe.fp8_support = cc >= 9
        except Exception:  # pragma: no cover
            probe.fp8_support = False

    try:
        import xformers  # type: ignore

        probe.xformers_available = True
        probe.xformers_version = getattr(getattr(xformers, "version", None), "__version__", None)
    except Exception:
        probe.xformers_available = False
        probe.xformers_version = None

    return probe


@dataclass
class _LoadedModelRecord:
    """Internal bookkeeping for loaded models."""

    model: object
    loader: object | None
    base_module: torch.nn.Module | None
    load_device: torch.device
    offload_device: torch.device
    storage_dtype: torch.dtype
    residency_mode: ResidencyMode = ResidencyMode.FULL
    streamed_runtime: StreamedCoreRuntime | None = None
    streamed_footprint: StreamedFootprint | None = None
    last_streamed_snapshot: StreamedResidencySnapshot | None = None
    inclusive_memory: int = 0
    exclusive_memory: int = 0
    model_accelerated: bool = False
    load_telemetry_device: str | None = None
    load_alloc_delta_bytes: int | None = None
    load_reserved_delta_bytes: int | None = None
    load_free_delta_bytes: int | None = None
    load_alloc_after_bytes: int | None = None
    load_reserved_after_bytes: int | None = None
    load_free_after_bytes: int | None = None
    load_total_after_bytes: int | None = None

    def __hash__(self) -> int:
        return hash(id(self.model))

    @staticmethod
    def _identity_candidates(other: object) -> tuple[object, ...]:
        attrs = (
            "patcher",
            "loader",
            "wrapped",
            "target",
            "model",
            "module",
            "inner",
            "inner_model",
            "base",
            "_base",
            "first_stage_model",
        )
        stack: List[object] = [other]
        seen: Set[int] = set()
        candidates: List[object] = []
        while stack:
            current = stack.pop()
            if current is None:
                continue
            ident = id(current)
            if ident in seen:
                continue
            seen.add(ident)
            candidates.append(current)
            for attr in attrs:
                if not hasattr(current, attr):
                    continue
                try:
                    value = getattr(current, attr)
                except Exception:
                    continue
                if value is not None and value is not current:
                    stack.append(value)
        return tuple(candidates)

    def matches(self, other: object) -> bool:
        identity_ids = {id(candidate) for candidate in (self.model, self.loader, self.base_module) if candidate is not None}
        if not identity_ids:
            return False
        for candidate in self._identity_candidates(other):
            if id(candidate) in identity_ids:
                return True
        return False


class CodexMemoryManager:
    """Central service coordinating device/dtype selection and model lifecycle."""

    def __init__(
        self,
        config: RuntimeMemoryConfig,
        *,
        probe: HardwareProbe | None = None,
    ) -> None:
        self._config = config
        self._probe = probe or _probe_hardware()
        self._oom_exception: type[BaseException] = _detect_oom_exception()
        self._loaded_models: MutableSequence[_LoadedModelRecord] = []
        self._signal_empty_cache: bool = False
        self._vae_always_tiled: bool = False
        self._cpu_device = torch.device(DeviceBackend.CPU.value)
        self._primary_device: torch.device = self._resolve_primary_device()
        self._mount_device: torch.device = self._device_from_backend(
            config.mount_device_backend,
            fallback=self._primary_device,
        )
        self._offload_device: torch.device = self._device_from_backend(
            config.offload_device_backend,
            fallback=self._primary_device,
        )
        self._attention = config.attention
        self._swap = config.swap
        self._budgets = config.budgets
        self._precision_states: Dict[DeviceRole, _PrecisionState] = {}

        self._log_probe()
        self._apply_deterministic_mode()
        self._configure_attention()
        self._initialize_precision_states()

    # --------------------------------------------------------------------- setup helpers
    def _log_probe(self) -> None:
        logger.debug("hardware probe: %s", self._probe.to_dict())

    def _apply_deterministic_mode(self) -> None:
        if self._config.deterministic_algorithms:
            logger.info("Enabling deterministic torch algorithms (warn_only=True).")
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _configure_attention(self) -> None:
        if self._attention.backend == AttentionBackend.PYTORCH and self._probe.cuda_available:
            try:
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_flash_sdp(bool(self._attention.enable_flash))
                torch.backends.cuda.enable_mem_efficient_sdp(bool(self._attention.enable_mem_efficient))
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Failed to configure PyTorch SDP backends from runtime attention config.") from exc
        self._log_attention_runtime()

    @staticmethod
    def _torch_sdp_flag(getter_name: str) -> bool | None:
        getter = getattr(torch.backends.cuda, getter_name, None)
        if getter is None or not callable(getter):
            return None
        try:
            return bool(getter())
        except Exception:
            return None

    def _sdpa_policy_label(self) -> str:
        if self._attention.backend != AttentionBackend.PYTORCH:
            return "n/a"
        if self._attention.enable_flash and self._attention.enable_mem_efficient:
            return "auto"
        if self._attention.enable_flash:
            return "flash_only"
        if self._attention.enable_mem_efficient:
            return "mem_efficient_only"
        return "math_only"

    def _log_attention_runtime(self) -> None:
        runtime_flash: bool | None = None
        runtime_mem_efficient: bool | None = None
        runtime_math: bool | None = None
        if self._probe.cuda_available:
            runtime_flash = self._torch_sdp_flag("flash_sdp_enabled")
            runtime_mem_efficient = self._torch_sdp_flag("mem_efficient_sdp_enabled")
            runtime_math = self._torch_sdp_flag("math_sdp_enabled")

        logger.info(
            "[memory] attention runtime | backend=%s sdpa_policy=%s sdpa_enable_flash=%s "
            "sdpa_enable_mem_efficient=%s sdpa_enable_math=%s sdpa_runtime_flash=%s "
            "sdpa_runtime_mem_efficient=%s sdpa_runtime_math=%s xformers_available=%s "
            "xformers_enabled=%s",
            self._attention.backend.value,
            self._sdpa_policy_label(),
            bool(self._attention.enable_flash) if self._attention.backend == AttentionBackend.PYTORCH else False,
            bool(self._attention.enable_mem_efficient) if self._attention.backend == AttentionBackend.PYTORCH else False,
            self._attention.backend == AttentionBackend.PYTORCH,
            runtime_flash,
            runtime_mem_efficient,
            runtime_math,
            self._probe.xformers_available,
            self.xformers_enabled(),
        )

    def _initialize_precision_states(self) -> None:
        self._precision_states.clear()
        for role in DeviceRole:
            ladder, manual = self._precision_policy_for(role)
            if not ladder:
                continue
            state = _PrecisionState(role=role, ladder=tuple(ladder), manual_override=manual)
            self._precision_states[role] = state
            ladder_display = " -> ".join(self._dtype_label(dt) for dt in state.ladder)
            logger.debug(
                "precision ladder[%s]=%s manual=%s",
                role.value,
                ladder_display,
                manual,
            )

    def _precision_policy_for(self, role: DeviceRole) -> Tuple[List[torch.dtype], bool]:
        flags = self._config.precision
        manual = False
        ladder: List[torch.dtype] = []

        global_forced = torch.float16 if flags.all_fp16 else None

        if role == DeviceRole.VAE:
            if flags.vae_fp32:
                ladder = [torch.float32]
                manual = True
            elif flags.vae_fp16:
                ladder = [torch.float16]
                manual = True
            elif flags.vae_bf16:
                ladder = [torch.bfloat16]
                manual = True
            elif global_forced is not None:
                ladder = [global_forced]
                manual = True
            else:
                ladder = self._auto_ladder_vae()
        elif role == DeviceRole.CORE:
            if flags.core_fp16:
                ladder = [torch.float16]
                manual = True
            elif flags.core_bf16:
                ladder = [torch.bfloat16]
                manual = True
            elif flags.core_fp8_e4m3fn or flags.core_fp8_e5m2:
                ladder = [torch.float16]
                manual = True
            elif global_forced is not None:
                ladder = [global_forced]
                manual = True
            else:
                ladder = self._auto_ladder_core()
        elif role == DeviceRole.TEXT_ENCODER:
            if flags.clip_fp32:
                ladder = [torch.float32]
                manual = True
            elif flags.clip_fp16:
                ladder = [torch.float16]
                manual = True
            elif flags.clip_bf16:
                ladder = [torch.bfloat16]
                manual = True
            elif flags.clip_fp8_e4m3fn or flags.clip_fp8_e5m2:
                ladder = [torch.float16]
                manual = True
            elif global_forced is not None:
                ladder = [global_forced]
                manual = True
            else:
                ladder = self._auto_ladder_text_encoder()
        else:
            ladder = [torch.float32]
            manual = True

        policy = self._config.component_policy(role)
        if policy.forced_dtype:
            try:
                forced = getattr(torch, policy.forced_dtype)
            except AttributeError as exc:
                raise MemoryConfigurationError(
                    f"Unsupported dtype '{policy.forced_dtype}' for {role.value}."
                ) from exc
            ladder = [forced]
            manual = True

        return ladder, manual

    def _auto_ladder_vae(self) -> List[torch.dtype]:
        ladder = [torch.float16, torch.float32]
        if self._probe.native_bf16:
            ladder.insert(0, torch.bfloat16)
        return ladder

    def _auto_ladder_core(self) -> List[torch.dtype]:
        if self._probe.native_bf16:
            return [torch.bfloat16, torch.float16]
        return [torch.float16]

    def _auto_ladder_text_encoder(self) -> List[torch.dtype]:
        if self._probe.native_bf16:
            return [torch.bfloat16, torch.float16]
        return [torch.float16]

    @staticmethod
    def _dtype_label(dtype: torch.dtype) -> str:
        text = str(dtype)
        return text.split(".")[-1]

    # --------------------------------------------------------------------- device/dtype
    def _resolve_cuda_device(self) -> torch.device:
        if not self._probe.cuda_available:
            raise MemoryConfigurationError("CUDA requested but not available.")
        if self._config.gpu_device_id is not None:
            return torch.device(DeviceBackend.CUDA.value, int(self._config.gpu_device_id))
        primary_device = getattr(self, "_primary_device", None)
        if isinstance(primary_device, torch.device):
            if primary_device.type == DeviceBackend.CUDA.value and primary_device.index is not None:
                return torch.device(DeviceBackend.CUDA.value, int(primary_device.index))
        return torch.device(DeviceBackend.CUDA.value, torch.cuda.current_device())

    def _resolve_primary_device(self) -> torch.device:
        cfg = self._config
        backend = cfg.device_backend
        probe = self._probe

        def choose_cuda() -> torch.device:
            return self._resolve_cuda_device()

        if backend == DeviceBackend.AUTO:
            if probe.cuda_available:
                device = choose_cuda()
                logger.info(
                    "Device AUTO selected CUDA device %s (%s)",
                    device,
                    probe.cuda_device_name or "unknown",
                )
                return device
            # No CUDA/XPU/MPS/DirectML: fall back to CPU instead of aborting.
            logger.warning(
                "Device AUTO fallback to CPU (no GPU/accelerator detected). "
                "Configure diffusion device to CPU in the Web UI to silence this warning."
            )
            self._config.device_backend = DeviceBackend.CPU
            return self._cpu_device
        if backend == DeviceBackend.CUDA:
            return choose_cuda()
        if backend == DeviceBackend.MPS:
            if not probe.mps_available:
                raise MemoryConfigurationError("MPS backend requested but not available.")
            return torch.device(DeviceBackend.MPS.value)
        if backend == DeviceBackend.XPU:
            if not probe.xpu_available:
                raise MemoryConfigurationError("XPU backend requested but not available.")
            return torch.device(DeviceBackend.XPU.value)
        if backend == DeviceBackend.DIRECTML:
            if not cfg.allow_directml:
                raise MemoryConfigurationError("DirectML backend disabled by configuration.")
            return torch.device("dml")
        return self._cpu_device

    def _device_from_backend(
        self,
        backend: DeviceBackend,
        *,
        fallback: torch.device | None = None,
    ) -> torch.device:
        if backend == DeviceBackend.AUTO:
            return fallback or self._primary_device
        if backend == DeviceBackend.CUDA:
            return self._resolve_cuda_device()
        if backend == DeviceBackend.MPS:
            if not self._probe.mps_available:
                raise MemoryConfigurationError("MPS backend requested but not available.")
            return torch.device(DeviceBackend.MPS.value)
        if backend == DeviceBackend.XPU:
            if not self._probe.xpu_available:
                raise MemoryConfigurationError("XPU backend requested but not available.")
            return torch.device(DeviceBackend.XPU.value)
        if backend == DeviceBackend.DIRECTML:
            if not self._config.allow_directml:
                raise MemoryConfigurationError("DirectML requested but disabled.")
            return torch.device("dml")
        return self._cpu_device

    # --------------------------------------------------------------------- public accessors
    @property
    def config(self) -> RuntimeMemoryConfig:
        """Return the active runtime memory configuration (treat as read-only)."""
        return self._config

    @property
    def oom_exception(self) -> type[BaseException]:
        return self._oom_exception

    @property
    def cpu_device(self) -> torch.device:
        return self._cpu_device

    @property
    def hardware_probe(self) -> HardwareProbe:
        """Expose a read-only view of the hardware probe used for this manager."""
        return self._probe

    @property
    def signal_empty_cache(self) -> bool:
        return self._signal_empty_cache

    @signal_empty_cache.setter
    def signal_empty_cache(self, value: bool) -> None:
        self._signal_empty_cache = bool(value)

    @property
    def vae_always_tiled(self) -> bool:
        return self._vae_always_tiled

    @vae_always_tiled.setter
    def vae_always_tiled(self, value: bool) -> None:
        self._vae_always_tiled = bool(value)

    def primary_device(self) -> torch.device:
        return self._primary_device

    def mount_device(self) -> torch.device:
        return self._mount_device

    def offload_device(self) -> torch.device:
        return self._offload_device

    def get_device(self, role: DeviceRole) -> torch.device:
        policy = self._config.component_policy(role)
        backend = policy.preferred_backend
        if backend == DeviceBackend.AUTO:
            return self._mount_device
        return self._device_from_backend(backend, fallback=self._mount_device)

    def get_offload_device(self, role: DeviceRole) -> torch.device:
        policy = self._config.component_policy(role)
        if not policy.allow_offload or self._swap.policy == SwapPolicy.NEVER:
            return self.get_device(role)
        return self._offload_device

    def dtype_for_role(
        self,
        role: DeviceRole,
        *,
        supported: Sequence[torch.dtype] = (torch.float16, torch.bfloat16, torch.float32),
        native_dtype: torch.dtype | None = None,
    ) -> torch.dtype:
        if not supported:
            raise MemoryConfigurationError(f"No supported dtypes passed for {role.value}.")

        policy = self._config.component_policy(role)
        device = self.get_device(role)
        if policy.forced_dtype:
            try:
                forced_dtype = getattr(torch, policy.forced_dtype)
            except AttributeError as exc:
                raise MemoryConfigurationError(f"Unsupported dtype '{policy.forced_dtype}' for {role.value}.") from exc
            if device.type == DeviceBackend.CPU.value and torch.float32 in supported and forced_dtype != torch.float32:
                return torch.float32
            if forced_dtype in supported:
                return forced_dtype
            return supported[-1]

        if native_dtype is not None:
            if device.type == DeviceBackend.CPU.value and torch.float32 in supported and native_dtype != torch.float32:
                raise MemoryConfigurationError(
                    f"{role.value} weights are {native_dtype} but CPU execution requires float32. "
                    f"Set an explicit {role.value} storage dtype override (fp32)."
                )
            if native_dtype in supported:
                return native_dtype
            raise MemoryConfigurationError(
                f"{role.value} weights are {native_dtype} but this dtype is not supported on {device} "
                f"(supported={supported}). Set an explicit {role.value} storage dtype override."
            )

        if device.type == DeviceBackend.CPU.value:
            if torch.float32 in supported:
                return torch.float32
            return supported[-1]
        state = self._precision_states.get(role)
        if state is None:
            return supported[0]
        return state.select(supported)

    def compute_dtype_for_role(
        self,
        role: DeviceRole,
        *,
        supported: Sequence[torch.dtype] = (torch.float16, torch.bfloat16, torch.float32),
        storage_dtype: torch.dtype | None = None,
    ) -> torch.dtype:
        """Return the preferred compute dtype for a component role.

        This is distinct from `dtype_for_role` (storage dtype). The compute dtype controls
        activation precision during forward passes when `allow_manual_cast` is enabled.

        Default policy:
        - CORE/TEXT_ENCODER compute in fp16 on accelerator devices.
        - VAE compute defaults to storage dtype on accelerator devices.
        - CPU execution uses fp32 (fail-loud guard against implicit reduced precision).
        - Other roles compute in the selected storage dtype unless overridden.
        """

        storage_dtype = storage_dtype or self.dtype_for_role(role, supported=supported)
        policy = self._config.component_policy(role)
        device = self.get_device(role)

        raw_forced = getattr(policy, "forced_compute_dtype", None)
        if raw_forced:
            try:
                forced = getattr(torch, raw_forced)
            except AttributeError as exc:
                raise MemoryConfigurationError(
                    f"Unsupported compute dtype '{raw_forced}' for {role.value}."
                ) from exc
            if device.type == DeviceBackend.CPU.value and torch.float32 in supported and forced != torch.float32:
                return torch.float32
            if forced not in supported:
                raise MemoryConfigurationError(
                    f"Compute dtype '{raw_forced}' for {role.value} is not supported on {device} (supported={supported})."
                )
            compute = forced
        else:
            if device.type == DeviceBackend.CPU.value:
                compute = torch.float32
            elif role in (DeviceRole.CORE, DeviceRole.TEXT_ENCODER):
                if torch.float16 in supported:
                    compute = torch.float16
                else:
                    compute = storage_dtype
            elif role == DeviceRole.VAE:
                compute = storage_dtype
            else:
                compute = storage_dtype

        if compute != storage_dtype and not policy.allow_manual_cast:
            raise MemoryConfigurationError(
                f"{role.value} compute dtype={compute} requires allow_manual_cast=True (storage dtype={storage_dtype})."
            )
        return compute

    def current_precision(self, role: DeviceRole) -> Optional[torch.dtype]:
        state = self._precision_states.get(role)
        if not state:
            return None
        return state.current()

    def allow_precision_fallback(self, role: DeviceRole) -> bool:
        state = self._precision_states.get(role)
        return bool(state and state.allow_fallback())

    def precision_hint(self, role: DeviceRole) -> str:
        return _PRECISION_HINTS.get(role, "set manual precision for the component")

    def report_precision_failure(self, role: DeviceRole, *, location: str, reason: str) -> Optional[torch.dtype]:
        state = self._precision_states.get(role)
        device = None
        try:
            device = self.get_device(role)
        except Exception:
            device = None

        if state is None:
            logger.error(
                "Precision failure for %s at %s (reason=%s) but no ladder is configured.",
                role.value,
                location,
                reason,
            )
            return None

        current = state.current()
        if not state.allow_fallback():
            hint = self.precision_hint(role)
            logger.error(
                "Precision fallback exhausted for %s on %s (dtype=%s). Reason: %s. Manual action required: %s",
                role.value,
                device,
                self._dtype_label(current),
                reason,
                hint,
            )
            return None

        next_dtype = state.advance()
        if next_dtype is None:
            hint = self.precision_hint(role)
            logger.error(
                "Precision fallback unavailable for %s on %s (dtype=%s). Reason: %s. Manual action required: %s",
                role.value,
                device,
                self._dtype_label(current),
                reason,
                hint,
            )
            return None

        logger.warning(
            "Precision fallback triggered for %s at %s on %s: %s -> %s (%s)",
            role.value,
            location,
            device,
            self._dtype_label(current),
            self._dtype_label(next_dtype),
            reason,
        )
        mark_fallback_used()
        return next_dtype

    # --------------------------------------------------------------------- tensor helpers
    def module_size(
        self,
        module,
        *,
        exclude_device: torch.device | None = None,
        include_device: torch.device | None = None,
        return_split: bool = False,
    ) -> int | Tuple[int, int, int]:
        module_mem = 0
        weight_mem = 0

        state_dict = module.state_dict()
        for key, tensor in state_dict.items():
            if exclude_device is not None and tensor.device == exclude_device:
                continue
            if include_device is not None and tensor.device != include_device:
                continue

            element_size = tensor.element_size()
            module_mem += tensor.nelement() * element_size
            if return_split and key.endswith(("weight", "bias")):
                weight_mem += tensor.nelement() * element_size

        if return_split:
            return module_mem, weight_mem, module_mem - weight_mem
        return module_mem

    def cast_to_device(self, tensor: torch.Tensor, device: torch.device, dtype: torch.dtype | None, *, copy: bool = False) -> torch.Tensor:
        target_dtype = dtype or tensor.dtype
        if tensor.device == device and tensor.dtype == target_dtype and not copy:
            return tensor
        if copy:
            tensor = tensor.clone()
        return tensor.to(device=device, dtype=target_dtype)

    # --------------------------------------------------------------------- memory metrics
    def get_free_memory(self, device: torch.device | None = None, *, return_torch_stats: bool = False) -> int | Tuple[int, int]:
        device = device or self._primary_device

        if device.type == DeviceBackend.CPU.value:
            import psutil  # type: ignore

            virtual = psutil.virtual_memory()
            total = virtual.available
            return (total, total) if return_torch_stats else total

        if device.type == DeviceBackend.CUDA.value:
            stats = torch.cuda.memory_stats(device)
            torch_reserved = stats.get("reserved_bytes.all.current", 0)
            free, total = torch.cuda.mem_get_info(device)
            if return_torch_stats:
                return free, max(torch_reserved, free)
            return free

        if device.type == "xpu":
            stats = torch.xpu.memory_stats(device)  # type: ignore[attr-defined]
            torch_reserved = stats.get("reserved_bytes.all.current", 0)
            try:
                free, total = torch.xpu.mem_get_info(device)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                free, total = 0, 0
            if return_torch_stats:
                return free, max(torch_reserved, free)
            return free

        if return_torch_stats:
            return 0, 0
        return 0

    def minimum_inference_memory(self) -> int:
        return max(self._budgets.minimum_inference_mb * 1024 * 1024, 0)

    def memory_snapshot(self) -> Dict[str, object]:
        """Return a JSON-friendly snapshot of current memory usage and managed models.

        The snapshot is intentionally shallow and avoids side effects so it can be
        called from diagnostics endpoints without perturbing runtime behaviour.
        """
        device = self._primary_device

        # Hardware/probe info
        try:
            probe_dict: Dict[str, object] = self._probe.to_dict()
        except Exception:  # pragma: no cover - extremely defensive
            probe_dict = {}

        # Torch-level stats (best-effort; only populated when supported)
        torch_stats = self._read_device_memory_counters_bytes(device) or {}

        # Managed model records
        models: List[Dict[str, object]] = []
        total_inclusive = 0
        total_exclusive = 0
        total_streamed_host = 0
        total_streamed_peak = 0
        total_streamed_current = 0

        for record in self._loaded_models:
            try:
                module = record.base_module or self._extract_module(record.loader or record.model)
            except Exception:
                module = None

            if module is not None:
                module_name = module.__class__.__name__
            else:
                module_name = type(record.model).__name__

            streamed_footprint = record.streamed_footprint
            streamed_snapshot = None
            if record.residency_mode is ResidencyMode.STREAMED:
                streamed_snapshot = self._verify_streamed_snapshot(record)
                if streamed_footprint is None:
                    raise MemoryLoadError(
                        f"Missing streamed footprint for {self._record_label(record)}."
                    )

            models.append(
                {
                    "module": module_name,
                    "load_device": str(record.load_device),
                    "offload_device": str(record.offload_device),
                    "storage_dtype": str(record.storage_dtype).replace("torch.", ""),
                    "residency_mode": record.residency_mode.value,
                    "inclusive_bytes": int(record.inclusive_memory),
                    "exclusive_bytes": int(record.exclusive_memory),
                    "accelerated": bool(record.model_accelerated),
                    "streamed_host_bytes": (
                        streamed_footprint.total_host_bytes
                        if streamed_footprint is not None
                        else None
                    ),
                    "streamed_peak_device_bytes": (
                        streamed_footprint.peak_device_bytes
                        if streamed_footprint is not None
                        else None
                    ),
                    "streamed_phase": (
                        streamed_snapshot.phase.value
                        if streamed_snapshot is not None
                        else None
                    ),
                    "streamed_storage_device": (
                        str(streamed_snapshot.storage_device)
                        if streamed_snapshot is not None
                        else None
                    ),
                    "streamed_compute_device": (
                        str(streamed_snapshot.compute_device)
                        if streamed_snapshot is not None
                        else None
                    ),
                    "streamed_static_resident_bytes": (
                        streamed_snapshot.static_resident_bytes
                        if streamed_snapshot is not None
                        else None
                    ),
                    "streamed_resident_segment_indices": (
                        list(streamed_snapshot.resident_segment_indices)
                        if streamed_snapshot is not None
                        else None
                    ),
                    "streamed_resident_segment_bytes": (
                        streamed_snapshot.resident_segment_bytes
                        if streamed_snapshot is not None
                        else None
                    ),
                    "streamed_current_device_bytes": (
                        streamed_snapshot.current_device_bytes
                        if streamed_snapshot is not None
                        else None
                    ),
                    "load_telemetry_device": record.load_telemetry_device,
                    "load_alloc_delta_bytes": record.load_alloc_delta_bytes,
                    "load_reserved_delta_bytes": record.load_reserved_delta_bytes,
                    "load_free_delta_bytes": record.load_free_delta_bytes,
                    "load_alloc_after_bytes": record.load_alloc_after_bytes,
                    "load_reserved_after_bytes": record.load_reserved_after_bytes,
                    "load_free_after_bytes": record.load_free_after_bytes,
                    "load_total_after_bytes": record.load_total_after_bytes,
                }
            )
            total_inclusive += int(record.inclusive_memory)
            total_exclusive += int(record.exclusive_memory)
            if streamed_footprint is not None and streamed_snapshot is not None:
                total_streamed_host += streamed_footprint.total_host_bytes
                total_streamed_peak += streamed_footprint.peak_device_bytes
                total_streamed_current += streamed_snapshot.current_device_bytes

        budgets = {
            "minimum_inference_mb": int(self._budgets.minimum_inference_mb),
            "hard_reservation_mb": int(self._budgets.hard_reservation_mb),
            "safety_margin_mb": int(self._budgets.safety_margin_mb),
        }
        attention = {
            "backend": self._attention.backend.value,
            "sdpa_policy": self._sdpa_policy_label(),
            "force_upcast": bool(self._attention.force_upcast),
            "enable_flash": bool(self._attention.enable_flash),
            "enable_mem_efficient": bool(self._attention.enable_mem_efficient),
            "pytorch_sdp_enabled": self._attention.backend == AttentionBackend.PYTORCH,
        }

        device_backend = getattr(self._config.device_backend, "value", str(self._config.device_backend))

        return {
            "device_backend": device_backend,
            "primary_device": str(device),
            "probe": probe_dict,
            "budgets": budgets,
            "attention": attention,
            "torch": torch_stats,
            "models": models,
            "totals": {
                "models_inclusive_bytes": total_inclusive,
                "models_exclusive_bytes": total_exclusive,
                "streamed_host_bytes": total_streamed_host,
                "streamed_peak_device_bytes": total_streamed_peak,
                "streamed_current_device_bytes": total_streamed_current,
            },
        }

    # --------------------------------------------------------------------- dtype helpers
    def should_use_fp16(
        self,
        *,
        device: torch.device | None = None,
        model_params: int = 0,
        prioritize_performance: bool = True,
        manual_cast: bool = False,
    ) -> bool:
        device = device or self._primary_device
        if device.type == DeviceBackend.CPU.value:
            return False
        if manual_cast:
            return True
        policy = self._config.component_policy(DeviceRole.CORE)
        if policy.forced_dtype and policy.forced_dtype != "float16":
            return False
        if not prioritize_performance:
            return False
        return True

    def should_use_bf16(
        self,
        *,
        device: torch.device | None = None,
        model_params: int = 0,
        prioritize_performance: bool = True,
        manual_cast: bool = False,
    ) -> bool:
        if not self._probe.bf16_support:
            return False
        if manual_cast:
            return True
        if not prioritize_performance:
            return False
        return True

    # --------------------------------------------------------------------- attention helpers
    def xformers_enabled(self) -> bool:
        if self._config.disable_xformers:
            return False
        return self._probe.xformers_available and self._attention.backend == AttentionBackend.XFORMERS

    def xformers_enabled_vae(self) -> bool:
        return self.xformers_enabled() and self._config.enable_xformers_vae

    def force_upcast_attention_dtype(self) -> torch.dtype:
        if self._attention.force_upcast:
            return torch.float32
        return torch.float16

    def pytorch_attention_enabled(self) -> bool:
        return self._attention.backend == AttentionBackend.PYTORCH

    def _memory_debug_enabled(self) -> bool:
        return os.environ.get("CODEX_MEMORY_DEBUG", "0").strip() == "1" or logger.isEnabledFor(logging.DEBUG)

    # --------------------------------------------------------------------- cache helpers
    def soft_empty_cache(self, force: bool = False) -> None:
        if self._primary_device.type == DeviceBackend.CUDA.value:
            try:
                torch.cuda.empty_cache()
            except self._oom_exception as exc:
                logger.debug("Suppressed CUDA empty_cache OOM during soft_empty_cache: %s", exc)
            except Exception as exc:
                logger.debug("Suppressed CUDA empty_cache failure during soft_empty_cache: %s", exc)
        if force:
            self._signal_empty_cache = False

    @guarded_load_entrypoint(
        action="runtime.memory.manager.unload_all_models",
        allowed_stages=(
            LoadAuthorityStage.LOAD,
            LoadAuthorityStage.MATERIALIZE,
            LoadAuthorityStage.UNLOAD,
            LoadAuthorityStage.RELOAD,
            LoadAuthorityStage.CLEANUP,
        ),
    )
    def unload_all_models(self) -> None:
        failures: List[Tuple[_LoadedModelRecord, Exception]] = []
        for record in list(self._loaded_models):
            try:
                self._unload_record(record, avoid_model_moving=True, reason="unload_all_models")
                self._remove_loaded_record(
                    record,
                    reason="unload_all_models",
                    verify_model_absent=False,
                )
            except Exception as exc:
                failures.append((record, exc))
                continue

        if failures:
            details = self._format_record_failures(failures)
            raise MemoryLoadError(
                f"Failed to unload {len(failures)} model(s) during unload_all_models: {details}"
            ) from failures[0][1]
        if self._loaded_models:
            raise MemoryLoadError(
                "unload_all_models completed with lingering residency records: "
                f"{len(self._loaded_models)} record(s) still registered."
            )

        logger.info("Unloaded all models, cache cleared.")

    def loaded_models(self) -> Tuple[_LoadedModelRecord, ...]:
        return tuple(self._loaded_models)

    def is_model_loaded(self, model: object) -> bool:
        """Return True when `model` is currently registered as loaded.

        This is a public wrapper around the internal loaded-model registry and is
        safe to call from engines/use-cases to implement stage-scoped load/unload
        logic without poking private methods.
        """
        return self._find_loaded_model(model) is not None

    @guarded_load_entrypoint(
        action="runtime.memory.manager.unload_model_clones",
        allowed_stages=(
            LoadAuthorityStage.LOAD,
            LoadAuthorityStage.MATERIALIZE,
            LoadAuthorityStage.UNLOAD,
            LoadAuthorityStage.RELOAD,
            LoadAuthorityStage.CLEANUP,
        ),
    )
    def unload_model_clones(self, model: object) -> None:
        if not hasattr(model, "is_clone"):
            return
        predicate = getattr(model, "is_clone")
        if not callable(predicate):
            return
        for record in list(self._loaded_models):
            try:
                matches = predicate(record.model)
            except Exception:  # pragma: no cover
                matches = False
            if matches:
                self._unload_record(record, avoid_model_moving=True, reason="unload_model_clones")
                self._remove_loaded_record(
                    record,
                    reason="unload_model_clones",
                    verify_model_absent=False,
                )

        lingering = 0
        for record in self._loaded_models:
            try:
                if predicate(record.model):
                    lingering += 1
            except Exception:  # pragma: no cover
                continue
        if lingering:
            raise MemoryLoadError(
                "unload_model_clones completed with lingering clone residency: "
                f"{lingering} clone record(s) still registered."
            )

    @guarded_load_entrypoint(
        action="runtime.memory.manager.unload_model",
        allowed_stages=(
            LoadAuthorityStage.LOAD,
            LoadAuthorityStage.MATERIALIZE,
            LoadAuthorityStage.UNLOAD,
            LoadAuthorityStage.RELOAD,
            LoadAuthorityStage.CLEANUP,
        ),
    )
    def unload_model(
        self,
        model: object,
        *,
        source: str = "runtime.memory.manager.unload_model",
        stage: str | None = None,
        component_hint: str | None = None,
        event_reason: str | None = None,
    ) -> None:
        record = self._find_loaded_model(model)
        if record is None:
            if smart_offload_enabled():
                action_memory_before = self._smart_offload_memory_fields(prefix="memory_before")
                action_memory_after = self._smart_offload_memory_fields(prefix="memory_after")
                log_smart_offload_action(
                    SmartOffloadAction.UNLOAD_NOOP,
                    source=source,
                    stage=stage,
                    component=component_hint or type(model).__name__,
                    reason=event_reason,
                    **action_memory_before,
                    **action_memory_after,
                )
            return
        self._unload_record(
            record,
            avoid_model_moving=False,
            reason="unload_model",
            event_source=source,
            event_stage=stage,
            event_component=component_hint,
            event_reason=event_reason,
        )
        self._remove_loaded_record(
            record,
            reason="unload_model",
            verify_model_absent=True,
        )
        if self._primary_device.type == DeviceBackend.CUDA.value:
            try:
                torch.cuda.empty_cache()
            except self._oom_exception as exc:
                logger.debug("Suppressed CUDA empty_cache OOM during unload_model: %s", exc)
            except Exception as exc:
                logger.debug("Suppressed CUDA empty_cache failure during unload_model: %s", exc)

    # --------------------------------------------------------------------- load/unload
    @guarded_load_entrypoint(
        action="runtime.memory.manager.load_models",
        allowed_stages=(
            LoadAuthorityStage.LOAD,
            LoadAuthorityStage.MATERIALIZE,
            LoadAuthorityStage.RELOAD,
        ),
    )
    def load_models(
        self,
        models: Sequence[object],
        *,
        memory_required: int = 0,
        hard_memory_preservation: int = 0,
        source: str = "runtime.memory.manager.load_models",
        stage: str | None = None,
        component_hint: str | None = None,
        event_reason: str | None = None,
    ) -> None:
        if not models:
            return
        if type(memory_required) is not int or memory_required < 0:
            raise MemoryLoadError(
                f"memory_required must be an exact non-negative int, got {memory_required!r}."
            )
        if type(hard_memory_preservation) is not int or hard_memory_preservation < 0:
            raise MemoryLoadError(
                "hard_memory_preservation must be an exact non-negative int, "
                f"got {hard_memory_preservation!r}."
            )

        execution_start = time.perf_counter()
        inference_reserve = max(self.minimum_inference_memory(), memory_required)
        models_to_load: List[_LoadedModelRecord] = []
        already_loaded: List[_LoadedModelRecord] = []
        memory_debug_enabled = self._memory_debug_enabled()

        if memory_debug_enabled and self._primary_device.type == DeviceBackend.CUDA.value:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self._primary_device)
            allocated = torch.cuda.memory_allocated(self._primary_device)
            reserved = torch.cuda.memory_reserved(self._primary_device)
            logger.info(
                "[memory-debug] BEFORE load_models: free=%.2f GB, allocated=%.2f GB, reserved=%.2f GB, total=%.2f GB",
                free_bytes / 1e9, allocated / 1e9, reserved / 1e9, total_bytes / 1e9,
            )
            logger.info(
                "[memory-debug] Currently loaded models (%d): %s",
                len(self._loaded_models),
                [r.base_module.__class__.__name__ if r.base_module else "?" for r in self._loaded_models],
            )

        for model in models:
            record = self._find_loaded_model(model)
            if record:
                if not self._record_alias_exists(record, already_loaded):
                    already_loaded.append(record)
                continue
            candidate = self._create_record(model)
            if self._record_alias_exists(candidate, models_to_load):
                continue
            models_to_load.append(candidate)

        if models_to_load:
            self._allocate_memory(
                models_to_load,
                inference_reserve,
                hard_memory_preservation,
                already_loaded,
            )
            loaded_this_call: List[_LoadedModelRecord] = []
            current: _LoadedModelRecord | None = None
            try:
                for current in models_to_load:
                    self._load_record(
                        current,
                        event_source=source,
                        event_stage=stage,
                        event_component=component_hint,
                        event_reason=event_reason,
                    )
                    self._loaded_models.insert(0, current)
                    loaded_this_call.append(current)
            except BaseException as load_error:
                rollback_targets: List[_LoadedModelRecord] = []
                if current is not None:
                    rollback_targets.append(current)
                rollback_targets.extend(reversed(loaded_this_call))

                rollback_failures = self._rollback_partially_loaded_records(rollback_targets)
                for record, rollback_error in rollback_failures:
                    load_error.add_note(
                        "Memory manager secondary rollback failure after the primary load error: "
                        f"model={self._record_label(record)} "
                        f"error={type(rollback_error).__name__}: {rollback_error}"
                    )
                raise
        else:
            self._cleanup_for_loaded_models(
                already_loaded,
                inference_reserve,
                hard_memory_preservation,
            )

        elapsed = time.perf_counter() - execution_start
        logger.info("Model load completed (%d new, %d existing) in %.2fs.", len(models_to_load), len(already_loaded), elapsed)

    def load_model(
        self,
        model: object,
        *,
        memory_required: int = 0,
        hard_memory_preservation: int = 0,
        source: str = "runtime.memory.manager.load_model",
        stage: str | None = None,
        component_hint: str | None = None,
        event_reason: str | None = None,
    ) -> None:
        self.load_models(
            [model],
            memory_required=memory_required,
            hard_memory_preservation=hard_memory_preservation,
            source=source,
            stage=stage,
            component_hint=component_hint,
            event_reason=event_reason,
        )

    def free_memory(
        self,
        memory_required: int,
        *,
        device: torch.device | None = None,
        keep_loaded: Sequence[_LoadedModelRecord] | None = None,
        free_all: bool = False,
    ) -> None:
        device = device or self._primary_device
        keep_loaded = keep_loaded or ()

        def _targets_requested_device(record: _LoadedModelRecord) -> bool:
            if record.load_device.type != device.type:
                return False
            if record.load_device.index is None or device.index is None:
                return True
            return int(record.load_device.index) == int(device.index)

        release_candidates = [
            record
            for record in self._loaded_models
            if record not in keep_loaded and _targets_requested_device(record)
        ]
        released = 0

        if not free_all and int(self.get_free_memory(device)) >= memory_required:
            logger.debug("Memory target already satisfied on %s (required=%d).", device, memory_required)
            return

        for record in release_candidates:
            releasable_bytes = record.exclusive_memory
            if record.residency_mode is ResidencyMode.STREAMED:
                releasable_bytes = self._verify_streamed_snapshot(record).current_device_bytes
            self._unload_record(record, avoid_model_moving=free_all, reason="free_memory")
            self._remove_loaded_record(
                record,
                reason="free_memory",
                verify_model_absent=False,
            )
            released += releasable_bytes
            if device.type == DeviceBackend.CUDA.value:
                torch.cuda.empty_cache()
            if not free_all and int(self.get_free_memory(device)) >= memory_required:
                break

        if device.type == DeviceBackend.CUDA.value and not release_candidates:
            torch.cuda.empty_cache()
        logger.debug("Freed %d bytes on %s (required=%d).", released, device, memory_required)

    # --------------------------------------------------------------------- internals
    def _find_loaded_model(self, model: object) -> _LoadedModelRecord | None:
        for record in self._loaded_models:
            if record.matches(model):
                return record
        return None

    def _remove_loaded_record(
        self,
        record: _LoadedModelRecord,
        *,
        reason: str,
        verify_model_absent: bool,
    ) -> None:
        try:
            self._loaded_models.remove(record)
        except ValueError as exc:
            raise MemoryLoadError(
                f"Failed to unregister model record during {reason}: {self._record_label(record)} was not in loaded registry."
            ) from exc

        if any(existing is record for existing in self._loaded_models):
            raise MemoryLoadError(
                f"Failed to unregister model record during {reason}: duplicate identity for "
                f"{self._record_label(record)} remains in loaded registry."
            )

        if verify_model_absent:
            lingering = self._find_loaded_model(record.model)
            if lingering is not None:
                raise MemoryLoadError(
                    f"Post-unload residency verification failed during {reason}: "
                    f"{self._record_label(record)} is still registered as loaded."
                )

    @staticmethod
    def _record_alias_exists(record: _LoadedModelRecord, records: Sequence[_LoadedModelRecord]) -> bool:
        for existing in records:
            if record.matches(existing.model) or existing.matches(record.model):
                return True
        return False

    @staticmethod
    def _record_label(record: _LoadedModelRecord) -> str:
        module = record.base_module
        if module is not None:
            return module.__class__.__name__
        return type(record.model).__name__

    @staticmethod
    def _require_streamed_runtime(record: _LoadedModelRecord) -> tuple[StreamedCoreRuntime, StreamedFootprint]:
        runtime = record.streamed_runtime
        footprint = record.streamed_footprint
        if record.residency_mode is not ResidencyMode.STREAMED or runtime is None or footprint is None:
            raise MemoryLoadError(
                f"Streamed residency record is incomplete for {CodexMemoryManager._record_label(record)}."
            )
        return runtime, footprint

    def _verify_streamed_snapshot(
        self,
        record: _LoadedModelRecord,
        *,
        expected_phase: StreamedResidencyPhase | None = None,
    ) -> StreamedResidencySnapshot:
        runtime, footprint = self._require_streamed_runtime(record)
        snapshot = runtime.verify_residency()
        if not isinstance(snapshot, StreamedResidencySnapshot):
            raise MemoryLoadError(
                f"Streamed runtime for {self._record_label(record)} returned "
                f"{type(snapshot).__name__} instead of StreamedResidencySnapshot."
            )
        if snapshot.storage_device != record.offload_device:
            raise MemoryLoadError(
                f"Streamed runtime for {self._record_label(record)} reported storage_device="
                f"{snapshot.storage_device}, expected {record.offload_device}."
            )
        if snapshot.compute_device != record.load_device:
            raise MemoryLoadError(
                f"Streamed runtime for {self._record_label(record)} reported compute_device="
                f"{snapshot.compute_device}, expected {record.load_device}."
            )
        if expected_phase is not None and snapshot.phase is not expected_phase:
            raise MemoryLoadError(
                f"Streamed runtime for {self._record_label(record)} reported phase="
                f"{snapshot.phase.value}, expected {expected_phase.value}."
            )
        resident_count = len(snapshot.resident_segment_indices)
        if resident_count > footprint.max_resident_segments:
            raise MemoryLoadError(
                f"Streamed runtime for {self._record_label(record)} exceeds its resident segment limit: "
                f"resident={resident_count}, maximum={footprint.max_resident_segments}."
            )
        if snapshot.resident_segment_bytes > footprint.max_segment_bytes * resident_count:
            raise MemoryLoadError(
                f"Streamed runtime for {self._record_label(record)} exceeds its segment byte contract: "
                f"resident_bytes={snapshot.resident_segment_bytes}, resident_count={resident_count}, "
                f"max_segment_bytes={footprint.max_segment_bytes}."
            )
        if snapshot.phase is not StreamedResidencyPhase.OFFLOADED:
            if snapshot.static_resident_bytes != footprint.static_device_bytes:
                raise MemoryLoadError(
                    f"Streamed runtime for {self._record_label(record)} reported static_resident_bytes="
                    f"{snapshot.static_resident_bytes}, expected {footprint.static_device_bytes}."
                )
            packed_limit = (
                footprint.static_device_bytes
                + (footprint.max_segment_bytes * footprint.max_resident_segments)
            )
            if snapshot.current_device_bytes > packed_limit:
                raise MemoryLoadError(
                    f"Streamed runtime for {self._record_label(record)} exceeds its packed device limit: "
                    f"current={snapshot.current_device_bytes}, limit={packed_limit}."
                )
        record.last_streamed_snapshot = snapshot
        return snapshot

    @staticmethod
    def _bytes_to_mib(value: int) -> float:
        return round(float(max(0, int(value))) / (1024.0 * 1024.0), 2)

    @staticmethod
    def _signed_bytes_to_mib(value: int) -> float:
        return round(float(int(value)) / (1024.0 * 1024.0), 2)

    @staticmethod
    def _first_module_device(module: torch.nn.Module) -> torch.device | None:
        first_parameter = next(module.parameters(), None)
        if first_parameter is not None:
            return first_parameter.device
        first_buffer = next(module.buffers(), None)
        if first_buffer is not None:
            return first_buffer.device
        return None

    def _read_device_memory_counters_bytes(self, device: torch.device) -> Dict[str, int] | None:
        try:
            if device.type == DeviceBackend.CUDA.value:
                stats = torch.cuda.memory_stats(device)
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                allocated_bytes = int(stats.get("allocated_bytes.all.current", 0))
                reserved_bytes = int(stats.get("reserved_bytes.all.current", 0))
            elif device.type == "xpu":
                stats = torch.xpu.memory_stats(device)  # type: ignore[attr-defined]
                free_bytes, total_bytes = torch.xpu.mem_get_info(device)  # type: ignore[attr-defined]
                allocated_bytes = int(stats.get("allocated_bytes.all.current", 0))
                reserved_bytes = int(stats.get("reserved_bytes.all.current", 0))
            elif device.type == "mps":
                allocated_bytes = int(torch.mps.current_allocated_memory())
                total_bytes = int(torch.mps.driver_allocated_memory())
                reserved_bytes = allocated_bytes
                free_bytes = max(total_bytes - allocated_bytes, 0)
            else:
                return None
        except Exception:
            return None

        return {
            "allocated_bytes": int(allocated_bytes),
            "reserved_bytes": int(reserved_bytes),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    def _smart_offload_memory_fields(self, *, prefix: str) -> Dict[str, object]:
        """Best-effort primary-device memory telemetry for smart-offload logs."""

        device = self._primary_device
        fields: Dict[str, object] = {
            f"{prefix}_device": str(device),
        }

        counters = self._read_device_memory_counters_bytes(device)
        if counters is None:
            return fields

        stats = (
            ("alloc", counters["allocated_bytes"]),
            ("reserved", counters["reserved_bytes"]),
            ("free", counters["free_bytes"]),
            ("total", counters["total_bytes"]),
        )
        for key, bytes_value in stats:
            fields[f"{prefix}_{key}_mb"] = self._bytes_to_mib(bytes_value)

        return fields

    def _format_record_failures(self, failures: Sequence[Tuple[_LoadedModelRecord, BaseException]]) -> str:
        return "; ".join(f"{self._record_label(record)}: {exc}" for record, exc in failures)

    def _rollback_partially_loaded_records(
        self,
        records: Sequence[_LoadedModelRecord],
    ) -> List[Tuple[_LoadedModelRecord, BaseException]]:
        failures: List[Tuple[_LoadedModelRecord, BaseException]] = []
        seen: Set[int] = set()

        for record in records:
            ident = id(record)
            if ident in seen:
                continue
            seen.add(ident)
            try:
                self._unload_record(record, reason="rollback_after_load_failure")
            except BaseException as exc:
                failures.append((record, exc))
                continue
            if not any(loaded_record is record for loaded_record in self._loaded_models):
                continue
            try:
                self._remove_loaded_record(
                    record,
                    reason="rollback_after_load_failure",
                    verify_model_absent=False,
                )
            except BaseException as exc:
                failures.append((record, exc))

        return failures

    def _verify_unload_residency(
        self,
        *,
        record: _LoadedModelRecord,
        loader: object,
        module: torch.nn.Module | None,
        pre_unload_device: torch.device | None,
        target_device: torch.device | None,
        avoid_model_moving: bool,
    ) -> None:
        if record.model_accelerated:
            raise MemoryLoadError(
                f"Post-unload residency verification failed for {self._record_label(record)}: "
                "record is still marked accelerated."
            )

        if record.residency_mode is ResidencyMode.STREAMED:
            self._verify_streamed_snapshot(
                record,
                expected_phase=StreamedResidencyPhase.OFFLOADED,
            )
            return

        if avoid_model_moving:
            return

        if target_device is None:
            raise MemoryLoadError(
                f"Post-unload residency verification failed for {self._record_label(record)}: "
                "target_device is undefined."
            )

        observed_device = self._first_module_device(module) if module is not None else None
        if observed_device is None:
            loader_device = getattr(loader, "current_device", None)
            if loader_device is not None:
                try:
                    observed_device = torch.device(loader_device)
                except Exception as exc:
                    raise MemoryLoadError(
                        f"Post-unload residency verification failed for {self._record_label(record)}: "
                        f"invalid loader.current_device={loader_device!r}."
                    ) from exc

        if observed_device is None:
            raise MemoryLoadError(
                f"Post-unload residency verification failed for {self._record_label(record)}: "
                f"unable to resolve device after unload (pre={pre_unload_device}, target={target_device})."
            )

        if observed_device != target_device:
            raise MemoryLoadError(
                f"Post-unload residency verification failed for {self._record_label(record)}: "
                f"observed_device={observed_device} does not match target_device={target_device} "
                f"(pre={pre_unload_device})."
            )

    # ------------------------------------------------------------------ load target helpers
    def _extract_module(self, obj: object, *, visited: Optional[Set[int]] = None) -> torch.nn.Module | None:
        visited = visited or set()
        stack: List[object] = [obj]
        while stack:
            current = stack.pop()
            if current is None:
                continue
            ident = id(current)
            if ident in visited:
                continue
            visited.add(ident)
            if isinstance(current, torch.nn.Module):
                return current
            for attr in (
                "model",
                "module",
                "_module",
                "target",
                "wrapped",
                "_wrapped",
                "inner",
                "inner_model",
                "base",
                "_base",
                "first_stage_model",
            ):
                if not hasattr(current, attr):
                    continue
                try:
                    value = getattr(current, attr)
                except Exception:
                    continue
                if value is not None and value is not current:
                    stack.append(value)
        return None

    def _resolve_loader(self, model: object) -> tuple[object, torch.nn.Module]:
        candidate_attrs = (
            None,
            "patcher",
            "loader",
            "wrapped",
            "target",
            "model",
            "module",
        )

        for attr in candidate_attrs:
            candidate = model if attr is None else getattr(model, attr, None)
            if candidate is None:
                continue
            loader, module = self._classify_loader(candidate)
            if loader is not None and module is not None:
                return loader, module

        module = self._extract_module(model)
        if module is not None:
            return model, module
        raise MemoryLoadError(
            f"Unable to resolve a loadable module from {type(model).__name__}; "
            "expected a ModelPatcher, torch.nn.Module, or wrapper exposing one."
        )

    def _classify_loader(self, candidate: object) -> tuple[object | None, torch.nn.Module | None]:
        module = self._extract_module(candidate)
        if module is None:
            return None, None
        if hasattr(candidate, "codex_patch_model") or hasattr(candidate, "model_patches_to"):
            return candidate, module
        if isinstance(candidate, torch.nn.Module):
            return candidate, module
        return None, module

    def _create_record(self, model: object) -> _LoadedModelRecord:
        loader, base_module = self._resolve_loader(model)
        load_device_source = loader if loader is not None else model
        raw_load_device = getattr(load_device_source, "load_device", getattr(model, "load_device", self._mount_device))
        try:
            load_device = torch.device(raw_load_device)
        except Exception as exc:
            raise MemoryLoadError(
                f"Invalid load_device for {type(model).__name__}: {raw_load_device!r}"
            ) from exc
        offload_device = self._offload_device
        storage_dtype_getter = getattr(load_device_source, "model_dtype", None)
        if not callable(storage_dtype_getter):
            storage_dtype_getter = getattr(model, "model_dtype", None)
        storage_dtype = storage_dtype_getter() if callable(storage_dtype_getter) else torch.float32
        raw_residency_mode = getattr(load_device_source, "residency_mode", ResidencyMode.FULL)
        if not isinstance(raw_residency_mode, ResidencyMode):
            raise MemoryLoadError(
                f"Invalid residency_mode for {type(model).__name__}: {raw_residency_mode!r}."
            )
        raw_streamed_runtime = getattr(load_device_source, "streamed_core_runtime", None)
        streamed_runtime: StreamedCoreRuntime | None = None
        streamed_footprint: StreamedFootprint | None = None
        if raw_streamed_runtime is not None:
            if not isinstance(raw_streamed_runtime, StreamedCoreRuntime):
                raise MemoryLoadError(
                    f"Invalid streamed_core_runtime for {type(model).__name__}: "
                    f"{type(raw_streamed_runtime).__name__}."
                )
            if raw_residency_mode is not ResidencyMode.STREAMED:
                raise MemoryLoadError(
                    f"{type(model).__name__} exposes a streamed runtime but residency_mode="
                    f"{raw_residency_mode.value}."
                )
            streamed_runtime = raw_streamed_runtime
            streamed_footprint = streamed_runtime.footprint
            if not isinstance(streamed_footprint, StreamedFootprint):
                raise MemoryLoadError(
                    f"Invalid streamed footprint for {type(model).__name__}: "
                    f"{type(streamed_footprint).__name__}."
                )
        elif raw_residency_mode is ResidencyMode.STREAMED:
            raise MemoryLoadError(
                f"{type(model).__name__} declares streamed residency without a streamed runtime."
            )

        record = _LoadedModelRecord(
            model=model,
            loader=loader,
            base_module=base_module,
            load_device=load_device,
            offload_device=offload_device,
            storage_dtype=storage_dtype,
            residency_mode=raw_residency_mode,
            streamed_runtime=streamed_runtime,
            streamed_footprint=streamed_footprint,
        )
        if streamed_footprint is not None:
            record.inclusive_memory = streamed_footprint.total_host_bytes
            record.exclusive_memory = streamed_footprint.peak_device_bytes
        return record

    @staticmethod
    def _module_compute_dtype(module: torch.nn.Module, *, fallback: torch.dtype | None = None) -> torch.dtype | None:
        try:
            dtype_attr = getattr(module, "computation_dtype", None)
            if callable(dtype_attr):
                resolved = dtype_attr()
            elif dtype_attr is not None:
                resolved = dtype_attr
            elif hasattr(module, "dtype"):
                resolved = getattr(module, "dtype")
            else:
                resolved = None
        except Exception:  # pragma: no cover
            resolved = None
        if isinstance(resolved, torch.dtype):
            return resolved
        return fallback

    def _load_record(
        self,
        record: _LoadedModelRecord,
        *,
        event_source: str,
        event_stage: str | None = None,
        event_component: str | None = None,
        event_reason: str | None = None,
    ) -> None:
        loader = record.loader or record.model
        module = record.base_module or self._extract_module(loader)
        if module is None:
            raise MemoryLoadError(
                f"Failed to resolve base module for {type(record.model).__name__}; cannot compute memory usage."
            )
        # Ensure cache of module for downstream accounting
        record.base_module = module
        target_name = module.__class__.__name__
        record.load_telemetry_device = str(record.load_device)
        record.load_alloc_delta_bytes = None
        record.load_reserved_delta_bytes = None
        record.load_free_delta_bytes = None
        record.load_alloc_after_bytes = None
        record.load_reserved_after_bytes = None
        record.load_free_after_bytes = None
        record.load_total_after_bytes = None

        before_load_counters: Dict[str, int] | None = None
        after_load_counters: Dict[str, int] | None = None
        action_memory_before: Dict[str, object] = {}
        if smart_offload_enabled():
            action_memory_before = self._smart_offload_memory_fields(prefix="memory_before")
        memory_debug_enabled = self._memory_debug_enabled()

        if memory_debug_enabled:
            if record.residency_mode is ResidencyMode.STREAMED:
                _, streamed_footprint = self._require_streamed_runtime(record)
                module_size_bytes = streamed_footprint.total_host_bytes
            else:
                module_size_bytes = self.module_size(module)
            logger.info(
                "[memory-debug] _load_record: module=%s mode=%s host_size=%.2f GB target_device=%s",
                target_name,
                record.residency_mode.value,
                module_size_bytes / 1e9,
                record.load_device,
            )

            try:
                first_param = next(module.parameters(), None)
                if first_param is not None:
                    logger.info(
                        "[memory-debug] module %s current device: %s dtype=%s",
                        target_name, first_param.device, first_param.dtype,
                    )
            except Exception:
                logger.debug("[memory-debug] failed to inspect module parameter state", exc_info=True)

        try:
            if smart_offload_enabled() and getattr(record.load_device, "type", "") == DeviceBackend.CUDA.value:
                torch.cuda.empty_cache()
                if memory_debug_enabled and self._primary_device.type == DeviceBackend.CUDA.value:
                    free_bytes, _ = torch.cuda.mem_get_info(self._primary_device)
                    logger.info("[memory-debug] AFTER empty_cache: free=%.2f GB", free_bytes / 1e9)

            if memory_debug_enabled:
                before_load_counters = self._read_device_memory_counters_bytes(record.load_device)

            compute_dtype = self._module_compute_dtype(module, fallback=record.storage_dtype)

            if hasattr(loader, "model_patches_to"):
                if memory_debug_enabled:
                    logger.info("[memory-debug] calling model_patches_to(%s)", record.load_device)
                loader.model_patches_to(record.load_device)
                if compute_dtype is not None:
                    if memory_debug_enabled:
                        logger.info("[memory-debug] calling model_patches_to(%s)", compute_dtype)
                    loader.model_patches_to(compute_dtype)
            elif hasattr(loader, "to"):
                if memory_debug_enabled:
                    logger.info("[memory-debug] calling loader.to(device=%s)", record.load_device)
                loader.to(device=record.load_device)
                if record.storage_dtype is not None:
                    loader.to(dtype=record.storage_dtype)
            if hasattr(loader, "codex_patch_model"):
                if memory_debug_enabled:
                    logger.info("[memory-debug] calling codex_patch_model(%s)", record.load_device)
                loader.codex_patch_model(record.load_device)
            elif record.residency_mode is ResidencyMode.STREAMED:
                raise MemoryLoadError(
                    f"Streamed model {target_name} requires a loader with codex_patch_model(...)."
                )
            if record.residency_mode is ResidencyMode.FULL and hasattr(loader, "current_device"):
                setattr(loader, "current_device", record.load_device)
            if record.residency_mode is ResidencyMode.STREAMED:
                _, streamed_footprint = self._require_streamed_runtime(record)
                streamed_snapshot = self._verify_streamed_snapshot(
                    record,
                    expected_phase=StreamedResidencyPhase.READY,
                )
                record.inclusive_memory = streamed_footprint.total_host_bytes
                record.exclusive_memory = streamed_footprint.peak_device_bytes
            else:
                streamed_footprint = None
                streamed_snapshot = None
                record.inclusive_memory = self.module_size(module)
                record.exclusive_memory = record.inclusive_memory
            record.model_accelerated = True
            if memory_debug_enabled and self._primary_device.type == DeviceBackend.CUDA.value:
                free_bytes, _ = torch.cuda.mem_get_info(self._primary_device)
                logger.info("[memory-debug] AFTER load %s: free=%.2f GB", target_name, free_bytes / 1e9)

            if memory_debug_enabled:
                after_load_counters = self._read_device_memory_counters_bytes(record.load_device)
                if after_load_counters is not None:
                    record.load_alloc_after_bytes = after_load_counters["allocated_bytes"]
                    record.load_reserved_after_bytes = after_load_counters["reserved_bytes"]
                    record.load_free_after_bytes = after_load_counters["free_bytes"]
                    record.load_total_after_bytes = after_load_counters["total_bytes"]

                if before_load_counters is not None and after_load_counters is not None:
                    record.load_alloc_delta_bytes = (
                        after_load_counters["allocated_bytes"] - before_load_counters["allocated_bytes"]
                    )
                    record.load_reserved_delta_bytes = (
                        after_load_counters["reserved_bytes"] - before_load_counters["reserved_bytes"]
                    )
                    record.load_free_delta_bytes = (
                        after_load_counters["free_bytes"] - before_load_counters["free_bytes"]
                    )

                if after_load_counters is not None:
                    alloc_delta_mib = "n/a"
                    reserved_delta_mib = "n/a"
                    free_delta_mib = "n/a"
                    if record.load_alloc_delta_bytes is not None:
                        alloc_delta_mib = f"{self._signed_bytes_to_mib(record.load_alloc_delta_bytes):.2f}"
                    if record.load_reserved_delta_bytes is not None:
                        reserved_delta_mib = f"{self._signed_bytes_to_mib(record.load_reserved_delta_bytes):.2f}"
                    if record.load_free_delta_bytes is not None:
                        free_delta_mib = f"{self._signed_bytes_to_mib(record.load_free_delta_bytes):.2f}"

                    logger.info(
                        "[memory-debug] ITEM load module=%s device=%s alloc_delta_mib=%s reserved_delta_mib=%s "
                        "free_delta_mib=%s alloc_after_mib=%.2f reserved_after_mib=%.2f free_after_mib=%.2f total_after_mib=%.2f",
                        target_name,
                        record.load_telemetry_device,
                        alloc_delta_mib,
                        reserved_delta_mib,
                        free_delta_mib,
                        self._bytes_to_mib(record.load_alloc_after_bytes or 0),
                        self._bytes_to_mib(record.load_reserved_after_bytes or 0),
                        self._bytes_to_mib(record.load_free_after_bytes or 0),
                        self._bytes_to_mib(record.load_total_after_bytes or 0),
                    )
                else:
                    logger.debug(
                        "[memory-debug] ITEM load telemetry unavailable module=%s device=%s",
                        target_name,
                        record.load_telemetry_device,
                    )

            if streamed_footprint is not None and streamed_snapshot is not None:
                logger.info(
                    "[memory] loaded %s mode=%s storage_device=%s compute_device=%s "
                    "storage_dtype=%s compute_dtype=%s host_bytes=%d peak_device_bytes=%d "
                    "current_device_bytes=%d phase=%s",
                    target_name,
                    record.residency_mode.value,
                    streamed_snapshot.storage_device,
                    streamed_snapshot.compute_device,
                    record.storage_dtype,
                    compute_dtype,
                    streamed_footprint.total_host_bytes,
                    streamed_footprint.peak_device_bytes,
                    streamed_snapshot.current_device_bytes,
                    streamed_snapshot.phase.value,
                )
            else:
                logger.info(
                    "[memory] loaded %s mode=%s device=%s storage=%s compute=%s mem=%d",
                    target_name,
                    record.residency_mode.value,
                    record.load_device,
                    record.storage_dtype,
                    compute_dtype,
                    record.inclusive_memory,
                )
            if smart_offload_enabled():
                action_memory_after = self._smart_offload_memory_fields(prefix="memory_after")
                log_smart_offload_action(
                    SmartOffloadAction.LOAD,
                    source=event_source,
                    stage=event_stage,
                    component=event_component or target_name,
                    reason=event_reason,
                    load_device=str(record.load_device),
                    offload_device=str(record.offload_device),
                    storage_dtype=str(record.storage_dtype),
                    compute_dtype=str(compute_dtype),
                    **action_memory_before,
                    **action_memory_after,
                )
        except self._oom_exception as exc:
            raise MemoryLoadError(f"OOM while loading {record.model}: {exc}") from exc
        except Exception as exc:
            raise MemoryLoadError(f"Failed to load model {record.model}: {exc}") from exc

    def _unload_record(
        self,
        record: _LoadedModelRecord,
        *,
        avoid_model_moving: bool = False,
        reason: str = "unknown",
        event_source: str | None = None,
        event_stage: str | None = None,
        event_component: str | None = None,
        event_reason: str | None = None,
    ) -> None:
        loader = record.loader or record.model
        module = record.base_module or self._extract_module(loader)
        if module is not None:
            record.base_module = module
        target_device: torch.device | None = None
        pre_unload_device = self._first_module_device(module) if module is not None else None
        action_memory_before: Dict[str, object] = {}
        if smart_offload_enabled():
            action_memory_before = self._smart_offload_memory_fields(prefix="memory_before")
        try:
            if not avoid_model_moving:
                target_device = self._offload_device
                if (
                    record.load_device.type == DeviceBackend.CPU.value
                    and target_device.type != DeviceBackend.CPU.value
                ):
                    raise MemoryLoadError(
                        f"Refusing unload for {self._record_label(record)}: "
                        f"load_device={record.load_device} with offload_device={target_device} "
                        "(unload must not move CPU-resident models onto an accelerator)."
                    )
                if (
                    record.load_device.type != DeviceBackend.CPU.value
                    and target_device == record.load_device
                ):
                    raise MemoryLoadError(
                        f"Refusing unload for {self._record_label(record)}: "
                        f"offload_device={target_device} matches load_device={record.load_device} "
                        "(Contract R requires real de-residency)."
                    )
                record.offload_device = target_device
            if hasattr(loader, "codex_unpatch_model"):
                loader.codex_unpatch_model(target_device)
            elif record.residency_mode is ResidencyMode.STREAMED:
                streamed_runtime, _ = self._require_streamed_runtime(record)
                streamed_runtime.offload_all()
            elif hasattr(loader, "to") and not avoid_model_moving:
                loader.to(target_device)
            record.model_accelerated = False
            self._verify_unload_residency(
                record=record,
                loader=loader,
                module=module,
                pre_unload_device=pre_unload_device,
                target_device=target_device,
                avoid_model_moving=avoid_model_moving,
            )
            logger.debug(
                "Unloaded model %s mode=%s (avoid_move=%s).",
                record.model,
                record.residency_mode.value,
                avoid_model_moving,
            )
            if smart_offload_enabled():
                action_memory_after = self._smart_offload_memory_fields(prefix="memory_after")
                log_smart_offload_action(
                    SmartOffloadAction.UNLOAD,
                    source=event_source or reason,
                    stage=event_stage,
                    component=event_component or self._record_label(record),
                    reason=event_reason,
                    load_device=str(record.load_device),
                    offload_device=str(record.offload_device),
                    target_device=str(target_device) if target_device is not None else "unknown",
                    avoid_model_moving=avoid_model_moving,
                    **action_memory_before,
                    **action_memory_after,
                )
        except Exception as exc:
            raise MemoryLoadError(
                f"Failed to unload model {self._record_label(record)} "
                f"(avoid_model_moving={avoid_model_moving}, target_device={target_device}): {exc}"
            ) from exc

    def _ensure_device_admission(
        self,
        *,
        device: torch.device,
        model_requirement_bytes: int,
        inference_reserve_bytes: int,
        hard_memory_preservation_bytes: int,
        keep_loaded: Sequence[_LoadedModelRecord],
    ) -> None:
        total_required_bytes = (
            model_requirement_bytes
            + inference_reserve_bytes
            + hard_memory_preservation_bytes
        )
        self.free_memory(
            total_required_bytes,
            device=device,
            keep_loaded=keep_loaded,
        )
        if device.type == DeviceBackend.CPU.value:
            return
        free_bytes = int(self.get_free_memory(device))
        if free_bytes < total_required_bytes:
            raise MemoryLoadError(
                "Insufficient device memory after eviction: "
                f"device={device}, free_bytes={free_bytes}, "
                f"model_requirement_bytes={model_requirement_bytes}, "
                f"inference_reserve_bytes={inference_reserve_bytes}, "
                f"hard_memory_preservation_bytes={hard_memory_preservation_bytes}, "
                f"total_required_bytes={total_required_bytes}."
            )

    def _cleanup_for_loaded_models(
        self,
        records: Sequence[_LoadedModelRecord],
        inference_reserve_bytes: int,
        hard_memory_preservation_bytes: int,
    ) -> None:
        devices = {record.load_device for record in records if record.load_device.type != DeviceBackend.CPU.value}
        for device in devices:
            self._ensure_device_admission(
                device=device,
                model_requirement_bytes=0,
                inference_reserve_bytes=inference_reserve_bytes,
                hard_memory_preservation_bytes=hard_memory_preservation_bytes,
                keep_loaded=records,
            )

    def _allocate_memory(
        self,
        records: Sequence[_LoadedModelRecord],
        inference_reserve_bytes: int,
        hard_memory_preservation_bytes: int,
        already_loaded: Sequence[_LoadedModelRecord],
    ) -> None:
        total_required: dict[torch.device, int] = {}
        for record in records:
            module = record.base_module or self._extract_module(record.loader or record.model)
            if module is None:
                raise MemoryLoadError(
                    f"Unable to resolve module for {type(record.model).__name__} while allocating memory."
                )
            record.base_module = module
            if record.residency_mode is ResidencyMode.STREAMED:
                _, footprint = self._require_streamed_runtime(record)
                record.inclusive_memory = footprint.total_host_bytes
                record.exclusive_memory = footprint.peak_device_bytes
                device_requirement = footprint.peak_device_bytes
            else:
                record.inclusive_memory = self.module_size(module)
                record.exclusive_memory = record.inclusive_memory
                device_requirement = record.inclusive_memory
            total_required[record.load_device] = total_required.get(record.load_device, 0) + device_requirement

        for device, requirement in total_required.items():
            self._ensure_device_admission(
                device=device,
                model_requirement_bytes=requirement,
                inference_reserve_bytes=inference_reserve_bytes,
                hard_memory_preservation_bytes=hard_memory_preservation_bytes,
                keep_loaded=already_loaded,
            )

    # --------------------------------------------------------------------- factory
    @classmethod
    def create(cls, config: RuntimeMemoryConfig, probe: HardwareProbe | None = None) -> "CodexMemoryManager":
        if probe is None:
            probe = _probe_hardware()
        if config.device_backend == DeviceBackend.CUDA and not probe.cuda_available:
            raise HardwareProbeError("CUDA backend requested but no CUDA device detected.")
        return cls(config, probe=probe)
__all__ = ["CodexMemoryManager"]

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima engine specification and runtime assembly (Cosmos Predict2 / Anima adapter).
Assembles a Codex-native runtime from the parsed Anima core bundle, validates sha-selected external assets
(Qwen3-0.6B text encoder + WanVAE-style VAE), and eagerly loads external text/vae/tokenizer components through canonical loaders.
Produces a denoiser patcher suitable for the canonical txt2img/img2img pipelines (Option A).
Sets Anima predictor defaults, including SIMPLE schedule mode selection for tail-downsample sigma-ladder parity.
Rejects explicit CORE storage/compute dtype splits before lazy-core or external-component materialization.

Symbols (top-level; keep in sync; no ghosts):
- `_torch_dtype_label` (function): Convert canonical torch dtypes into runtime metadata labels (`fp16`/`bf16`/`fp32`).
- `_predictor` (function): Build Anima predictor defaults (discrete-flow + tail-downsample SIMPLE schedule mode).
- `_load_external_text_encoder` (function): Load and validate an external Anima Qwen3-0.6B text-encoder asset.
- `_load_external_vae` (function): Load and validate Anima WAN VAE from external safetensors.
- `_require_external_asset_path` (function): Require non-empty external asset option values.
- `_require_existing_external_asset_path` (function): Require existing external asset files on disk.
- `_LazyAnimaCoreDenoiser` (class): Lazy Anima core wrapper that materializes `AnimaDiT` on first use.
- `AnimaTextPipelines` (dataclass): Text pipeline container (Qwen3 embeddings + offline T5 tokenizer).
- `AnimaEngineRuntime` (dataclass): Assembled runtime container (denoiser + VAE + text pipelines + patchers).
- `AnimaEngineSpec` (dataclass): Engine spec (family defaults + flow shift/multiplier overrides).
- `assemble_anima_runtime` (function): Assemble an Anima runtime from a `DiffusionModelBundle` and engine options.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from apps.backend.patchers.denoiser import DenoiserPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.patchers.base import ModelPatcher
from apps.backend.runtime.model_registry.family_runtime import FamilyRuntimeSpec, get_family_spec
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.memory.exceptions import MemoryConfigurationError
from apps.backend.runtime.sampling_adapters.prediction import (
    SIMPLE_SCHEDULE_MODE_TAIL_DOWNSAMPLE_SIGMAS,
    PredictionDiscreteFlow,
)

logger = get_backend_logger("backend.engines.anima.spec")


def _torch_dtype_label(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    raise ValueError(f"Unsupported torch dtype: {dtype!r}")


@dataclass(frozen=True, slots=True)
class AnimaTextPipelines:
    qwen3_text: Any
    t5_tokenizer: Any


@dataclass(slots=True)
class AnimaEngineRuntime:
    vae: VAE
    denoiser: DenoiserPatcher
    text: AnimaTextPipelines
    qwen: ModelPatcher
    device: torch.device = field(default_factory=lambda: memory_management.manager.mount_device())
    core_storage_dtype: str = "bf16"
    core_compute_dtype: str = "fp32"
    te_storage_dtype: str = "bf16"
    te_compute_dtype: str = "fp32"
    vae_storage_dtype: str = "bf16"
    vae_compute_dtype: str = "fp32"


@dataclass(frozen=True, slots=True)
class AnimaEngineSpec:
    name: str = "anima"
    family: ModelFamily = ModelFamily.ANIMA
    _flow_shift_override: float | None = field(default=None, repr=False)
    _flow_multiplier_override: float | None = field(default=None, repr=False)

    def _get_family_spec(self) -> FamilyRuntimeSpec:
        return get_family_spec(self.family)

    @property
    def flow_shift(self) -> float:
        spec = self._get_family_spec()
        if self._flow_shift_override is not None:
            return float(self._flow_shift_override)
        if spec.flow_shift is None:
            raise RuntimeError("AnimaEngineSpec.flow_shift missing: family runtime spec flow_shift is None.")
        return float(spec.flow_shift)

    @property
    def flow_multiplier(self) -> float:
        if self._flow_multiplier_override is not None:
            return float(self._flow_multiplier_override)
        # Source-of-truth: Anima signature extras declare multiplier=1.0; use-case/engine may override.
        return 1.0


def _predictor(*, spec: AnimaEngineSpec) -> PredictionDiscreteFlow:
    return PredictionDiscreteFlow(
        prediction_type="const",
        shift=float(spec.flow_shift),
        multiplier=float(spec.flow_multiplier),
        timesteps=1000,
        simple_schedule_mode=SIMPLE_SCHEDULE_MODE_TAIL_DOWNSAMPLE_SIGMAS,
    )


def _load_external_text_encoder(*, tenc_path: str, torch_dtype: torch.dtype, device: torch.device) -> object:
    from apps.backend.runtime.families.anima.text_encoder import load_anima_qwen3_06b_text_encoder

    return load_anima_qwen3_06b_text_encoder(tenc_path, torch_dtype=torch_dtype, device=device)


def _load_external_vae(*, vae_path: str, torch_dtype: torch.dtype, device: torch.device) -> object:
    from apps.backend.runtime.families.anima.wan_vae import load_wan_vae_from_safetensors

    return load_wan_vae_from_safetensors(vae_path, torch_dtype=torch_dtype, device=device)


def _require_external_asset_path(*, opts: Mapping[str, Any], key: str, label: str) -> str:
    raw = opts.get(key)
    if raw is None:
        raise ValueError(f"Anima requires an external {label} via `{key}` (sha-selected); missing.")
    if not isinstance(raw, str):
        raise TypeError(
            f"Anima option `{key}` must be a non-empty string path (sha-selected); got {type(raw).__name__}."
        )
    value = raw.strip()
    if not value:
        raise ValueError(f"Anima requires an external {label} via `{key}` (sha-selected); missing.")
    return value


def _require_existing_external_asset_path(*, raw_path: str, label: str) -> str:
    resolved = os.path.expanduser(raw_path)
    if not os.path.isfile(resolved):
        raise RuntimeError(f"Anima {label} path not found: {resolved}")
    return resolved


class _LazyAnimaCoreDenoiser(torch.nn.Module):
    """Lazy wrapper that defers strict Anima transformer load until first model use."""

    def __init__(
        self,
        *,
        transformer_state_dict: Mapping[str, torch.Tensor],
        storage_dtype: torch.dtype,
        computation_dtype: torch.dtype,
        load_device: torch.device,
        offload_device: torch.device,
        initial_device: torch.device,
    ) -> None:
        super().__init__()
        if not isinstance(transformer_state_dict, ABCMapping):
            raise TypeError(
                "Anima bundle 'transformer' component must be a mapping of tensors; "
                f"got {type(transformer_state_dict).__name__}."
            )
        self.storage_dtype = storage_dtype
        self.computation_dtype = computation_dtype
        self.load_device = load_device
        self.offload_device = offload_device
        self.initial_device = initial_device
        self._materialize_lock = threading.Lock()
        self._transformer_state_dict: Mapping[str, torch.Tensor] | None = transformer_state_dict
        self._materialized_core: torch.nn.Module | None = None
        # Prevent `hasattr(model, "lora_loader")` probes from materializing the core during patcher init.
        self.lora_loader = None

    @property
    def dtype(self) -> torch.dtype:
        core = self._materialized_core
        if core is not None and hasattr(core, "dtype"):
            return core.dtype  # type: ignore[return-value]
        return self.storage_dtype

    def _ensure_materialized(self, *, trigger: str) -> torch.nn.Module:
        core = self._materialized_core
        if core is not None:
            return core

        with self._materialize_lock:
            core = self._materialized_core
            if core is not None:
                return core

            state_dict = self._transformer_state_dict
            if state_dict is None:
                raise RuntimeError(
                    "Anima lazy core materialization invariant violated: transformer state_dict is unavailable."
                )

            from apps.backend.runtime.families.anima.loader import load_anima_dit_from_state_dict

            logger.info("Materializing Anima core transformer lazily (trigger=%s).", trigger)
            try:
                core = load_anima_dit_from_state_dict(
                    state_dict,  # type: ignore[arg-type]
                    device=self.initial_device,
                    dtype=self.storage_dtype if isinstance(self.storage_dtype, torch.dtype) else None,
                )
            except Exception as exc:
                raise RuntimeError(f"Anima lazy core materialization failed during '{trigger}': {exc}") from exc

            core.storage_dtype = self.storage_dtype
            core.computation_dtype = self.computation_dtype
            core.load_device = self.load_device
            core.initial_device = self.initial_device
            core.offload_device = self.offload_device
            self._materialized_core = core
            # Drop retained state dict once load succeeds to avoid duplicate memory retention.
            self._transformer_state_dict = None
            return core

    def materialize(self) -> torch.nn.Module:
        return self._ensure_materialized(trigger="materialize")

    def _apply(self, fn):
        core = self._ensure_materialized(trigger="_apply")
        core._apply(fn)
        return self

    def forward(self, *args, **kwargs):
        core = self._ensure_materialized(trigger="forward")
        return core(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            if name.startswith("_"):
                raise exc
            core = self._ensure_materialized(trigger=f"attr:{name}")
            return getattr(core, name)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if args:
            if len(args) > 3:
                raise TypeError(f"state_dict expected at most 3 positional args; got {len(args)}")
            if len(args) >= 1:
                destination = args[0]
            if len(args) >= 2:
                prefix = args[1]
            if len(args) == 3:
                keep_vars = args[2]

        core = self._materialized_core
        if core is not None:
            return core.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()  # type: ignore[attr-defined]

        state_dict = self._transformer_state_dict
        if state_dict is None:
            raise RuntimeError(
                "Anima lazy state_dict invariant violated: transformer state_dict is unavailable before materialization."
            )
        for key, tensor in state_dict.items():
            dst_key = f"{prefix}{key}"
            if isinstance(tensor, torch.Tensor) and not keep_vars:
                destination[dst_key] = tensor.detach()
            else:
                destination[dst_key] = tensor
        return destination

    def load_state_dict(self, *args, **kwargs):
        core = self._ensure_materialized(trigger="load_state_dict")
        return core.load_state_dict(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        core = self._ensure_materialized(trigger="parameters")
        return core.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        core = self._ensure_materialized(trigger="named_parameters")
        return core.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)

    def buffers(self, recurse: bool = True):
        core = self._ensure_materialized(trigger="buffers")
        return core.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        core = self._ensure_materialized(trigger="named_buffers")
        return core.named_buffers(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)


def assemble_anima_runtime(
    *,
    spec: AnimaEngineSpec,
    estimated_config: Any,
    codex_components: Mapping[str, object],
    engine_options: Mapping[str, Any] | None = None,
) -> AnimaEngineRuntime:
    """Assemble Anima runtime from a minimal (state-dict) bundle plus sha-selected external assets."""
    opts = dict(engine_options or {})
    vae_path = _require_external_asset_path(opts=opts, key="vae_path", label="VAE")
    tenc_path = _require_external_asset_path(opts=opts, key="tenc_path", label="text encoder")

    # Core transformer component is provided in native checkpoint keyspace (`net.*`) and resolved lazily by the loader.
    transformer_sd = codex_components.get("transformer")
    if transformer_sd is None:
        raise RuntimeError("Anima bundle missing required component 'transformer' (state dict).")

    vae_path = _require_existing_external_asset_path(raw_path=vae_path, label="VAE")
    tenc_path = _require_existing_external_asset_path(raw_path=tenc_path, label="text encoder")

    native_core_dtype: torch.dtype | None = None
    try:
        first_key = next(iter(transformer_sd.keys()))  # type: ignore[attr-defined]
        first_tensor = transformer_sd[first_key]  # type: ignore[index]
        if isinstance(first_tensor, torch.Tensor):
            native_core_dtype = first_tensor.dtype
    except Exception:
        native_core_dtype = None

    core_storage = memory_management.manager.dtype_for_role(DeviceRole.CORE, native_dtype=native_core_dtype)
    core_policy = memory_management.manager.config.component_policy(DeviceRole.CORE)
    requested_compute_name = core_policy.forced_compute_dtype
    if requested_compute_name:
        try:
            requested_compute = getattr(torch, requested_compute_name)
        except AttributeError as exc:
            raise MemoryConfigurationError(
                "Anima CORE compute dtype configuration is unsupported: "
                f"storage={_torch_dtype_label(core_storage)} "
                f"requested_compute={requested_compute_name!r}."
            ) from exc
        if requested_compute != core_storage:
            raise MemoryConfigurationError(
                "Anima CORE storage/compute dtype mismatch: "
                f"storage={_torch_dtype_label(core_storage)} "
                f"requested_compute={_torch_dtype_label(requested_compute)}. "
                "Anima ordinary modules require matching storage and compute dtypes."
            )

    requested_compute_label = requested_compute_name or "auto"
    try:
        core_compute = memory_management.manager.compute_dtype_for_role(
            DeviceRole.CORE,
            supported=(core_storage,),
            storage_dtype=core_storage,
        )
    except MemoryConfigurationError as exc:
        raise MemoryConfigurationError(
            "Anima CORE dtype resolution failed: "
            f"storage={_torch_dtype_label(core_storage)} "
            f"requested_compute={requested_compute_label!r}."
        ) from exc
    if core_compute != core_storage:
        raise MemoryConfigurationError(
            "Anima CORE dtype invariant violated after resolution: "
            f"storage={_torch_dtype_label(core_storage)} "
            f"compute={_torch_dtype_label(core_compute)}."
        )
    load_device = memory_management.manager.get_device(DeviceRole.CORE)
    offload_device = memory_management.manager.get_offload_device(DeviceRole.CORE)
    initial_device = offload_device

    model = _LazyAnimaCoreDenoiser(
        transformer_state_dict=transformer_sd,  # type: ignore[arg-type]
        storage_dtype=core_storage,
        computation_dtype=core_compute,
        load_device=load_device,
        offload_device=offload_device,
        initial_device=initial_device,
    )

    denoiser = DenoiserPatcher.from_model(
        model=model,
        diffusers_scheduler=None,
        predictor=_predictor(spec=spec),
        config=estimated_config,
    )

    te_load_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
    te_offload_device = memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER)
    te_storage = memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER)
    te_compute = memory_management.manager.compute_dtype_for_role(DeviceRole.TEXT_ENCODER, storage_dtype=te_storage)
    qwen_text_encoder = _load_external_text_encoder(
        tenc_path=tenc_path,
        torch_dtype=te_storage,
        device=te_load_device,
    )
    from apps.backend.runtime.families.anima.text_encoder import (
        AnimaQwenTextEncoder,
        AnimaQwenTextProcessingEngine,
        load_anima_t5_tokenizer,
        resolve_anima_qwen_max_length,
    )

    if not isinstance(qwen_text_encoder, AnimaQwenTextEncoder):
        raise RuntimeError(
            "Anima external text encoder loader returned invalid wrapper type: "
            f"{type(qwen_text_encoder).__name__}."
        )
    qwen_model = getattr(qwen_text_encoder, "model", None)
    if not isinstance(qwen_model, torch.nn.Module):
        raise RuntimeError(
            "Anima external text encoder loader returned invalid `.model` contract: "
            f"{type(qwen_model).__name__}."
        )

    vae_load_device = memory_management.manager.get_device(DeviceRole.VAE)
    vae_storage = memory_management.manager.dtype_for_role(DeviceRole.VAE)
    vae_compute = memory_management.manager.compute_dtype_for_role(DeviceRole.VAE, storage_dtype=vae_storage)
    vae_model = _load_external_vae(vae_path=vae_path, torch_dtype=vae_storage, device=vae_load_device)
    if not isinstance(vae_model, torch.nn.Module):
        raise RuntimeError(
            "Anima external VAE loader returned invalid model type: "
            f"{type(vae_model).__name__}."
        )

    # Wrap VAE with shared patcher interface (encode/decode + normalization via family spec fallback).
    vae = VAE(model=vae_model, family=ModelFamily.ANIMA)

    # Text encoder patcher for memory management integration.
    qwen = ModelPatcher(
        qwen_model,
        load_device=te_load_device,
        offload_device=te_offload_device,
    )

    # Text pipelines: Qwen embeddings + offline T5 tokenizer.
    qwen_max_length = resolve_anima_qwen_max_length()
    text_engine = AnimaQwenTextProcessingEngine(qwen_text_encoder, max_length=qwen_max_length)
    t5_tokenizer = load_anima_t5_tokenizer()
    text_pipelines = AnimaTextPipelines(qwen3_text=text_engine, t5_tokenizer=t5_tokenizer)

    core_dev = memory_management.manager.get_device(DeviceRole.CORE)
    device = core_dev
    logger.debug(
        "Anima runtime assembled: device=%s core_storage=%s core_compute=%s te_storage=%s te_compute=%s vae_storage=%s vae_compute=%s",
        device,
        _torch_dtype_label(core_storage),
        _torch_dtype_label(core_compute),
        _torch_dtype_label(te_storage),
        _torch_dtype_label(te_compute),
        _torch_dtype_label(vae_storage),
        _torch_dtype_label(vae_compute),
    )

    return AnimaEngineRuntime(
        vae=vae,
        denoiser=denoiser,
        text=text_pipelines,
        qwen=qwen,
        device=device,
        core_storage_dtype=_torch_dtype_label(core_storage),
        core_compute_dtype=_torch_dtype_label(core_compute),
        te_storage_dtype=_torch_dtype_label(te_storage),
        te_compute_dtype=_torch_dtype_label(te_compute),
        vae_storage_dtype=_torch_dtype_label(vae_storage),
        vae_compute_dtype=_torch_dtype_label(vae_compute),
    )


ANIMA_SPEC = AnimaEngineSpec()

__all__ = [
    "ANIMA_SPEC",
    "AnimaEngineRuntime",
    "AnimaEngineSpec",
    "AnimaTextPipelines",
    "assemble_anima_runtime",
]

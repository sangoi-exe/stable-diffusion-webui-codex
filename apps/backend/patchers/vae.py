"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical VAE patcher for diffusers/native 2D and WAN-aware first stages, including family-aware latent normalization and memory-managed storage/compute precision.
Owns pristine `[0,1]` encode copy/normalization, device-local regular decode clamp, seeded posterior sampling, tiled context/crop stitching, progress, precision retry, and CPU fallback.
Preserves tiled `[0,1]` internal buffers and public BCHW `[-1,1]` decode output while failing loud on unsupported output or geometry contracts.

Symbols (top-level; keep in sync; no ghosts):
- `_tensor_stats` (function): Opt-in VAE tensor diagnostics for shape/dtype/device and basic statistics (`CODEX_VAE_TENSOR_STATS=1` plus DEBUG logging).
- `_unwrap_decode_output` (function): Normalizes diffusers decode outputs to a plain tensor (`DecoderOutput.sample` or passthrough).
- `_new_encode_generator` (function): Builds a device-local seeded generator for deterministic VAE posterior sampling when img2img requests supply an encode seed.
- `_unwrap_encode_output` (function): Normalizes encode outputs to a latent tensor, preferring an explicit owner-produced `.latents` tensor before distribution sampling.
- `_report_vae_progress` (function): Reports VAE encode/decode block progress into backend state for phase-aware streaming.
- `_NormalizingFirstStage` (class): Wrapper around a first-stage VAE that applies strict scalar/per-channel latent normalization (including optional shift semantics) and proxies encode/decode APIs.
- `VaeTileGeometry` (class import): Shared typed tile geometry contract consumed by tiled decode/encode helpers.
- `DEFAULT_VAE_DECODE_TILED_GEOMETRY` (constant import): Default decode tiled geometry for non-family-specific fallback paths.
- `resolve_vae_decode_tiled_geometry` (function import): Shared family-aware decode tiled geometry resolver.
- `iter_vae_tile_windows` (function import): Shared tile-window iterator with strict geometry/padding checks.
- `VAE` (class): ModelPatcher for VAEs; provides encode/decode APIs (optionally tiled), device/dtype placement, and fallback/normalization logic
  (includes nested helpers for memory-management and diffusers/WAN VAE compatibility).
"""

import contextlib
import gc
import logging
import math
import os

import torch

try:  # Optional import; diffusers may not be present in minimal environments
    from diffusers.models.autoencoder_kl import AutoencoderKL as DiffusersAutoencoderKL
except Exception:  # noqa: BLE001
    DiffusersAutoencoderKL = None

try:  # Optional; only needed to detect native LDM VAEs explicitly
    from apps.backend.runtime.common.vae_ldm import AutoencoderKL_LDM
except Exception:  # noqa: BLE001
    AutoencoderKL_LDM = None

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceBackend, DeviceRole
from apps.backend.runtime.memory.smart_offload import smart_fallback_enabled
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.common.vae_tiled import (
    DEFAULT_VAE_DECODE_TILED_GEOMETRY,
    VaeTileGeometry,
    iter_vae_tile_windows,
    resolve_vae_decode_tiled_geometry,
)
from apps.backend.runtime.logging import get_backend_logger
from .base import ModelPatcher
from .vae_normalization_policy import read_vae_config_field, resolve_vae_normalization_policy

logger = get_backend_logger(__name__)


def _tensor_stats(label: str, tensor: torch.Tensor) -> None:
    if os.environ.get("CODEX_VAE_TENSOR_STATS", "0").strip() != "1" or not logger.isEnabledFor(logging.DEBUG):
        return
    if tensor is None:
        logger.debug("[vae] %s: <none>", label)
        return
    with torch.no_grad():
        data = tensor.detach()
        stats_tensor = data
        logger.debug(
            "[vae] %s: shape=%s dtype=%s device=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
            label,
            tuple(data.shape),
            data.dtype,
            data.device,
            float(stats_tensor.min().item()),
            float(stats_tensor.max().item()),
            float(stats_tensor.mean().item()),
            float(stats_tensor.std(unbiased=False).item()),
        )


def _unwrap_decode_output(output):
    """Extract tensor from diffusers DecoderOutput or passthrough."""
    if hasattr(output, "sample"):
        sample = getattr(output, "sample")
        if torch.is_tensor(sample):
            return sample
    return output


def _new_encode_generator(*, encode_seed: int | None, device: torch.device | str) -> torch.Generator | None:
    if encode_seed is None:
        return None
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    generator_device = resolved_device if resolved_device.type in {"cpu", "cuda"} else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(encode_seed))
    return generator


def _unwrap_encode_output(output, *, generator: torch.Generator | None = None):
    """Extract the owner-produced latent tensor or sample a distribution output."""

    def _sample_output(sample_owner):
        if generator is None:
            return sample_owner.sample()
        try:
            return sample_owner.sample(generator=generator)
        except TypeError:
            try:
                return sample_owner.sample(generator)
            except TypeError as exc:
                raise RuntimeError("VAE encode output does not support generator-seeded sampling.") from exc
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("VAE encode output failed during generator-seeded sampling.") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("VAE encode output failed during generator-seeded sampling.") from exc

    explicit_latents = getattr(output, "latents", None)
    if torch.is_tensor(explicit_latents):
        return explicit_latents

    # Newer diffusers-style outputs: AutoencoderKLOutput with latent_dist.
    if hasattr(output, "latent_dist"):
        dist = getattr(output, "latent_dist")
        if hasattr(dist, "sample"):
            try:
                return _sample_output(dist)
            except Exception:  # noqa: BLE001
                if generator is not None:
                    raise
                pass
        if hasattr(dist, "mean") and torch.is_tensor(dist.mean):
            return dist.mean
    # Objects that are themselves distributions (e.g., DiagonalGaussianDistribution)
    if hasattr(output, "sample") and callable(getattr(output, "sample", None)):
        try:
            sample = _sample_output(output)
            if torch.is_tensor(sample):
                return sample
        except Exception:  # noqa: BLE001
            if generator is not None:
                raise
            pass
    if hasattr(output, "mean") and torch.is_tensor(getattr(output, "mean")):
        return getattr(output, "mean")
    if hasattr(output, "sample") and torch.is_tensor(getattr(output, "sample")):
        return output.sample
    if torch.is_tensor(output):
        return output
    # Legacy/variant encoders may return tuples like (latents, aux) or (AutoencoderKLOutput, aux).
    # Walk the tuple/list and recursively unwrap the first tensor-like item we find.
    if isinstance(output, (tuple, list)) and output:
        for item in output:
            if torch.is_tensor(item):
                return item
            try:
                inner = _unwrap_encode_output(item, generator=generator)
                if torch.is_tensor(inner):
                    return inner
            except Exception:
                if generator is not None:
                    raise
                continue
    # Fallback: surface an explicit error instead of returning an unsupported type.
    raise RuntimeError(f"VAE encode returned unsupported output type: {type(output)!r}")


def _report_vae_progress(*, phase: str, block_index: int, total_blocks: int) -> None:
    from apps.backend.core.state import state as backend_state

    owner_token = backend_state.current_thread_progress_owner_token().strip()
    if not owner_token:
        raise RuntimeError("VAE progress update requires a seeded progress owner token.")
    sampling_step, sampling_total, _sampling_block_index, _sampling_block_total = backend_state.current_thread_sampling_context()

    backend_state.update_vae_progress(
        phase=phase,
        block_index=int(block_index),
        total_blocks=int(total_blocks),
        owner_token=owner_token,
        sampling_step=int(sampling_step),
        sampling_total=sampling_total,
    )


class _NormalizingFirstStage:
    """Adapter that guarantees process_in/out around a diffusers VAE.

    - scalar-only path:
      - process_in: (x - shift) * scale
      - process_out: (x / scale) + shift
    - per-channel path:
      - process_in: (x - (latents_mean + shift)) * scale / latents_std
      - process_out: x * latents_std / scale + (latents_mean + shift)
    Also proxies encode/decode/to/attributes to the wrapped object.
    """

    def __init__(
        self,
        base,
        *,
        scale: float,
        shift: float | None,
        latents_mean: tuple[float, ...] | None = None,
        latents_std: tuple[float, ...] | None = None,
    ) -> None:
        self._base = base
        self._scale = float(scale)
        self._shift = None if shift is None else float(shift)
        if not math.isfinite(self._scale) or self._scale == 0.0:
            raise RuntimeError(f"Invalid VAE scaling_factor: {self._scale!r} (must be finite and non-zero).")
        if self._shift is not None and not math.isfinite(self._shift):
            raise RuntimeError(f"Invalid VAE shift_factor: {self._shift!r} (must be finite).")

        if (latents_mean is None) != (latents_std is None):
            raise RuntimeError("VAE latent stats must provide both latents_mean and latents_std (or neither).")

        self._latents_mean = None
        self._latents_std = None
        if latents_mean is not None and latents_std is not None:
            mean_values = tuple(float(x) for x in latents_mean)
            std_values = tuple(float(x) for x in latents_std)
            if not mean_values:
                raise RuntimeError("VAE latent stats are empty; expected at least one channel value.")
            if len(mean_values) != len(std_values):
                raise RuntimeError(
                    "VAE latent stats length mismatch: "
                    f"len(latents_mean)={len(mean_values)} len(latents_std)={len(std_values)}."
                )
            if any(not math.isfinite(value) for value in mean_values):
                raise RuntimeError("VAE latents_mean contains non-finite values.")
            if any((not math.isfinite(value)) or value <= 0.0 for value in std_values):
                raise RuntimeError("VAE latents_std must contain finite positive values.")
            default_dtype = torch.get_default_dtype()
            self._latents_mean = torch.tensor(mean_values, dtype=default_dtype)
            self._latents_std = torch.tensor(std_values, dtype=default_dtype)

    # Proxy core API used by VAE wrapper
    def encode(self, *args, **kwargs):  # noqa: D401
        return self._base.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):  # noqa: D401
        return self._base.decode(*args, **kwargs)

    def to(self, *args, **kwargs):  # noqa: D401
        return self._base.to(*args, **kwargs)

    # Normalization API expected by engines
    def process_in(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("process_in expects a torch.Tensor")
        stats = self._stats_for(x)
        shift = 0.0 if self._shift is None else self._shift
        if stats is None:
            return (x - shift) * self._scale
        latents_mean, latents_std = stats
        return (x - (latents_mean + shift)) * self._scale / latents_std

    def process_out(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("process_out expects a torch.Tensor")
        stats = self._stats_for(x)
        shift = 0.0 if self._shift is None else self._shift
        if stats is None:
            return (x / self._scale) + shift
        latents_mean, latents_std = stats
        return x * latents_std / self._scale + (latents_mean + shift)

    def _stats_for(self, x: torch.Tensor):
        if self._latents_mean is None or self._latents_std is None:
            return None
        if x.ndim not in (4, 5):
            raise RuntimeError(
                "VAE latent stats only support 4D/5D tensors; "
                f"got shape={tuple(x.shape)}."
            )
        channels = int(x.shape[1]) if x.ndim >= 2 else -1
        expected_channels = int(self._latents_mean.shape[0])
        if channels != expected_channels:
            raise RuntimeError(
                "VAE latent channel mismatch for per-channel normalization: "
                f"tensor_channels={channels} expected_channels={expected_channels} "
                f"shape={tuple(x.shape)}."
            )
        view_shape = (1, expected_channels) + (1,) * (x.ndim - 2)
        latents_mean = self._latents_mean.to(device=x.device, dtype=x.dtype).view(view_shape)
        latents_std = self._latents_std.to(device=x.device, dtype=x.dtype).view(view_shape)
        return latents_mean, latents_std

    def __getattr__(self, name: str):
        # Delegate any other attribute access to the base VAE
        return getattr(self._base, name)

    @staticmethod
    def wrap(base, *, family=None):
        """Wrap a VAE with normalization.
        
        Args:
            base: The base VAE model.
            family: Optional ModelFamily for fallback scaling/shift values.
        
        Returns:
            _NormalizingFirstStage wrapper.
        """
        cfg = getattr(base, "config", None)
        policy = resolve_vae_normalization_policy(config=cfg, family=family)
        _, latents_mean = read_vae_config_field(cfg, "latents_mean")
        _, latents_std = read_vae_config_field(cfg, "latents_std")

        def _coerce_optional_float_tuple(name: str, value):
            if value is None:
                return None
            try:
                return tuple(float(x) for x in value)
            except TypeError as exc:
                raise RuntimeError(f"VAE config field '{name}' must be an iterable of numbers.") from exc
            except ValueError as exc:
                raise RuntimeError(f"VAE config field '{name}' contains non-numeric values.") from exc

        latents_mean_values = _coerce_optional_float_tuple("latents_mean", latents_mean)
        latents_std_values = _coerce_optional_float_tuple("latents_std", latents_std)

        if latents_mean_values is not None and latents_std_values is not None:
            logger.info(
                "[VAE] normalization enabled: scaling_factor=%s shift_factor=%s channels=%d (per-channel stats)",
                policy.scaling_factor,
                policy.shift_factor,
                len(latents_mean_values),
            )
        else:
            logger.info(
                "[VAE] normalization enabled: scaling_factor=%s shift_factor=%s",
                policy.scaling_factor,
                policy.shift_factor,
            )
        return _NormalizingFirstStage(
            base,
            scale=float(policy.scaling_factor),
            shift=policy.shift_factor,
            latents_mean=latents_mean_values,
            latents_std=latents_std_values,
        )

class VAE:
    def __init__(self, model=None, device=None, dtype=None, no_init=False, *, family=None):
        if no_init:
            return

        self.memory_used_encode = (
            lambda shape, dtype: (1767 * shape[2] * shape[3]) * torch.empty((), dtype=dtype).element_size()
        )
        self.memory_used_decode = (
            lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * torch.empty((), dtype=dtype).element_size()
        )
        config = getattr(model, "config", None)
        has_spatial_scale, raw_spatial_scale = read_vae_config_field(config, "scale_factor_spatial")
        if has_spatial_scale:
            try:
                self.downscale_ratio = int(raw_spatial_scale)
            except Exception as exc:  # noqa: BLE001 - fail-loud config boundary
                raise RuntimeError(
                    f"VAE scale_factor_spatial must be an integer; got {raw_spatial_scale!r}."
                ) from exc
            if self.downscale_ratio <= 0:
                raise RuntimeError(
                    f"VAE scale_factor_spatial must be positive; got {self.downscale_ratio}."
                )
        else:
            has_down_blocks, raw_down_blocks = read_vae_config_field(config, "down_block_types")
            if (
                not has_down_blocks
                or not isinstance(raw_down_blocks, (list, tuple))
                or not raw_down_blocks
            ):
                raise RuntimeError(
                    "VAE config must define positive scale_factor_spatial or a non-empty down_block_types sequence."
                )
            self.downscale_ratio = int(2 ** (len(raw_down_blocks) - 1))

        has_latent_channels, raw_latent_channels = read_vae_config_field(config, "latent_channels")
        if not has_latent_channels:
            has_latent_channels, raw_latent_channels = read_vae_config_field(config, "z_dim")
        if not has_latent_channels:
            raise RuntimeError("VAE config must define latent_channels or z_dim.")
        try:
            self.latent_channels = int(raw_latent_channels)
        except Exception as exc:  # noqa: BLE001 - fail-loud config boundary
            raise RuntimeError(
                f"VAE latent channel count must be an integer; got {raw_latent_channels!r}."
            ) from exc
        if self.latent_channels <= 0:
            raise RuntimeError(
                f"VAE latent channel count must be positive; got {self.latent_channels}."
            )
        if family is not None and not isinstance(family, ModelFamily):
            raise RuntimeError(f"Invalid VAE family type: {type(family)!r}. Expected ModelFamily or None.")
        self.family: ModelFamily | None = family
        self._decode_geometry_override_logged = False

        # Ensure process_in/out are always available via adapter
        self.first_stage_model = _NormalizingFirstStage.wrap(model.eval(), family=self.family)

        if device is None:
            device = memory_management.manager.get_device(DeviceRole.VAE)

        self.device = device
        offload_device = memory_management.manager.get_offload_device(DeviceRole.VAE)

        if dtype is None:
            native_storage = None
            try:
                native_storage = next(model.parameters()).dtype
            except Exception:  # noqa: BLE001
                native_storage = None
            if native_storage is None:
                native_storage = memory_management.manager.dtype_for_role(DeviceRole.VAE)
            dtype = memory_management.manager.dtype_for_role(DeviceRole.VAE, native_dtype=native_storage)

        self.vae_dtype: torch.dtype | None = None
        self.vae_compute_dtype: torch.dtype | None = None
        self._pending_dtype = dtype  # Will be applied lazily when VAE is first used
        self._supports_manual_cast_split = self._detect_manual_cast_split_support()
        self._manual_cast_split_warned = False
        self.offload_device = offload_device
        intermediate_policy = memory_management.manager.config.component_policy(DeviceRole.INTERMEDIATE)
        if intermediate_policy.preferred_backend == DeviceBackend.AUTO:
            self.output_device = memory_management.manager.cpu_device
        else:
            self.output_device = memory_management.manager.get_device(DeviceRole.INTERMEDIATE)

        self.patcher = ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device
        )

    def clone(self):
        n = VAE(no_init=True)
        n.patcher = self.patcher.clone()
        n.memory_used_encode = self.memory_used_encode
        n.memory_used_decode = self.memory_used_decode
        n.downscale_ratio = self.downscale_ratio
        n.latent_channels = self.latent_channels
        n.first_stage_model = self.first_stage_model
        n.device = self.device
        n.vae_dtype = self.vae_dtype
        n.vae_compute_dtype = self.vae_compute_dtype
        n._supports_manual_cast_split = self._supports_manual_cast_split
        n._manual_cast_split_warned = self._manual_cast_split_warned
        n.family = self.family
        n._decode_geometry_override_logged = self._decode_geometry_override_logged
        n.output_device = self.output_device
        return n

    def _detect_manual_cast_split_support(self) -> bool:
        base = getattr(self.first_stage_model, "_base", self.first_stage_model)
        if hasattr(base, "parameters_manual_cast"):
            return True
        modules_getter = getattr(base, "modules", None)
        if callable(modules_getter):
            try:
                for module in modules_getter():
                    if hasattr(module, "parameters_manual_cast"):
                        return True
            except Exception:  # noqa: BLE001
                logger.debug("VAE manual-cast capability probe failed.", exc_info=True)
        return False

    def _effective_forward_dtype(
        self,
        *,
        storage_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        context: str,
    ) -> torch.dtype:
        if compute_dtype == storage_dtype:
            return storage_dtype
        if self._supports_manual_cast_split:
            return compute_dtype
        if not self._manual_cast_split_warned:
            base = getattr(self.first_stage_model, "_base", self.first_stage_model)
            logger.warning(
                "VAE compute split suppressed at %s: module=%s has no manual-cast markers; forcing forward dtype to storage (%s).",
                context,
                base.__class__.__name__,
                str(storage_dtype),
            )
            self._manual_cast_split_warned = True
        return storage_dtype

    def _resolve_dtypes(self) -> tuple[torch.dtype, torch.dtype]:
        native_storage = self.vae_dtype or self._pending_dtype or memory_management.manager.dtype_for_role(DeviceRole.VAE)
        storage_dtype = memory_management.manager.dtype_for_role(DeviceRole.VAE, native_dtype=native_storage)
        compute_dtype = memory_management.manager.compute_dtype_for_role(DeviceRole.VAE, storage_dtype=storage_dtype)
        return storage_dtype, compute_dtype

    def _active_forward_dtype(
        self,
        *,
        storage_dtype: torch.dtype | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> torch.dtype:
        resolved_storage = storage_dtype or self.vae_dtype or self._pending_dtype or memory_management.manager.dtype_for_role(
            DeviceRole.VAE
        )
        if compute_dtype is not None:
            return self._effective_forward_dtype(
                storage_dtype=resolved_storage,
                compute_dtype=compute_dtype,
                context="active_forward_dtype(explicit)",
            )
        if self.vae_compute_dtype is not None:
            return self._effective_forward_dtype(
                storage_dtype=resolved_storage,
                compute_dtype=self.vae_compute_dtype,
                context="active_forward_dtype(cached)",
            )
        if storage_dtype is not None:
            return storage_dtype
        if self.vae_dtype is not None:
            return self.vae_dtype
        fallback_compute = memory_management.manager.compute_dtype_for_role(
            DeviceRole.VAE,
            storage_dtype=resolved_storage,
        )
        return self._effective_forward_dtype(
            storage_dtype=resolved_storage,
            compute_dtype=fallback_compute,
            context="active_forward_dtype(fallback)",
        )

    def _autocast_context(self, forward_dtype: torch.dtype):
        device_type = torch.device(self.device).type
        supported: dict[str, set[torch.dtype]] = {
            "cuda": {torch.float16, torch.bfloat16},
            "cpu": {torch.bfloat16},
            "xpu": {torch.float16, torch.bfloat16},
            "mps": {torch.float16},
        }
        if forward_dtype not in supported.get(device_type, set()):
            return contextlib.nullcontext()
        try:
            return torch.autocast(device_type=device_type, dtype=forward_dtype)
        except Exception:  # noqa: BLE001
            logger.debug(
                "Autocast unavailable for VAE device=%s dtype=%s; falling back to regular forward.",
                device_type,
                str(forward_dtype),
            )
            return contextlib.nullcontext()

    def _apply_precision(self, dtype: torch.dtype, device: torch.device | str | None = None) -> None:
        if dtype == self.vae_dtype:
            return
        previous = self.vae_dtype
        target_device = device if device is not None else self.device
        base = getattr(self.first_stage_model, "_base", self.first_stage_model)
        base.to(device=target_device, dtype=dtype)
        self.vae_dtype = dtype
        logger.info(
            "VAE precision updated: %s -> %s on %s",
            "none" if previous is None else str(previous),
            str(dtype),
            target_device,
        )

    def _log_precision_resolution(
        self,
        *,
        storage_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        forward_dtype: torch.dtype,
        context: str,
    ) -> None:
        if storage_dtype == forward_dtype:
            logger.debug(
                "VAE precision %s: storage=%s compute=%s forward=%s",
                context,
                str(storage_dtype),
                str(compute_dtype),
                str(forward_dtype),
            )
            return
        logger.info(
            "VAE precision %s: storage=%s compute=%s -> forward=%s (compute-preferred runtime)",
            context,
            str(storage_dtype),
            str(compute_dtype),
            str(forward_dtype),
        )

    def _decode_forward(self, samples: torch.Tensor, *, forward_dtype: torch.dtype) -> torch.Tensor:
        with self._autocast_context(forward_dtype):
            decoded_raw = self.first_stage_model.decode(samples.to(device=self.device, dtype=forward_dtype))
        return _unwrap_decode_output(decoded_raw)

    def _encode_forward(
        self,
        pixels_in: torch.Tensor,
        *,
        forward_dtype: torch.dtype,
        regulation,
        encode_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        base = getattr(self.first_stage_model, "_base", self.first_stage_model)
        pixels_typed = pixels_in.to(device=self.device, dtype=forward_dtype, copy=True)
        pixels_typed.mul_(2.0).sub_(1.0)
        with self._autocast_context(forward_dtype):
            if DiffusersAutoencoderKL is not None and isinstance(base, DiffusersAutoencoderKL):
                encoded_raw = base.encode(pixels_typed, return_dict=True)
            elif AutoencoderKL_LDM is not None and isinstance(base, AutoencoderKL_LDM):
                encoded_raw = base.encode(pixels_typed, regulation)
            else:
                try:
                    encoded_raw = base.encode(pixels_typed, regulation)
                except TypeError:
                    encoded_raw = base.encode(pixels_typed)
        if isinstance(encoded_raw, (tuple, list)) and encoded_raw:
            encoded_raw = encoded_raw[0]
        return _unwrap_encode_output(encoded_raw, generator=encode_generator).to(self.output_device)

    @staticmethod
    def _decode_crop_bounds(
        *,
        window_start: int,
        window_end: int,
        context_start: int,
        decoded_extent: int,
        upscale_ratio: int,
    ) -> tuple[int, int]:
        crop_start = (window_start - context_start) * upscale_ratio
        crop_end = crop_start + ((window_end - window_start) * upscale_ratio)
        if crop_start < 0 or crop_end > decoded_extent or crop_start >= crop_end:
            raise RuntimeError(
                "Invalid tiled VAE decode crop bounds: "
                f"crop=({crop_start}, {crop_end}) decoded_extent={decoded_extent} "
                f"context_start={context_start} window=({window_start}, {window_end}) ratio={upscale_ratio}."
            )
        return crop_start, crop_end

    @staticmethod
    def _encode_crop_bounds(
        *,
        window_start: int,
        window_end: int,
        context_start: int,
        encoded_extent: int,
        downscale_ratio: int,
    ) -> tuple[int, int, int, int] | None:
        out_start = window_start // downscale_ratio
        out_end = window_end // downscale_ratio
        if out_end <= out_start:
            return None
        context_start_latent = context_start // downscale_ratio
        crop_start = out_start - context_start_latent
        crop_end = crop_start + (out_end - out_start)
        if crop_start < 0 or crop_end > encoded_extent or crop_start >= crop_end:
            raise RuntimeError(
                "Invalid tiled VAE encode crop bounds: "
                f"crop=({crop_start}, {crop_end}) encoded_extent={encoded_extent} "
                f"context_start={context_start} window=({window_start}, {window_end}) ratio={downscale_ratio}."
            )
        return out_start, out_end, crop_start, crop_end

    def _resolve_decode_tiled_geometry(self) -> VaeTileGeometry:
        geometry = resolve_vae_decode_tiled_geometry(family=self.family)
        if geometry != DEFAULT_VAE_DECODE_TILED_GEOMETRY and not self._decode_geometry_override_logged:
            family = self.family
            if family is None:
                raise RuntimeError("Non-default VAE decode tiled geometry requires a known model family.")
            logger.info(
                "Applying family decode tiled geometry override: family=%s tile_x=%d tile_y=%d overlap=%d",
                family.value,
                geometry.tile_x,
                geometry.tile_y,
                geometry.overlap,
            )
            self._decode_geometry_override_logged = True
        return geometry

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16, progress_callback=None):
        if samples.ndim != 4:
            raise RuntimeError(f"decode_tiled_ expects NCHW latents; got shape={tuple(samples.shape)}.")
        geometry = VaeTileGeometry(
            tile_x=int(tile_x),
            tile_y=int(tile_y),
            overlap=max(0, int(overlap)),
        )
        forward_dtype = self._active_forward_dtype()
        output_height = round(samples.shape[2] * self.downscale_ratio)
        output_width = round(samples.shape[3] * self.downscale_ratio)
        output = torch.empty(
            (samples.shape[0], 3, output_height, output_width),
            device=self.output_device,
            dtype=forward_dtype,
        )
        windows = tuple(
            iter_vae_tile_windows(
                height=int(samples.shape[2]),
                width=int(samples.shape[3]),
                tile_y=geometry.tile_y,
                tile_x=geometry.tile_x,
                pad_y=geometry.overlap,
                pad_x=geometry.overlap,
            )
        )
        if not windows:
            raise RuntimeError("decode_tiled_ produced no tile windows; check tile geometry.")
        decode_scale = int(self.downscale_ratio)

        total_tiles = max(1, int(samples.shape[0]) * len(windows))
        processed_tiles = 0
        for batch_index in range(samples.shape[0]):
            for window in windows:
                latent_tile = samples[
                    batch_index : batch_index + 1,
                    :,
                    window.context_y0 : window.context_y1,
                    window.context_x0 : window.context_x1,
                ]
                decoded_tile = self._decode_forward(latent_tile, forward_dtype=forward_dtype).to(self.output_device)
                crop_y0, crop_y1 = self._decode_crop_bounds(
                    window_start=window.core_y0,
                    window_end=window.core_y1,
                    context_start=window.context_y0,
                    decoded_extent=int(decoded_tile.shape[2]),
                    upscale_ratio=decode_scale,
                )
                crop_x0, crop_x1 = self._decode_crop_bounds(
                    window_start=window.core_x0,
                    window_end=window.core_x1,
                    context_start=window.context_x0,
                    decoded_extent=int(decoded_tile.shape[3]),
                    upscale_ratio=decode_scale,
                )
                out_y0 = window.core_y0 * decode_scale
                out_y1 = window.core_y1 * decode_scale
                out_x0 = window.core_x0 * decode_scale
                out_x1 = window.core_x1 * decode_scale
                output[batch_index : batch_index + 1, :, out_y0:out_y1, out_x0:out_x1] = decoded_tile[
                    :,
                    :,
                    crop_y0:crop_y1,
                    crop_x0:crop_x1,
                ]
                processed_tiles += 1
                if callable(progress_callback):
                    try:
                        progress_callback(int(processed_tiles), int(total_tiles))
                    except Exception:
                        logger.debug("decode_tiled_ progress callback failed", exc_info=True)
        return torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0)

    def encode_tiled_(
        self,
        pixel_samples,
        tile_x=512,
        tile_y=512,
        overlap=64,
        progress_callback=None,
        *,
        encode_generator: torch.Generator | None = None,
    ):
        if pixel_samples.ndim != 4:
            raise RuntimeError(f"encode_tiled_ expects NCHW pixels; got shape={tuple(pixel_samples.shape)}.")
        geometry = VaeTileGeometry(
            tile_x=int(tile_x),
            tile_y=int(tile_y),
            overlap=max(0, int(overlap)),
        )
        forward_dtype = self._active_forward_dtype()
        output_height = round(pixel_samples.shape[2] // self.downscale_ratio)
        output_width = round(pixel_samples.shape[3] // self.downscale_ratio)
        output = torch.empty(
            (pixel_samples.shape[0], self.latent_channels, output_height, output_width),
            device=self.output_device,
            dtype=forward_dtype,
        )
        windows = tuple(
            iter_vae_tile_windows(
                height=int(pixel_samples.shape[2]),
                width=int(pixel_samples.shape[3]),
                tile_y=geometry.tile_y,
                tile_x=geometry.tile_x,
                pad_y=geometry.overlap,
                pad_x=geometry.overlap,
            )
        )
        if not windows:
            raise RuntimeError("encode_tiled_ produced no tile windows; check tile geometry.")
        downscale_ratio = int(self.downscale_ratio)
        regulation = self.patcher.model_options.get("model_vae_regulation", None)

        total_tiles = max(1, int(pixel_samples.shape[0]) * len(windows))
        processed_tiles = 0
        for batch_index in range(pixel_samples.shape[0]):
            for window in windows:
                pixels_tile = pixel_samples[
                    batch_index : batch_index + 1,
                    :,
                    window.context_y0 : window.context_y1,
                    window.context_x0 : window.context_x1,
                ]
                encoded_tile = self._encode_forward(
                    pixels_tile,
                    forward_dtype=forward_dtype,
                    regulation=regulation,
                    encode_generator=encode_generator,
                )
                y_bounds = self._encode_crop_bounds(
                    window_start=window.core_y0,
                    window_end=window.core_y1,
                    context_start=window.context_y0,
                    encoded_extent=int(encoded_tile.shape[2]),
                    downscale_ratio=downscale_ratio,
                )
                x_bounds = self._encode_crop_bounds(
                    window_start=window.core_x0,
                    window_end=window.core_x1,
                    context_start=window.context_x0,
                    encoded_extent=int(encoded_tile.shape[3]),
                    downscale_ratio=downscale_ratio,
                )
                if y_bounds is None or x_bounds is None:
                    continue
                out_y0, out_y1, crop_y0, crop_y1 = y_bounds
                out_x0, out_x1, crop_x0, crop_x1 = x_bounds
                output[batch_index : batch_index + 1, :, out_y0:out_y1, out_x0:out_x1] = encoded_tile[
                    :,
                    :,
                    crop_y0:crop_y1,
                    crop_x0:crop_x1,
                ]
                processed_tiles += 1
                if callable(progress_callback):
                    try:
                        progress_callback(int(processed_tiles), int(total_tiles))
                    except Exception:
                        logger.debug("encode_tiled_ progress callback failed", exc_info=True)
        return output

    def _decode_cpu_fallback(self, samples_in: torch.Tensor) -> torch.Tensor:
        """Best-effort CPU decode path used after CUDA OOM when smart fallback is enabled.

        This bypasses GPU memory heuristics and runs a single full-image decode on CPU,
        restoring the original VAE device afterwards where possible.
        """
        base = getattr(self.first_stage_model, "_base", self.first_stage_model)
        orig_device: torch.device | None = None
        orig_dtype: torch.dtype | None = None
        try:
            try:
                params = base.parameters()
                first = next(params)
                orig_device = first.device
                orig_dtype = first.dtype
            except Exception:  # noqa: BLE001
                orig_device = None
                orig_dtype = None

            cpu_device = memory_management.manager.cpu_device
            storage_hint = orig_dtype or self.vae_dtype or self._pending_dtype or samples_in.dtype
            cpu_forward_dtype = memory_management.manager.compute_dtype_for_role(
                DeviceRole.VAE,
                storage_dtype=storage_hint,
            )
            base.to(device=cpu_device, dtype=cpu_forward_dtype)

            with torch.no_grad():
                samples_cpu = samples_in.to(cpu_device, dtype=cpu_forward_dtype)
                decoded_raw = self.first_stage_model.decode(samples_cpu)
                decoded = _unwrap_decode_output(decoded_raw)
                decoded.clamp_(-1.0, 1.0).add_(1.0).mul_(0.5)
                pixel_samples = decoded.to(self.output_device)

            return pixel_samples
        finally:
            if orig_device is not None:
                try:
                    restore_dtype = orig_dtype or self.vae_dtype or self._pending_dtype or samples_in.dtype
                    base.to(device=orig_device, dtype=restore_dtype)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to restore VAE device after CPU fallback.", exc_info=True)

    def _encode_cpu_fallback(
        self,
        pixel_samples_chw: torch.Tensor,
        regulation,
        *,
        encode_seed: int | None = None,
    ) -> torch.Tensor:
        """Best-effort CPU encode path used after CUDA OOM when smart fallback is enabled.

        Mirrors the GPU encode logic but runs entirely on CPU to avoid repeated
        OOM loops on large inputs. Restores the original VAE device afterwards
        when possible.
        """
        base = getattr(self.first_stage_model, "_base", self.first_stage_model)
        orig_device: torch.device | None = None
        orig_dtype: torch.dtype | None = None
        try:
            try:
                params = base.parameters()
                first = next(params)
                orig_device = first.device
                orig_dtype = first.dtype
            except Exception:  # noqa: BLE001
                orig_device = None
                orig_dtype = None

            cpu_device = memory_management.manager.cpu_device
            storage_hint = orig_dtype or self.vae_dtype or self._pending_dtype or pixel_samples_chw.dtype
            cpu_forward_dtype = memory_management.manager.compute_dtype_for_role(
                DeviceRole.VAE,
                storage_dtype=storage_hint,
            )
            base.to(device=cpu_device, dtype=cpu_forward_dtype)

            encode_generator = _new_encode_generator(encode_seed=encode_seed, device=cpu_device)
            with torch.no_grad():
                pixels_in = pixel_samples_chw.to(cpu_device, dtype=cpu_forward_dtype, copy=True)
                pixels_in.mul_(2.0).sub_(1.0)

                if DiffusersAutoencoderKL is not None and isinstance(base, DiffusersAutoencoderKL):
                    encoded_raw = base.encode(pixels_in, return_dict=True)
                elif AutoencoderKL_LDM is not None and isinstance(base, AutoencoderKL_LDM):
                    encoded_raw = base.encode(pixels_in, regulation)
                else:
                    try:
                        encoded_raw = base.encode(pixels_in, regulation)
                    except TypeError:
                        encoded_raw = base.encode(pixels_in)

                if isinstance(encoded_raw, (tuple, list)) and encoded_raw:
                    encoded_raw = encoded_raw[0]
                encoded = _unwrap_encode_output(encoded_raw, generator=encode_generator).to(self.output_device)

            return encoded
        finally:
            if orig_device is not None:
                try:
                    restore_dtype = orig_dtype or self.vae_dtype or self._pending_dtype or pixel_samples_chw.dtype
                    base.to(device=orig_device, dtype=restore_dtype)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to restore VAE device after CPU encode fallback.", exc_info=True)

    def decode_inner(self, samples_in):
        _tensor_stats("decode_inner.latents", samples_in)
        progress_phase = "decode"
        decode_total_blocks = 1
        while True:
            pixel_samples_are_minus_one_to_one = False
            desired_storage, desired_compute = self._resolve_dtypes()
            forward_dtype = self._active_forward_dtype(
                storage_dtype=desired_storage,
                compute_dtype=desired_compute,
            )
            self._log_precision_resolution(
                storage_dtype=desired_storage,
                compute_dtype=desired_compute,
                forward_dtype=forward_dtype,
                context="decode_inner",
            )
            self._apply_precision(desired_storage)
            self.vae_compute_dtype = desired_compute
            use_tiled = bool(memory_management.manager.vae_always_tiled)

            try:
                if use_tiled:
                    decode_geometry = self._resolve_decode_tiled_geometry()
                    decode_total_blocks = max(1, int(samples_in.shape[0]))
                    _report_vae_progress(
                        phase=progress_phase,
                        block_index=0,
                        total_blocks=decode_total_blocks,
                    )
                    memory_management.manager.load_model(self.patcher)

                    def _decode_progress(idx: int, total: int) -> None:
                        nonlocal decode_total_blocks
                        decode_total_blocks = max(1, int(total))
                        _report_vae_progress(
                            phase=progress_phase,
                            block_index=int(idx),
                            total_blocks=decode_total_blocks,
                        )

                    pixel_samples = self.decode_tiled_(
                        samples_in,
                        tile_x=decode_geometry.tile_x,
                        tile_y=decode_geometry.tile_y,
                        overlap=decode_geometry.overlap,
                        progress_callback=_decode_progress,
                    )
                    pixel_samples_are_minus_one_to_one = False
                else:
                    memory_used = self.memory_used_decode(samples_in.shape, forward_dtype)
                    memory_management.manager.load_models([self.patcher], memory_required=memory_used)
                    free_memory = memory_management.manager.get_free_memory(self.device)
                    batch_number = max(1, int(free_memory / memory_used))
                    total_batches = max(1, int(math.ceil(float(samples_in.shape[0]) / float(batch_number))))
                    decode_total_blocks = total_batches
                    _report_vae_progress(
                        phase=progress_phase,
                        block_index=0,
                        total_blocks=decode_total_blocks,
                    )

                    pixel_samples = torch.empty(
                        (
                            samples_in.shape[0],
                            3,
                            round(samples_in.shape[2] * self.downscale_ratio),
                            round(samples_in.shape[3] * self.downscale_ratio),
                        ),
                        device=self.output_device,
                        dtype=forward_dtype,
                    )
                    for batch_idx, x in enumerate(range(0, samples_in.shape[0], batch_number), start=1):
                        samples = samples_in[x:x + batch_number]
                        decoded = self._decode_forward(samples, forward_dtype=forward_dtype)
                        decoded.clamp_(-1.0, 1.0)
                        pixel_samples[x:x + batch_number] = decoded.to(self.output_device)
                        pixel_samples_are_minus_one_to_one = True
                        _tensor_stats("decode_inner.batch_decoded", decoded)
                        _report_vae_progress(
                            phase=progress_phase,
                            block_index=batch_idx,
                            total_blocks=decode_total_blocks,
                        )
            except memory_management.manager.oom_exception:
                if smart_fallback_enabled():
                    logger.warning(
                        "VAE decode OOM on %s with tiled=%s; attempting CPU fallback.",
                        self.device,
                        bool(memory_management.manager.vae_always_tiled),
                    )
                    _report_vae_progress(phase=progress_phase, block_index=0, total_blocks=1)
                    pixel_samples = self._decode_cpu_fallback(samples_in)
                    pixel_samples_are_minus_one_to_one = False
                    decode_total_blocks = 1
                    _report_vae_progress(phase=progress_phase, block_index=1, total_blocks=1)
                else:
                    if use_tiled:
                        raise RuntimeError(
                            "VAE tiled decode ran out of memory with smart fallback disabled. "
                            "Disable vae_always_tiled or enable smart fallback."
                        )
                    logger.warning(
                        "Ran out of memory when regular VAE decoding; retrying with tiled VAE decoding."
                    )
                    try:
                        del decoded
                    except UnboundLocalError:
                        pass
                    try:
                        del pixel_samples
                    except UnboundLocalError:
                        pass
                    memory_management.manager.unload_model(
                        self.patcher,
                        source="apps.backend.patchers.vae.decode_inner",
                        stage="vae.decode_inner.tiled_retry_cleanup",
                        component_hint="vae",
                        event_reason="regular_decode_oom",
                    )
                    gc.collect()
                    memory_management.manager.soft_empty_cache(force=True)
                    memory_management.manager.load_model(
                        self.patcher,
                        source="apps.backend.patchers.vae.decode_inner",
                        stage="vae.decode_inner.tiled_retry",
                        component_hint="vae",
                        event_reason="regular_decode_oom",
                    )
                    decode_total_blocks = max(1, int(samples_in.shape[0]))

                    def _decode_retry_progress(idx: int, total: int) -> None:
                        nonlocal decode_total_blocks
                        decode_total_blocks = max(1, int(total))
                        _report_vae_progress(
                            phase=progress_phase,
                            block_index=int(idx),
                            total_blocks=decode_total_blocks,
                        )

                    decode_geometry = self._resolve_decode_tiled_geometry()
                    pixel_samples = self.decode_tiled_(
                        samples_in,
                        tile_x=decode_geometry.tile_x,
                        tile_y=decode_geometry.tile_y,
                        overlap=decode_geometry.overlap,
                        progress_callback=_decode_retry_progress,
                    )
                    pixel_samples_are_minus_one_to_one = False

            # Return BCHW format in [-1, 1] range directly
            # This is what sampling pipelines expect - no conversion needed in engines
            result = pixel_samples.to(self.output_device)
            if not pixel_samples_are_minus_one_to_one:
                result = result * 2.0 - 1.0  # [0,1] -> [-1,1]
            _tensor_stats("decode_inner.result", result)
            if torch.isnan(result).any():
                logger.warning(
                    "VAE decode produced NaNs on %s using dtype %s; requesting precision fallback.",
                    self.device,
                    str(self.vae_dtype),
                )
                next_dtype = memory_management.manager.report_precision_failure(
                    DeviceRole.VAE,
                    location="vae.decode",
                    reason="NaN detected in decoded output",
                )
                if next_dtype is None:
                    hint = memory_management.manager.precision_hint(DeviceRole.VAE)
                    raise RuntimeError(
                        f"VAE decode produced NaNs on {self.device} with dtype {self.vae_dtype}. {hint}"
                    )
                del pixel_samples
                del result
                self._apply_precision(next_dtype)
                memory_management.manager.soft_empty_cache(force=True)
                continue

            _report_vae_progress(
                phase=progress_phase,
                block_index=decode_total_blocks,
                total_blocks=decode_total_blocks,
            )
            return result

    def decode(self, samples_in):
        wrapper = self.patcher.model_options.get('model_vae_decode_wrapper', None)
        if wrapper is None:
            return self.decode_inner(samples_in)
        else:
            return wrapper(self.decode_inner, samples_in)

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
        memory_management.manager.load_model(self.patcher)
        desired_storage, desired_compute = self._resolve_dtypes()
        forward_dtype = self._active_forward_dtype(
            storage_dtype=desired_storage,
            compute_dtype=desired_compute,
        )
        self._log_precision_resolution(
            storage_dtype=desired_storage,
            compute_dtype=desired_compute,
            forward_dtype=forward_dtype,
            context="decode_tiled",
        )
        self._apply_precision(desired_storage)
        self.vae_compute_dtype = desired_compute
        geometry = VaeTileGeometry(
            tile_x=int(tile_x),
            tile_y=int(tile_y),
            overlap=max(0, int(overlap)),
        )
        if geometry == DEFAULT_VAE_DECODE_TILED_GEOMETRY:
            geometry = self._resolve_decode_tiled_geometry()
        output = self.decode_tiled_(
            samples,
            tile_x=geometry.tile_x,
            tile_y=geometry.tile_y,
            overlap=geometry.overlap,
        )
        # Return BCHW format in [-1, 1] range like decode_inner
        return output * 2.0 - 1.0

    def encode_inner(self, pixel_samples, *, encode_seed: int | None = None):
        regulation = self.patcher.model_options.get("model_vae_regulation", None)
        pixel_samples = pixel_samples.movedim(-1, 1)
        progress_phase = "encode"
        encode_total_blocks = 1

        while True:
            desired_storage, desired_compute = self._resolve_dtypes()
            forward_dtype = self._active_forward_dtype(
                storage_dtype=desired_storage,
                compute_dtype=desired_compute,
            )
            self._log_precision_resolution(
                storage_dtype=desired_storage,
                compute_dtype=desired_compute,
                forward_dtype=forward_dtype,
                context="encode_inner",
            )
            self._apply_precision(desired_storage)
            self.vae_compute_dtype = desired_compute
            use_tiled = bool(memory_management.manager.vae_always_tiled)

            try:
                encode_generator = _new_encode_generator(encode_seed=encode_seed, device=self.device)
                if use_tiled:
                    encode_total_blocks = max(1, int(pixel_samples.shape[0]))
                    _report_vae_progress(
                        phase=progress_phase,
                        block_index=0,
                        total_blocks=encode_total_blocks,
                    )
                    memory_management.manager.load_model(self.patcher)

                    def _encode_progress(idx: int, total: int) -> None:
                        nonlocal encode_total_blocks
                        encode_total_blocks = max(1, int(total))
                        _report_vae_progress(
                            phase=progress_phase,
                            block_index=int(idx),
                            total_blocks=encode_total_blocks,
                        )

                    samples = self.encode_tiled_(
                        pixel_samples,
                        progress_callback=_encode_progress,
                        encode_generator=encode_generator,
                    )
                else:
                    memory_used = self.memory_used_encode(pixel_samples.shape, forward_dtype)
                    memory_management.manager.load_models([self.patcher], memory_required=memory_used)
                    free_memory = memory_management.manager.get_free_memory(self.device)
                    batch_number = max(1, int(free_memory / memory_used))
                    total_batches = max(1, int(math.ceil(float(pixel_samples.shape[0]) / float(batch_number))))
                    encode_total_blocks = total_batches
                    _report_vae_progress(
                        phase=progress_phase,
                        block_index=0,
                        total_blocks=encode_total_blocks,
                    )
                    samples = torch.empty(
                        (
                            pixel_samples.shape[0],
                            self.latent_channels,
                            round(pixel_samples.shape[2] // self.downscale_ratio),
                            round(pixel_samples.shape[3] // self.downscale_ratio),
                        ),
                        device=self.output_device,
                        dtype=forward_dtype,
                    )
                    for batch_idx, x in enumerate(range(0, pixel_samples.shape[0], batch_number), start=1):
                        encoded = self._encode_forward(
                            pixel_samples[x:x + batch_number],
                            forward_dtype=forward_dtype,
                            regulation=regulation,
                            encode_generator=encode_generator,
                        )
                        samples[x:x + batch_number] = encoded
                        _report_vae_progress(
                            phase=progress_phase,
                            block_index=batch_idx,
                            total_blocks=encode_total_blocks,
                        )
            except memory_management.manager.oom_exception:
                if smart_fallback_enabled():
                    logger.warning(
                        "VAE encode OOM on %s with tiled=%s; attempting CPU fallback.",
                        self.device,
                        bool(memory_management.manager.vae_always_tiled),
                    )
                    _report_vae_progress(phase=progress_phase, block_index=0, total_blocks=1)
                    samples = self._encode_cpu_fallback(pixel_samples, regulation, encode_seed=encode_seed)
                    encode_total_blocks = 1
                    _report_vae_progress(phase=progress_phase, block_index=1, total_blocks=1)
                else:
                    if use_tiled:
                        raise RuntimeError(
                            "VAE tiled encode ran out of memory with smart fallback disabled. "
                            "Disable vae_always_tiled or enable smart fallback."
                        )
                    logger.warning(
                        "Ran out of memory when regular VAE encoding; retrying with tiled VAE encoding."
                    )
                    try:
                        del encoded
                    except UnboundLocalError:
                        pass
                    try:
                        del pixels_in
                    except UnboundLocalError:
                        pass
                    try:
                        del samples
                    except UnboundLocalError:
                        pass
                    memory_management.manager.unload_model(
                        self.patcher,
                        source="apps.backend.patchers.vae.encode_inner",
                        stage="vae.encode_inner.tiled_retry_cleanup",
                        component_hint="vae",
                        event_reason="regular_encode_oom",
                    )
                    gc.collect()
                    memory_management.manager.soft_empty_cache(force=True)
                    memory_management.manager.load_model(
                        self.patcher,
                        source="apps.backend.patchers.vae.encode_inner",
                        stage="vae.encode_inner.tiled_retry",
                        component_hint="vae",
                        event_reason="regular_encode_oom",
                    )
                    encode_total_blocks = max(1, int(pixel_samples.shape[0]))

                    def _encode_retry_progress(idx: int, total: int) -> None:
                        nonlocal encode_total_blocks
                        encode_total_blocks = max(1, int(total))
                        _report_vae_progress(
                            phase=progress_phase,
                            block_index=int(idx),
                            total_blocks=encode_total_blocks,
                        )

                    retry_encode_generator = _new_encode_generator(encode_seed=encode_seed, device=self.device)
                    samples = self.encode_tiled_(
                        pixel_samples,
                        progress_callback=_encode_retry_progress,
                        encode_generator=retry_encode_generator,
                    )

            if torch.isnan(samples).any():
                logger.warning(
                    "VAE encode produced NaNs on %s using dtype %s; requesting precision fallback.",
                    self.device,
                    str(self.vae_dtype),
                )
                next_dtype = memory_management.manager.report_precision_failure(
                    DeviceRole.VAE,
                    location="vae.encode",
                    reason="NaN detected in encoded output",
                )
                if next_dtype is None:
                    hint = memory_management.manager.precision_hint(DeviceRole.VAE)
                    raise RuntimeError(
                        f"VAE encode produced NaNs on {self.device} with dtype {self.vae_dtype}. {hint}"
                    )
                del samples
                self._apply_precision(next_dtype)
                memory_management.manager.soft_empty_cache(force=True)
                continue

            _report_vae_progress(
                phase=progress_phase,
                block_index=encode_total_blocks,
                total_blocks=encode_total_blocks,
            )
            return samples

    def encode(self, pixel_samples, *, encode_seed: int | None = None):
        wrapper = self.patcher.model_options.get('model_vae_encode_wrapper', None)
        if wrapper is None:
            return self.encode_inner(pixel_samples, encode_seed=encode_seed)
        else:
            return wrapper(lambda samples: self.encode_inner(samples, encode_seed=encode_seed), pixel_samples)

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64, *, encode_seed: int | None = None):
        memory_management.manager.load_model(self.patcher)
        pixel_samples = pixel_samples.movedim(-1, 1)
        desired_storage, desired_compute = self._resolve_dtypes()
        forward_dtype = self._active_forward_dtype(
            storage_dtype=desired_storage,
            compute_dtype=desired_compute,
        )
        self._log_precision_resolution(
            storage_dtype=desired_storage,
            compute_dtype=desired_compute,
            forward_dtype=forward_dtype,
            context="encode_tiled",
        )
        self._apply_precision(desired_storage)
        self.vae_compute_dtype = desired_compute
        encode_generator = _new_encode_generator(encode_seed=encode_seed, device=self.device)
        samples = self.encode_tiled_(
            pixel_samples,
            tile_x=tile_x,
            tile_y=tile_y,
            overlap=overlap,
            encode_generator=encode_generator,
        )
        return samples

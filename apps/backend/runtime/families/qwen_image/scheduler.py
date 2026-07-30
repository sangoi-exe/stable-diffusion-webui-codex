"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image FlowMatch Euler scheduler metadata, geometry, and native step helpers.
Validates HF-style scheduler metadata, derives packed image sequence lengths, builds the exact dynamically shifted
sigma/timestep ladder, and applies the deterministic FP32 Euler update without importing Diffusers.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageFlowSchedule` (dataclass): Exact inference sigma/timestep tensors for one Qwen Image denoise run.
- `QwenImageLatentGrid` (dataclass): Derived latent/packed-grid dimensions for one Qwen Image output size.
- `QwenImageSchedulerConfig` (dataclass): Strict scheduler metadata contract for supported Qwen Image payloads.
- `qwen_image_flow_euler_step` (function): Apply one deterministic FlowMatch Euler update in FP32.
- `qwen_image_flow_schedule` (function): Build the exact shifted/stretched Qwen Image sigma ladder.
- `qwen_image_flow_shift` (function): Resolve dynamic FlowMatch shift from packed image sequence length.
- `qwen_image_flow_shift_for_dimensions` (function): Resolve dynamic FlowMatch shift from output dimensions.
- `qwen_image_latent_grid` (function): Derive Qwen Image latent/packed-grid dimensions.
- `qwen_image_scheduler_config_from_mapping` (function): Validate and convert scheduler metadata mappings.
- `qwen_image_sequence_length` (function): Derive packed image sequence length from output dimensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .config import QWEN_IMAGE_IMAGE_MULTIPLE, QWEN_IMAGE_PATCH_SIZE, QWEN_IMAGE_VAE_SCALE_FACTOR, validate_qwen_image_dimensions


@dataclass(frozen=True, slots=True)
class QwenImageFlowSchedule:
    sigmas: torch.Tensor
    timesteps: torch.Tensor

    def __post_init__(self) -> None:
        if self.sigmas.ndim != 1 or self.timesteps.ndim != 1:
            raise RuntimeError("Qwen Image flow schedule sigmas/timesteps must be one-dimensional.")
        if int(self.sigmas.shape[0]) != int(self.timesteps.shape[0]) + 1:
            raise RuntimeError(
                "Qwen Image flow schedule must carry one terminal sigma beyond its timesteps; "
                f"sigmas={int(self.sigmas.shape[0])} timesteps={int(self.timesteps.shape[0])}."
            )
        if self.sigmas.dtype is not torch.float32 or self.timesteps.dtype is not torch.float32:
            raise RuntimeError("Qwen Image flow schedule sigmas/timesteps must use torch.float32.")
        if self.sigmas.device != self.timesteps.device:
            raise RuntimeError("Qwen Image flow schedule sigmas/timesteps must share one device.")
        if not bool(torch.isfinite(self.sigmas).all().item()) or not bool(torch.isfinite(self.timesteps).all().item()):
            raise RuntimeError("Qwen Image flow schedule contains non-finite values.")
        if float(self.sigmas[-1].item()) != 0.0:
            raise RuntimeError("Qwen Image flow schedule terminal sigma must be exactly zero.")
        if bool((self.sigmas[:-1] <= self.sigmas[1:]).any().item()):
            raise RuntimeError("Qwen Image flow schedule sigmas must be strictly decreasing.")


@dataclass(frozen=True, slots=True)
class QwenImageLatentGrid:
    width: int
    height: int
    latent_width: int
    latent_height: int
    packed_width: int
    packed_height: int
    sequence_length: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Qwen Image dimensions must be positive")
        if self.latent_width <= 0 or self.latent_height <= 0:
            raise ValueError("Qwen Image latent dimensions must be positive")
        if self.packed_width <= 0 or self.packed_height <= 0:
            raise ValueError("Qwen Image packed dimensions must be positive")
        if self.sequence_length != self.packed_width * self.packed_height:
            raise ValueError("Qwen Image sequence_length must equal packed_width * packed_height")


@dataclass(frozen=True, slots=True)
class QwenImageSchedulerConfig:
    class_name: str = "FlowMatchEulerDiscreteScheduler"
    base_image_seq_len: int = 256
    max_image_seq_len: int = 8192
    base_shift: float = 0.5
    max_shift: float = 0.9
    num_train_timesteps: int = 1000
    shift: float = 1.0
    shift_terminal: float = 0.02
    use_dynamic_shifting: bool = True
    time_shift_type: str = "exponential"
    invert_sigmas: bool = False
    stochastic_sampling: bool = False
    use_beta_sigmas: bool = False
    use_exponential_sigmas: bool = False
    use_karras_sigmas: bool = False

    def __post_init__(self) -> None:
        if self.class_name != "FlowMatchEulerDiscreteScheduler":
            raise ValueError("Qwen Image scheduler class must be FlowMatchEulerDiscreteScheduler")
        if self.base_image_seq_len <= 0 or self.max_image_seq_len <= self.base_image_seq_len:
            raise ValueError("Qwen Image scheduler sequence-length bounds are invalid")
        if self.num_train_timesteps <= 0:
            raise ValueError("Qwen Image scheduler num_train_timesteps must be positive")
        for name, value in (
            ("base_shift", self.base_shift),
            ("max_shift", self.max_shift),
            ("shift", self.shift),
            ("shift_terminal", self.shift_terminal),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"Qwen Image scheduler {name} must be finite")
        if self.base_shift <= 0.0 or self.max_shift <= 0.0 or self.shift <= 0.0:
            raise ValueError("Qwen Image scheduler shifts must be positive")
        if self.time_shift_type != "exponential":
            raise ValueError("Qwen Image scheduler time_shift_type must be 'exponential'")
        if not self.use_dynamic_shifting:
            raise ValueError("Qwen Image scheduler must use dynamic shifting")
        if self.invert_sigmas or self.stochastic_sampling:
            raise ValueError("Qwen Image scheduler metadata enables an unsupported sigma mode")
        if self.use_beta_sigmas or self.use_exponential_sigmas or self.use_karras_sigmas:
            raise ValueError("Qwen Image scheduler metadata enables an unsupported alternate sigma ladder")


QWEN_IMAGE_SUPPORTED_SCHEDULER_CONFIG = QwenImageSchedulerConfig()


def _require_equal(config: Mapping[str, object], key: str, expected: object, *, context: str) -> object:
    actual = config.get(key)
    if actual != expected:
        raise RuntimeError(
            "Unsupported Qwen Image scheduler config for %s. Field %r expected %r, got %r."
            % (context, key, expected, actual)
        )
    return actual


def qwen_image_scheduler_config_from_mapping(
    config: Mapping[str, object],
    *,
    context: str = "Qwen Image scheduler metadata",
) -> QwenImageSchedulerConfig:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{context}: scheduler config must be a mapping.")

    supported = QWEN_IMAGE_SUPPORTED_SCHEDULER_CONFIG
    _require_equal(config, "_class_name", supported.class_name, context=context)
    _require_equal(config, "base_image_seq_len", supported.base_image_seq_len, context=context)
    _require_equal(config, "max_image_seq_len", supported.max_image_seq_len, context=context)
    _require_equal(config, "base_shift", supported.base_shift, context=context)
    _require_equal(config, "max_shift", supported.max_shift, context=context)
    _require_equal(config, "num_train_timesteps", supported.num_train_timesteps, context=context)
    _require_equal(config, "shift", supported.shift, context=context)
    _require_equal(config, "shift_terminal", supported.shift_terminal, context=context)
    _require_equal(config, "use_dynamic_shifting", supported.use_dynamic_shifting, context=context)
    _require_equal(config, "time_shift_type", supported.time_shift_type, context=context)
    _require_equal(config, "invert_sigmas", supported.invert_sigmas, context=context)
    _require_equal(config, "stochastic_sampling", supported.stochastic_sampling, context=context)
    _require_equal(config, "use_beta_sigmas", supported.use_beta_sigmas, context=context)
    _require_equal(config, "use_exponential_sigmas", supported.use_exponential_sigmas, context=context)
    _require_equal(config, "use_karras_sigmas", supported.use_karras_sigmas, context=context)
    return supported


def qwen_image_latent_grid(width: object, height: object) -> QwenImageLatentGrid:
    width_int, height_int = validate_qwen_image_dimensions(width, height, context="Qwen Image scheduler dimensions")
    latent_width = width_int // QWEN_IMAGE_VAE_SCALE_FACTOR
    latent_height = height_int // QWEN_IMAGE_VAE_SCALE_FACTOR
    if latent_width % QWEN_IMAGE_PATCH_SIZE != 0 or latent_height % QWEN_IMAGE_PATCH_SIZE != 0:
        raise RuntimeError(
            "Qwen Image latent dimensions must be divisible by patch size "
            f"{QWEN_IMAGE_PATCH_SIZE}; got {latent_width}x{latent_height}."
        )
    packed_width = latent_width // QWEN_IMAGE_PATCH_SIZE
    packed_height = latent_height // QWEN_IMAGE_PATCH_SIZE
    sequence_length = packed_width * packed_height
    if width_int % QWEN_IMAGE_IMAGE_MULTIPLE != 0 or height_int % QWEN_IMAGE_IMAGE_MULTIPLE != 0:
        raise RuntimeError(
            f"Qwen Image output dimensions must be divisible by {QWEN_IMAGE_IMAGE_MULTIPLE}; "
            f"got {width_int}x{height_int}."
        )
    return QwenImageLatentGrid(
        width=width_int,
        height=height_int,
        latent_width=latent_width,
        latent_height=latent_height,
        packed_width=packed_width,
        packed_height=packed_height,
        sequence_length=sequence_length,
    )


def qwen_image_sequence_length(width: object, height: object) -> int:
    return qwen_image_latent_grid(width, height).sequence_length


def qwen_image_flow_shift(
    image_seq_len: object,
    *,
    scheduler_config: QwenImageSchedulerConfig = QWEN_IMAGE_SUPPORTED_SCHEDULER_CONFIG,
) -> float:
    try:
        sequence_length = int(image_seq_len)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - strict runtime validation
        raise RuntimeError("Qwen Image image_seq_len must be an integer.") from exc
    if sequence_length <= 0:
        raise RuntimeError(f"Qwen Image image_seq_len must be positive; got {sequence_length}.")

    span = int(scheduler_config.max_image_seq_len) - int(scheduler_config.base_image_seq_len)
    if span <= 0:
        raise RuntimeError("Qwen Image scheduler sequence-length bounds are invalid.")
    slope = (float(scheduler_config.max_shift) - float(scheduler_config.base_shift)) / float(span)
    intercept = float(scheduler_config.base_shift) - slope * float(scheduler_config.base_image_seq_len)
    shift = float(sequence_length) * slope + intercept
    if not math.isfinite(shift) or shift <= 0.0:
        raise RuntimeError(f"Qwen Image dynamic shift resolved to an invalid value: {shift!r}.")
    return shift


def qwen_image_flow_shift_for_dimensions(
    width: object,
    height: object,
    *,
    scheduler_config: QwenImageSchedulerConfig = QWEN_IMAGE_SUPPORTED_SCHEDULER_CONFIG,
) -> float:
    return qwen_image_flow_shift(qwen_image_sequence_length(width, height), scheduler_config=scheduler_config)


def qwen_image_flow_schedule(
    num_inference_steps: object,
    *,
    image_seq_len: object,
    scheduler_config: QwenImageSchedulerConfig = QWEN_IMAGE_SUPPORTED_SCHEDULER_CONFIG,
    device: torch.device | str | None = None,
) -> QwenImageFlowSchedule:
    try:
        step_count = int(num_inference_steps)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - strict runtime validation
        raise RuntimeError("Qwen Image num_inference_steps must be an integer.") from exc
    if step_count < 2:
        raise RuntimeError(f"Qwen Image num_inference_steps must be at least 2; got {step_count}.")

    shift = qwen_image_flow_shift(image_seq_len, scheduler_config=scheduler_config)
    sigmas_array = np.linspace(
        1.0,
        1.0 / float(step_count),
        step_count,
    ).astype(np.float32)
    exponent = math.exp(shift)
    sigmas_array = exponent / (exponent + (1.0 / sigmas_array - 1.0))

    one_minus = 1.0 - sigmas_array
    terminal_denominator = 1.0 - float(scheduler_config.shift_terminal)
    if terminal_denominator <= 0.0:
        raise RuntimeError(
            "Qwen Image scheduler shift_terminal must be less than 1.0; "
            f"got {scheduler_config.shift_terminal!r}."
        )
    scale_factor = one_minus[-1] / terminal_denominator
    if not bool(np.isfinite(scale_factor)) or float(scale_factor) <= 0.0:
        raise RuntimeError(f"Qwen Image scheduler terminal stretch scale is invalid: {scale_factor!r}.")
    sigmas_array = 1.0 - (one_minus / scale_factor)
    sigmas = torch.from_numpy(sigmas_array).to(dtype=torch.float32, device=device)
    timesteps = sigmas * float(scheduler_config.num_train_timesteps)
    sigmas = torch.cat((sigmas, torch.zeros(1, device=sigmas.device, dtype=torch.float32)))
    return QwenImageFlowSchedule(sigmas=sigmas, timesteps=timesteps)


def qwen_image_flow_euler_step(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    *,
    current_sigma: torch.Tensor,
    next_sigma: torch.Tensor,
) -> torch.Tensor:
    if tuple(model_output.shape) != tuple(sample.shape):
        raise RuntimeError(
            "Qwen Image Euler step requires matching model_output/sample shapes; "
            f"model_output={tuple(model_output.shape)} sample={tuple(sample.shape)}."
        )
    if current_sigma.numel() != 1 or next_sigma.numel() != 1:
        raise RuntimeError("Qwen Image Euler step sigmas must each contain exactly one value.")
    delta = next_sigma.to(device=sample.device, dtype=torch.float32) - current_sigma.to(
        device=sample.device,
        dtype=torch.float32,
    )
    previous = sample.to(dtype=torch.float32) + delta * model_output
    previous = previous.to(dtype=model_output.dtype)
    if not bool(torch.isfinite(previous).all().item()):
        raise RuntimeError("Qwen Image Euler step produced non-finite latents.")
    return previous


__all__ = [
    "QWEN_IMAGE_SUPPORTED_SCHEDULER_CONFIG",
    "QwenImageFlowSchedule",
    "QwenImageLatentGrid",
    "QwenImageSchedulerConfig",
    "qwen_image_flow_euler_step",
    "qwen_image_flow_schedule",
    "qwen_image_flow_shift",
    "qwen_image_flow_shift_for_dimensions",
    "qwen_image_latent_grid",
    "qwen_image_scheduler_config_from_mapping",
    "qwen_image_sequence_length",
]

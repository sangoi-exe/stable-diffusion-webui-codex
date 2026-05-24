"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Microsoft Lens FlowMatch scheduler metadata validation and schedule helper formulas.
Keeps empirical `mu`, explicit sigma ladder, and transformer timestep scaling under repo-owned code without importing Diffusers.

Symbols (top-level; keep in sync; no ghosts):
- `LensSchedulerConfig` (dataclass): Strict scheduler metadata contract for Lens folders.
- `compute_lens_empirical_mu` (function): Compute upstream Lens empirical FlowMatch `mu` from sequence length and step count.
- `lens_flow_mu_for_dimensions` (function): Compute Lens `mu` from bucketed width/height and step count.
- `lens_scheduler_config_from_mapping` (function): Validate scheduler metadata from HF-style config mappings.
- `lens_sigma_ladder` (function): Build the explicit upstream Lens sigma ladder.
- `scale_lens_timestep_for_transformer` (function): Apply the upstream Lens `/ 1000` transformer timestep scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .resolution import lens_sequence_length


@dataclass(frozen=True, slots=True)
class LensSchedulerConfig:
    class_name: str = "FlowMatchEulerDiscreteScheduler"
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    base_shift: float = 0.5
    max_shift: float = 1.15
    num_train_timesteps: int = 1000
    shift: float = 3.0
    shift_terminal: object | None = None
    use_dynamic_shifting: bool = True
    time_shift_type: str = "exponential"
    invert_sigmas: bool = False
    stochastic_sampling: bool = False
    use_beta_sigmas: bool = False
    use_exponential_sigmas: bool = False
    use_karras_sigmas: bool = False

    def __post_init__(self) -> None:
        if self.class_name != "FlowMatchEulerDiscreteScheduler":
            raise ValueError("Lens scheduler class must be FlowMatchEulerDiscreteScheduler")
        if self.base_image_seq_len <= 0 or self.max_image_seq_len <= self.base_image_seq_len:
            raise ValueError("Lens scheduler sequence-length bounds are invalid")
        if self.num_train_timesteps != 1000:
            raise ValueError("Lens scheduler num_train_timesteps must be 1000")
        for field_name, field_value in (
            ("base_shift", self.base_shift),
            ("max_shift", self.max_shift),
            ("shift", self.shift),
        ):
            if not math.isfinite(float(field_value)):
                raise ValueError(f"Lens scheduler {field_name} must be finite")
        if not self.use_dynamic_shifting:
            raise ValueError("Lens scheduler must use dynamic shifting")
        if self.time_shift_type != "exponential":
            raise ValueError("Lens scheduler time_shift_type must be 'exponential'")
        if self.shift_terminal is not None:
            raise ValueError("Lens scheduler shift_terminal must stay null")
        if self.invert_sigmas or self.stochastic_sampling:
            raise ValueError("Lens scheduler metadata enables an unsupported sigma mode")
        if self.use_beta_sigmas or self.use_exponential_sigmas or self.use_karras_sigmas:
            raise ValueError("Lens scheduler metadata enables an unsupported alternate sigma ladder")


LENS_SUPPORTED_SCHEDULER_CONFIG = LensSchedulerConfig()


def _require_equal(config: Mapping[str, object], key: str, expected: object, *, context: str) -> object:
    actual = config.get(key)
    if actual != expected:
        raise RuntimeError(
            "Unsupported Lens scheduler config for %s. Field %r expected %r, got %r."
            % (context, key, expected, actual)
        )
    return actual


def lens_scheduler_config_from_mapping(
    config: Mapping[str, object],
    *,
    context: str = "Lens scheduler metadata",
) -> LensSchedulerConfig:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{context}: scheduler config must be a mapping.")

    supported = LENS_SUPPORTED_SCHEDULER_CONFIG
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


def _require_positive_int(raw_value: object, *, context: str) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise RuntimeError(f"{context} must be a positive integer.")
    if raw_value <= 0:
        raise RuntimeError(f"{context} must be a positive integer; got {raw_value}.")
    return raw_value


def compute_lens_empirical_mu(image_seq_len: object, num_steps: object) -> float:
    sequence_length = _require_positive_int(image_seq_len, context="Lens image_seq_len")
    step_count = _require_positive_int(num_steps, context="Lens num_steps")
    high_length_slope = 0.00016927
    high_length_intercept = 0.45666666
    if sequence_length > 4300:
        return float(high_length_slope * sequence_length + high_length_intercept)

    low_step_slope = 8.73809524e-05
    low_step_intercept = 1.89833333
    mu_at_200_steps = high_length_slope * sequence_length + high_length_intercept
    mu_at_10_steps = low_step_slope * sequence_length + low_step_intercept
    step_slope = (mu_at_200_steps - mu_at_10_steps) / 190.0
    step_intercept = mu_at_200_steps - 200.0 * step_slope
    return float(step_slope * step_count + step_intercept)


def lens_flow_mu_for_dimensions(width: object, height: object, num_steps: object) -> float:
    return compute_lens_empirical_mu(lens_sequence_length(width, height), num_steps)


def lens_sigma_ladder(num_inference_steps: object) -> tuple[float, ...]:
    step_count = _require_positive_int(num_inference_steps, context="Lens num_inference_steps")
    if step_count == 1:
        return (1.0,)
    end_sigma = 1.0 / float(step_count)
    increment = (end_sigma - 1.0) / float(step_count - 1)
    return tuple(float(1.0 + increment * step_index) for step_index in range(step_count))


def scale_lens_timestep_for_transformer(timestep: object) -> object:
    try:
        return timestep / 1000  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - tensor-like values must support division
        raise RuntimeError("Lens transformer timestep must support division by 1000.") from exc


__all__ = [
    "LENS_SUPPORTED_SCHEDULER_CONFIG",
    "LensSchedulerConfig",
    "compute_lens_empirical_mu",
    "lens_flow_mu_for_dimensions",
    "lens_scheduler_config_from_mapping",
    "lens_sigma_ladder",
    "scale_lens_timestep_for_transformer",
]

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lightweight public Microsoft Lens runtime-family contract surface.
Exports variant, resolution, scheduler, and sampler skeleton helpers for the parked `lens` engine without importing heavy model/runtime classes.

Symbols (top-level; keep in sync; no ghosts):
- `LensResolutionBucket` (dataclass): Resolved official Lens output bucket.
- `LensSchedulerConfig` (dataclass): Strict Lens scheduler metadata contract.
- `LensVariantSpec` (dataclass): Immutable metadata for one supported Lens variant.
- `compute_lens_empirical_mu` (function): Compute the upstream Lens empirical FlowMatch `mu`.
- `lens_flow_mu_for_dimensions` (function): Compute Lens `mu` from dimensions and step count.
- `lens_resolution_bucket` (function): Resolve official Lens resolution buckets.
- `lens_scheduler_config_from_mapping` (function): Validate Lens scheduler metadata.
- `lens_sequence_length` (function): Derive Lens latent sequence length.
- `lens_sigma_ladder` (function): Build the upstream Lens sigma ladder.
- `lens_variant_spec` (function): Resolve a supported Lens variant spec.
- `require_lens_dimensions` (function): Validate dimensions against official buckets.
- `require_lens_variant` (function): Validate strict Lens variant values.
- `sample_lens_txt2img_not_implemented` (function): Fail-loud skeleton sampler hook.
- `scale_lens_timestep_for_transformer` (function): Apply Lens transformer timestep `/ 1000` scaling.
"""

from __future__ import annotations

from .config import (
    LENS_BASE_REPO_ID,
    LENS_BASE_VARIANT,
    LENS_DEFAULT_REPO_ID,
    LENS_DEFAULT_VARIANT,
    LENS_ENGINE_ID,
    LENS_EXTRAS_KEY,
    LENS_NOT_IMPLEMENTED_MESSAGE,
    LENS_RESOLUTION_BUCKETS,
    LENS_SUPPORTED_ASPECT_RATIOS,
    LENS_SUPPORTED_BASE_RESOLUTIONS,
    LENS_SUPPORTED_VARIANTS,
    LENS_TURBO_REPO_ID,
    LENS_TURBO_VARIANT,
    LENS_VARIANT_KEY,
    LensVariantSpec,
    lens_variant_spec,
    require_lens_variant,
)
from .resolution import LensResolutionBucket, lens_resolution_bucket, lens_sequence_length, require_lens_dimensions
from .sampler import sample_lens_txt2img_not_implemented
from .scheduler import (
    LensSchedulerConfig,
    compute_lens_empirical_mu,
    lens_flow_mu_for_dimensions,
    lens_scheduler_config_from_mapping,
    lens_sigma_ladder,
    scale_lens_timestep_for_transformer,
)

__all__ = [
    "LENS_BASE_REPO_ID",
    "LENS_BASE_VARIANT",
    "LENS_DEFAULT_REPO_ID",
    "LENS_DEFAULT_VARIANT",
    "LENS_ENGINE_ID",
    "LENS_EXTRAS_KEY",
    "LENS_NOT_IMPLEMENTED_MESSAGE",
    "LENS_RESOLUTION_BUCKETS",
    "LENS_SUPPORTED_ASPECT_RATIOS",
    "LENS_SUPPORTED_BASE_RESOLUTIONS",
    "LENS_SUPPORTED_VARIANTS",
    "LENS_TURBO_REPO_ID",
    "LENS_TURBO_VARIANT",
    "LENS_VARIANT_KEY",
    "LensResolutionBucket",
    "LensSchedulerConfig",
    "LensVariantSpec",
    "compute_lens_empirical_mu",
    "lens_flow_mu_for_dimensions",
    "lens_resolution_bucket",
    "lens_scheduler_config_from_mapping",
    "lens_sequence_length",
    "lens_sigma_ladder",
    "lens_variant_spec",
    "require_lens_dimensions",
    "require_lens_variant",
    "sample_lens_txt2img_not_implemented",
    "scale_lens_timestep_for_transformer",
]

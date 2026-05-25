"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lightweight public Microsoft Lens runtime-family contract surface.
Exports variant, quant-policy, resolution, scheduler, sampler skeleton helpers, and lazy bootstrap/text-feature helpers for the parked `lens` engine without importing heavy tokenizer/model modules during config-only imports.

Symbols (top-level; keep in sync; no ghosts):
- `LensResolutionBucket` (dataclass): Resolved official Lens output bucket.
- `LensSchedulerConfig` (dataclass): Strict Lens scheduler metadata contract.
- `LensVariantSpec` (dataclass): Immutable metadata for one supported Lens variant.
- `__getattr__` (function): Lazy resolver for bootstrap/tokenizer/text-feature exports.
- `compute_lens_empirical_mu` (function): Compute the upstream Lens empirical FlowMatch `mu`.
- `lens_flow_mu_for_dimensions` (function): Compute Lens `mu` from dimensions and step count.
- `lens_options_with_default_quant_policy` (function): Return Lens load options with the default internal dequant BF16 policy.
- `lens_resolution_bucket` (function): Resolve official Lens resolution buckets.
- `lens_scheduler_config_from_mapping` (function): Validate Lens scheduler metadata.
- `lens_sequence_length` (function): Derive Lens latent sequence length.
- `lens_sigma_ladder` (function): Build the upstream Lens sigma ladder.
- `lens_text_encoder_quant_policy_from_options` (function): Resolve the internal Lens text-encoder quant policy from load options.
- `lens_variant_spec` (function): Resolve a supported Lens variant spec.
- `require_lens_dimensions` (function): Validate dimensions against official buckets.
- `require_lens_text_encoder_quant_policy` (function): Validate the internal Lens text-encoder quant policy.
- `require_lens_variant` (function): Validate strict Lens variant values.
- `sample_lens_txt2img_not_implemented` (function): Fail-loud skeleton sampler hook.
- `scale_lens_timestep_for_transformer` (function): Apply Lens transformer timestep `/ 1000` scaling.
"""

from __future__ import annotations

from importlib import import_module

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
    LENS_TEXT_ENCODER_QUANT_POLICIES,
    LENS_TEXT_ENCODER_QUANT_POLICY_DEFAULT,
    LENS_TEXT_ENCODER_QUANT_POLICY_DEQUANT_BF16,
    LENS_TEXT_ENCODER_QUANT_POLICY_KEY,
    LENS_TURBO_REPO_ID,
    LENS_TURBO_VARIANT,
    LENS_VARIANT_KEY,
    LensVariantSpec,
    lens_options_with_default_quant_policy,
    lens_text_encoder_quant_policy_from_options,
    lens_variant_spec,
    require_lens_text_encoder_quant_policy,
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

_LAZY_EXPORTS = {
    "LensAlignedTextFeatures": ("apps.backend.runtime.families.lens.text_encoder", "LensAlignedTextFeatures"),
    "LensGptOssEncoder": ("apps.backend.runtime.families.lens.text_encoder", "LensGptOssEncoder"),
    "LensRenderedPrompt": ("apps.backend.runtime.families.lens.text_encoder", "LensRenderedPrompt"),
    "LensRuntimeBootstrapStatus": ("apps.backend.runtime.families.lens.bootstrap", "LensRuntimeBootstrapStatus"),
    "LensTextFeatures": ("apps.backend.runtime.families.lens.text_encoder", "LensTextFeatures"),
    "LensTokenizerBundle": ("apps.backend.runtime.families.lens.text_encoder", "LensTokenizerBundle"),
    "align_lens_text_features": ("apps.backend.runtime.families.lens.text_encoder", "align_lens_text_features"),
    "apply_lens_text_offset": ("apps.backend.runtime.families.lens.text_encoder", "apply_lens_text_offset"),
    "empty_negative_lens_text_features": (
        "apps.backend.runtime.families.lens.text_encoder",
        "empty_negative_lens_text_features",
    ),
    "load_lens_tokenizer": ("apps.backend.runtime.families.lens.text_encoder", "load_lens_tokenizer"),
    "probe_lens_runtime_bootstrap": ("apps.backend.runtime.families.lens.bootstrap", "probe_lens_runtime_bootstrap"),
    "render_lens_chat_prompt": ("apps.backend.runtime.families.lens.text_encoder", "render_lens_chat_prompt"),
    "repeat_lens_text_features": ("apps.backend.runtime.families.lens.text_encoder", "repeat_lens_text_features"),
    "tokenize_lens_prompt_texts": ("apps.backend.runtime.families.lens.text_encoder", "tokenize_lens_prompt_texts"),
    "validate_lens_selected_layers": ("apps.backend.runtime.families.lens.text_encoder", "validate_lens_selected_layers"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attribute_name = target
    return getattr(import_module(module_path), attribute_name)


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
    "LENS_TEXT_ENCODER_QUANT_POLICIES",
    "LENS_TEXT_ENCODER_QUANT_POLICY_DEFAULT",
    "LENS_TEXT_ENCODER_QUANT_POLICY_DEQUANT_BF16",
    "LENS_TEXT_ENCODER_QUANT_POLICY_KEY",
    "LENS_TURBO_REPO_ID",
    "LENS_TURBO_VARIANT",
    "LENS_VARIANT_KEY",
    "LensAlignedTextFeatures",
    "LensGptOssEncoder",
    "LensRenderedPrompt",
    "LensResolutionBucket",
    "LensRuntimeBootstrapStatus",
    "LensSchedulerConfig",
    "LensTextFeatures",
    "LensTokenizerBundle",
    "LensVariantSpec",
    "align_lens_text_features",
    "apply_lens_text_offset",
    "compute_lens_empirical_mu",
    "empty_negative_lens_text_features",
    "lens_flow_mu_for_dimensions",
    "lens_options_with_default_quant_policy",
    "lens_resolution_bucket",
    "lens_scheduler_config_from_mapping",
    "lens_sequence_length",
    "lens_sigma_ladder",
    "lens_text_encoder_quant_policy_from_options",
    "lens_variant_spec",
    "load_lens_tokenizer",
    "probe_lens_runtime_bootstrap",
    "render_lens_chat_prompt",
    "repeat_lens_text_features",
    "require_lens_dimensions",
    "require_lens_text_encoder_quant_policy",
    "require_lens_variant",
    "sample_lens_txt2img_not_implemented",
    "scale_lens_timestep_for_transformer",
    "tokenize_lens_prompt_texts",
    "validate_lens_selected_layers",
]

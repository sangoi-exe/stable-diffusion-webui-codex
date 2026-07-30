"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lightweight public Qwen Image Edit-2511 runtime-family contract surface.
Exports the Edit-only internal variant spec, geometry helpers, and scheduler metadata helpers for the single `qwen_image`
architecture family without importing heavy model/runtime classes.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageVariantSpec` (dataclass): Immutable metadata for one supported internal Qwen Image variant.
- `QwenImageFlowSchedule` (dataclass): Exact native FlowMatch inference sigma/timestep ladder.
- `QwenImageLatentGrid` (dataclass): Derived latent/packed-grid dimensions for Qwen Image scheduling.
- `QwenImageSchedulerConfig` (dataclass): Strict FlowMatch Euler scheduler metadata contract for Qwen Image.
- `QwenImageTransformerConfig` (dataclass): Strict Qwen Image transformer metadata contract.
- `qwen_image_edit_condition_dimensions` (function): Derive Edit-2511 processor condition-image dimensions.
- `qwen_image_edit_vae_dimensions` (function): Derive Edit-2511 VAE/reference-image dimensions.
- `qwen_image_flow_euler_step` (function): Apply one deterministic native FlowMatch Euler update.
- `qwen_image_flow_schedule` (function): Build the exact dynamically shifted native sigma/timestep ladder.
- `qwen_image_flow_shift` (function): Resolve Qwen Image dynamic FlowMatch shift from image sequence length.
- `qwen_image_flow_shift_for_dimensions` (function): Resolve Qwen Image dynamic shift from output dimensions.
- `qwen_image_latent_grid` (function): Derive Qwen Image latent/packed sequence geometry.
- `qwen_image_prompt_plan` (function): Render a prompt through the variant-owned Qwen Image template.
- `qwen_image_scheduler_config_from_mapping` (function): Validate scheduler metadata from HF-style config mappings.
- `qwen_image_sequence_length` (function): Derive packed image sequence length.
- `qwen_image_text_encoder_config_from_mapping` (function): Validate Qwen2.5-VL text-encoder metadata.
- `qwen_image_transformer_config_from_mapping` (function): Validate Qwen Image transformer metadata.
- `qwen_image_validate_external_vae_path` (function): Validate selected external Qwen Image VAE path/root/config.
- `qwen_image_variant_spec` (function): Resolve the immutable spec for a supported Qwen Image variant.
- `require_qwen_image_variant` (function): Validate an internal Qwen Image variant value.
- `validate_qwen_image_dimensions` (function): Enforce Qwen Image output-dimension divisibility.
"""

from __future__ import annotations

from .config import (
    QWEN_IMAGE_ARCHITECTURE_LABEL,
    QWEN_IMAGE_DEFAULT_TRUE_CFG,
    QWEN_IMAGE_DISTILLED_GUIDANCE,
    QWEN_IMAGE_EDIT_CONDITION_AREA,
    QWEN_IMAGE_EDIT_PIPELINE_CLASS,
    QWEN_IMAGE_EDIT_REPO_ID,
    QWEN_IMAGE_EDIT_VARIANT,
    QWEN_IMAGE_EDIT_VAE_AREA,
    QWEN_IMAGE_ENGINE_ID,
    QWEN_IMAGE_IMAGE_MULTIPLE,
    QWEN_IMAGE_LATENT_CHANNELS,
    QWEN_IMAGE_PATCH_SIZE,
    QWEN_IMAGE_PUBLIC_SAMPLER,
    QWEN_IMAGE_PUBLIC_SCHEDULER,
    QWEN_IMAGE_SUPPORTED_VARIANTS,
    QWEN_IMAGE_TOKENIZER_MAX_LENGTH,
    QWEN_IMAGE_TRANSFORMER_IN_CHANNELS,
    QWEN_IMAGE_VARIANT_KEY,
    QWEN_IMAGE_VAE_SCALE_FACTOR,
    QwenImageVariantSpec,
    qwen_image_edit_condition_dimensions,
    qwen_image_edit_vae_dimensions,
    qwen_image_variant_spec,
    require_qwen_image_variant,
    validate_qwen_image_dimensions,
)
from .scheduler import (
    QwenImageFlowSchedule,
    QwenImageLatentGrid,
    QwenImageSchedulerConfig,
    qwen_image_flow_euler_step,
    qwen_image_flow_schedule,
    qwen_image_flow_shift,
    qwen_image_flow_shift_for_dimensions,
    qwen_image_latent_grid,
    qwen_image_scheduler_config_from_mapping,
    qwen_image_sequence_length,
)
from .text_encoder import (
    QwenImagePromptPlan,
    QwenImageTextEncoderConfig,
    QwenImageVisionConfig,
    qwen_image_prompt_plan,
    qwen_image_text_encoder_config_from_mapping,
    qwen_image_validate_max_sequence_length,
)
from .transformer import QwenImageTransformerConfig, qwen_image_transformer_config_from_mapping
from .vae import qwen_image_validate_external_vae_path

__all__ = [
    "QWEN_IMAGE_ARCHITECTURE_LABEL",
    "QWEN_IMAGE_DEFAULT_TRUE_CFG",
    "QWEN_IMAGE_DISTILLED_GUIDANCE",
    "QWEN_IMAGE_EDIT_CONDITION_AREA",
    "QWEN_IMAGE_EDIT_PIPELINE_CLASS",
    "QWEN_IMAGE_EDIT_REPO_ID",
    "QWEN_IMAGE_EDIT_VARIANT",
    "QWEN_IMAGE_EDIT_VAE_AREA",
    "QWEN_IMAGE_ENGINE_ID",
    "QWEN_IMAGE_IMAGE_MULTIPLE",
    "QWEN_IMAGE_LATENT_CHANNELS",
    "QWEN_IMAGE_PATCH_SIZE",
    "QWEN_IMAGE_PUBLIC_SAMPLER",
    "QWEN_IMAGE_PUBLIC_SCHEDULER",
    "QWEN_IMAGE_SUPPORTED_VARIANTS",
    "QWEN_IMAGE_TOKENIZER_MAX_LENGTH",
    "QWEN_IMAGE_TRANSFORMER_IN_CHANNELS",
    "QWEN_IMAGE_VARIANT_KEY",
    "QWEN_IMAGE_VAE_SCALE_FACTOR",
    "QwenImageFlowSchedule",
    "QwenImageLatentGrid",
    "QwenImagePromptPlan",
    "QwenImageSchedulerConfig",
    "QwenImageTextEncoderConfig",
    "QwenImageTransformerConfig",
    "QwenImageVariantSpec",
    "QwenImageVisionConfig",
    "qwen_image_edit_condition_dimensions",
    "qwen_image_edit_vae_dimensions",
    "qwen_image_flow_euler_step",
    "qwen_image_flow_schedule",
    "qwen_image_flow_shift",
    "qwen_image_flow_shift_for_dimensions",
    "qwen_image_latent_grid",
    "qwen_image_prompt_plan",
    "qwen_image_scheduler_config_from_mapping",
    "qwen_image_sequence_length",
    "qwen_image_text_encoder_config_from_mapping",
    "qwen_image_transformer_config_from_mapping",
    "qwen_image_validate_external_vae_path",
    "qwen_image_validate_max_sequence_length",
    "qwen_image_variant_spec",
    "require_qwen_image_variant",
    "validate_qwen_image_dimensions",
]

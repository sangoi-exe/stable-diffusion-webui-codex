"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image Edit-2511 family constants, immutable runtime spec, and output geometry helpers.
Keeps the single-image `Qwen-Image-Edit-2511` img2img contract under the canonical `qwen_image` architecture family while
preserving prompt, CFG, tokenizer, and dimension contracts derived from the vendored Hugging Face metadata.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageVariantSpec` (dataclass): Immutable metadata for one supported Qwen Image variant.
- `qwen_image_edit_condition_dimensions` (function): Derive Edit-2511 processor condition-image dimensions.
- `qwen_image_edit_vae_dimensions` (function): Derive Edit-2511 VAE/reference-image dimensions.
- `qwen_image_variant_spec` (function): Return the immutable spec for a supported internal variant.
- `require_qwen_image_variant` (function): Validate and normalize an internal Qwen Image variant.
- `validate_qwen_image_dimensions` (function): Enforce Qwen Image output dimension invariants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

QWEN_IMAGE_ENGINE_ID = "qwen_image"
QWEN_IMAGE_ARCHITECTURE_LABEL = "Qwen-Image-2.0"
QWEN_IMAGE_VARIANT_KEY = "qwen_image_variant"

QWEN_IMAGE_EDIT_VARIANT = "edit_2511"
QWEN_IMAGE_SUPPORTED_VARIANTS = (QWEN_IMAGE_EDIT_VARIANT,)

QWEN_IMAGE_EDIT_REPO_ID = "Qwen/Qwen-Image-Edit-2511"
QWEN_IMAGE_EDIT_PIPELINE_CLASS = "QwenImageEditPlusPipeline"

QWEN_IMAGE_LATENT_CHANNELS = 16
QWEN_IMAGE_TRANSFORMER_IN_CHANNELS = 64
QWEN_IMAGE_CONTEXT_DIM = 3584
QWEN_IMAGE_PATCH_SIZE = 2
QWEN_IMAGE_VAE_SCALE_FACTOR = 8
QWEN_IMAGE_IMAGE_MULTIPLE = QWEN_IMAGE_VAE_SCALE_FACTOR * QWEN_IMAGE_PATCH_SIZE
QWEN_IMAGE_EDIT_DIMENSION_MULTIPLE = 32

QWEN_IMAGE_TOKENIZER_MAX_LENGTH = 1024
QWEN_IMAGE_DEFAULT_TRUE_CFG = 4.0
QWEN_IMAGE_DISTILLED_GUIDANCE = 1.0
QWEN_IMAGE_PUBLIC_SAMPLER = "euler"
QWEN_IMAGE_PUBLIC_SCHEDULER = "simple"

QWEN_IMAGE_EDIT_DEFAULT_WIDTH = 1024
QWEN_IMAGE_EDIT_DEFAULT_HEIGHT = 1024
QWEN_IMAGE_EDIT_DEFAULT_STEPS = 40
QWEN_IMAGE_EDIT_CONDITION_AREA = 384 * 384
QWEN_IMAGE_EDIT_VAE_AREA = 1024 * 1024

QWEN_IMAGE_EDIT_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain "
    "how the user's text instruction should alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
QWEN_IMAGE_EDIT_PROMPT_TEMPLATE_START_IDX = 64


@dataclass(frozen=True, slots=True)
class QwenImageVariantSpec:
    """Runtime metadata for one supported internal Qwen Image variant."""

    variant: str
    route_mode: str
    repo_id: str
    pipeline_class: str
    default_width: int
    default_height: int
    default_steps: int
    default_true_cfg: float
    prompt_template: str
    prompt_template_start_idx: int
    tokenizer_max_length: int = QWEN_IMAGE_TOKENIZER_MAX_LENGTH
    uses_edit_processor: bool = False
    condition_image_area: int | None = None
    vae_reference_area: int | None = None

    def __post_init__(self) -> None:
        if self.variant not in QWEN_IMAGE_SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported Qwen Image variant in spec: {self.variant!r}")
        if self.route_mode != "img2img":
            raise ValueError(f"Unsupported Qwen Image route_mode in spec: {self.route_mode!r}")
        validate_qwen_image_dimensions(self.default_width, self.default_height, context=f"{self.variant}.default_size")
        if self.default_steps <= 0:
            raise ValueError(f"{self.variant}.default_steps must be positive")
        if not math.isfinite(float(self.default_true_cfg)) or float(self.default_true_cfg) <= 0.0:
            raise ValueError(f"{self.variant}.default_true_cfg must be finite and positive")
        if not self.prompt_template:
            raise ValueError(f"{self.variant}.prompt_template must not be empty")
        if self.prompt_template_start_idx < 0:
            raise ValueError(f"{self.variant}.prompt_template_start_idx must be non-negative")
        if self.tokenizer_max_length != QWEN_IMAGE_TOKENIZER_MAX_LENGTH:
            raise ValueError(f"{self.variant}.tokenizer_max_length must be {QWEN_IMAGE_TOKENIZER_MAX_LENGTH}")
        if self.uses_edit_processor:
            if self.condition_image_area != QWEN_IMAGE_EDIT_CONDITION_AREA:
                raise ValueError(f"{self.variant}.condition_image_area must be {QWEN_IMAGE_EDIT_CONDITION_AREA}")
            if self.vae_reference_area != QWEN_IMAGE_EDIT_VAE_AREA:
                raise ValueError(f"{self.variant}.vae_reference_area must be {QWEN_IMAGE_EDIT_VAE_AREA}")
        elif self.condition_image_area is not None or self.vae_reference_area is not None:
            raise ValueError(f"{self.variant} must not declare edit image areas")


def validate_qwen_image_dimensions(width: object, height: object, *, context: str = "Qwen Image dimensions") -> tuple[int, int]:
    try:
        width_int = int(width)  # type: ignore[arg-type]
        height_int = int(height)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - strict public/runtime validation
        raise RuntimeError(f"{context}: width and height must be integers.") from exc

    if width_int <= 0 or height_int <= 0:
        raise RuntimeError(f"{context}: width and height must be positive; got {width_int}x{height_int}.")
    if width_int % QWEN_IMAGE_IMAGE_MULTIPLE != 0 or height_int % QWEN_IMAGE_IMAGE_MULTIPLE != 0:
        raise RuntimeError(
            f"{context}: width and height must be divisible by {QWEN_IMAGE_IMAGE_MULTIPLE}; "
            f"got {width_int}x{height_int}."
        )
    return width_int, height_int


QWEN_IMAGE_VARIANT_SPECS: Mapping[str, QwenImageVariantSpec] = {
    QWEN_IMAGE_EDIT_VARIANT: QwenImageVariantSpec(
        variant=QWEN_IMAGE_EDIT_VARIANT,
        route_mode="img2img",
        repo_id=QWEN_IMAGE_EDIT_REPO_ID,
        pipeline_class=QWEN_IMAGE_EDIT_PIPELINE_CLASS,
        default_width=QWEN_IMAGE_EDIT_DEFAULT_WIDTH,
        default_height=QWEN_IMAGE_EDIT_DEFAULT_HEIGHT,
        default_steps=QWEN_IMAGE_EDIT_DEFAULT_STEPS,
        default_true_cfg=QWEN_IMAGE_DEFAULT_TRUE_CFG,
        prompt_template=QWEN_IMAGE_EDIT_PROMPT_TEMPLATE,
        prompt_template_start_idx=QWEN_IMAGE_EDIT_PROMPT_TEMPLATE_START_IDX,
        uses_edit_processor=True,
        condition_image_area=QWEN_IMAGE_EDIT_CONDITION_AREA,
        vae_reference_area=QWEN_IMAGE_EDIT_VAE_AREA,
    ),
}


def require_qwen_image_variant(raw_variant: object, *, context: str = "qwen_image_variant") -> str:
    if not isinstance(raw_variant, str) or not raw_variant.strip():
        raise RuntimeError(f"{context} must be one of: {', '.join(QWEN_IMAGE_SUPPORTED_VARIANTS)}.")
    variant = raw_variant.strip()
    if variant not in QWEN_IMAGE_VARIANT_SPECS:
        raise RuntimeError(f"{context} must be one of: {', '.join(QWEN_IMAGE_SUPPORTED_VARIANTS)}.")
    return variant


def qwen_image_variant_spec(raw_variant: object) -> QwenImageVariantSpec:
    variant = require_qwen_image_variant(raw_variant)
    return QWEN_IMAGE_VARIANT_SPECS[variant]


def _dimensions_for_area(
    *,
    source_width: object,
    source_height: object,
    target_area: int,
    multiple: int,
    context: str,
) -> tuple[int, int]:
    try:
        source_width_int = int(source_width)  # type: ignore[arg-type]
        source_height_int = int(source_height)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - strict public/runtime validation
        raise RuntimeError(f"{context}: source width and height must be integers.") from exc

    if source_width_int <= 0 or source_height_int <= 0:
        raise RuntimeError(f"{context}: source width and height must be positive; got {source_width_int}x{source_height_int}.")
    if int(target_area) <= 0:
        raise RuntimeError(f"{context}: target_area must be positive; got {target_area!r}.")
    if int(multiple) <= 0:
        raise RuntimeError(f"{context}: multiple must be positive; got {multiple!r}.")

    ratio = float(source_width_int) / float(source_height_int)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise RuntimeError(f"{context}: invalid source aspect ratio for {source_width_int}x{source_height_int}.")

    width = int(round(math.sqrt(float(target_area) * ratio) / float(multiple)) * int(multiple))
    height = int(round((math.sqrt(float(target_area) * ratio) / ratio) / float(multiple)) * int(multiple))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"{context}: rounded dimensions are invalid for source {source_width_int}x{source_height_int}.")
    validate_qwen_image_dimensions(width, height, context=context)
    return width, height


def qwen_image_edit_condition_dimensions(source_width: object, source_height: object) -> tuple[int, int]:
    return _dimensions_for_area(
        source_width=source_width,
        source_height=source_height,
        target_area=QWEN_IMAGE_EDIT_CONDITION_AREA,
        multiple=QWEN_IMAGE_EDIT_DIMENSION_MULTIPLE,
        context="Qwen Image Edit condition image dimensions",
    )


def qwen_image_edit_vae_dimensions(source_width: object, source_height: object) -> tuple[int, int]:
    return _dimensions_for_area(
        source_width=source_width,
        source_height=source_height,
        target_area=QWEN_IMAGE_EDIT_VAE_AREA,
        multiple=QWEN_IMAGE_EDIT_DIMENSION_MULTIPLE,
        context="Qwen Image Edit VAE image dimensions",
    )


__all__ = [
    "QWEN_IMAGE_ARCHITECTURE_LABEL",
    "QWEN_IMAGE_CONTEXT_DIM",
    "QWEN_IMAGE_DEFAULT_TRUE_CFG",
    "QWEN_IMAGE_DISTILLED_GUIDANCE",
    "QWEN_IMAGE_EDIT_CONDITION_AREA",
    "QWEN_IMAGE_EDIT_DEFAULT_HEIGHT",
    "QWEN_IMAGE_EDIT_DEFAULT_STEPS",
    "QWEN_IMAGE_EDIT_DEFAULT_WIDTH",
    "QWEN_IMAGE_EDIT_DIMENSION_MULTIPLE",
    "QWEN_IMAGE_EDIT_PIPELINE_CLASS",
    "QWEN_IMAGE_EDIT_PROMPT_TEMPLATE",
    "QWEN_IMAGE_EDIT_PROMPT_TEMPLATE_START_IDX",
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
    "QWEN_IMAGE_VARIANT_SPECS",
    "QWEN_IMAGE_VAE_SCALE_FACTOR",
    "QwenImageVariantSpec",
    "qwen_image_edit_condition_dimensions",
    "qwen_image_edit_vae_dimensions",
    "qwen_image_variant_spec",
    "require_qwen_image_variant",
    "validate_qwen_image_dimensions",
]

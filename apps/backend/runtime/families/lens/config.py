"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Microsoft Lens family constants, internal variant specs, quant-policy identity, architecture invariants, and official resolution buckets.
Keeps `Lens`, `Lens-Turbo`, and `Lens-Base` as variants of the single parked `lens` engine while the bootstrap tranche proves GPT-OSS tokenizer/text-encoder contracts without exposing generation.

Symbols (top-level; keep in sync; no ghosts):
- `LensVariantSpec` (dataclass): Immutable metadata for one supported Lens variant.
- `lens_variant_spec` (function): Resolve the immutable spec for a supported Lens variant.
- `lens_text_encoder_quant_policy_from_options` (function): Resolve the internal Lens text-encoder quant policy from load options.
- `lens_options_with_default_quant_policy` (function): Return non-mutating Lens load options with the default dequant BF16 policy.
- `require_lens_text_encoder_quant_policy` (function): Validate the internal Lens text-encoder quant policy value.
- `require_lens_variant` (function): Validate a strict lowercase Lens variant value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

LENS_ENGINE_ID = "lens"
LENS_EXTRAS_KEY = "lens"
LENS_VARIANT_KEY = "lens_variant"
LENS_NOT_IMPLEMENTED_MESSAGE = "Lens txt2img runtime not yet implemented"
LENS_TEXT_ENCODER_QUANT_POLICY_KEY = "lens_text_encoder_quant_policy"
LENS_TEXT_ENCODER_QUANT_POLICY_DEQUANT_BF16 = "mxfp4_dequant_bf16"
LENS_TEXT_ENCODER_QUANT_POLICY_DEFAULT = LENS_TEXT_ENCODER_QUANT_POLICY_DEQUANT_BF16
LENS_TEXT_ENCODER_QUANT_POLICIES = (LENS_TEXT_ENCODER_QUANT_POLICY_DEQUANT_BF16,)

LENS_DEFAULT_VARIANT = "default"
LENS_TURBO_VARIANT = "turbo"
LENS_BASE_VARIANT = "base"
LENS_SUPPORTED_VARIANTS = (LENS_DEFAULT_VARIANT, LENS_TURBO_VARIANT, LENS_BASE_VARIANT)

LENS_DEFAULT_REPO_ID = "microsoft/Lens"
LENS_TURBO_REPO_ID = "microsoft/Lens-Turbo"
LENS_BASE_REPO_ID = "microsoft/Lens-Base"

LENS_PIPELINE_CLASS = "LensPipeline"
LENS_SCHEDULER_CLASS = "FlowMatchEulerDiscreteScheduler"
LENS_TEXT_ENCODER_CLASS = "LensGptOssEncoder"
LENS_TOKENIZER_CLASS = "PreTrainedTokenizerFast"
LENS_TRANSFORMER_CLASS = "LensTransformer2DModel"
LENS_VAE_CLASS = "AutoencoderKLFlux2"

LENS_VAE_SCALE_FACTOR = 16
LENS_VAE_LATENT_CHANNELS = 32
LENS_SEQUENCE_LATENT_CHANNELS = 128
LENS_TRANSFORMER_PATCH_SIZE = 2
LENS_TRANSFORMER_OUT_CHANNELS = 32
LENS_TRANSFORMER_NUM_LAYERS = 48
LENS_TRANSFORMER_ATTENTION_HEAD_DIM = 64
LENS_TRANSFORMER_NUM_ATTENTION_HEADS = 24
LENS_TRANSFORMER_INNER_DIM = 1536
LENS_TEXT_HIDDEN_DIM = 2880
LENS_TEXT_NUM_HIDDEN_LAYERS = 24
LENS_TEXT_MODEL_TYPE = "gpt_oss"
LENS_TEXT_OFFSET = 97
LENS_SELECTED_LAYER_INDEX = (5, 11, 17, 23)
LENS_AXES_DIMS_ROPE = (8, 28, 28)
LENS_MXFP4_QUANT_METHOD = "mxfp4"
LENS_TOKENIZER_BACKEND = "tokenizers"
LENS_TOKENIZER_CONFIG_CLASS = "TokenizersBackend"
LENS_MAX_SEQUENCE_LENGTH = 512

LENS_RESOLUTION_BUCKETS: Mapping[int, Mapping[str, tuple[int, int]]] = {
    1024: {
        "1:2": (1472, 736),
        "9:16": (1376, 768),
        "2:3": (1248, 832),
        "3:4": (1152, 864),
        "1:1": (1024, 1024),
        "4:3": (864, 1152),
        "3:2": (832, 1248),
        "16:9": (768, 1376),
        "2:1": (736, 1472),
    },
    1440: {
        "1:2": (2080, 1040),
        "9:16": (1936, 1088),
        "2:3": (1760, 1168),
        "3:4": (1616, 1216),
        "1:1": (1440, 1440),
        "4:3": (1216, 1616),
        "3:2": (1168, 1760),
        "16:9": (1088, 1936),
        "2:1": (1040, 2080),
    },
}
LENS_SUPPORTED_BASE_RESOLUTIONS = tuple(LENS_RESOLUTION_BUCKETS.keys())
LENS_SUPPORTED_ASPECT_RATIOS = tuple(LENS_RESOLUTION_BUCKETS[1024].keys())


@dataclass(frozen=True, slots=True)
class LensVariantSpec:
    """Runtime metadata for one supported internal Lens variant."""

    variant: str
    repo_id: str
    folder_name: str
    default_steps: int
    default_cfg: float

    def __post_init__(self) -> None:
        if self.variant not in LENS_SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported Lens variant in spec: {self.variant!r}")
        if not self.repo_id.startswith("microsoft/Lens"):
            raise ValueError(f"Lens repo_id must stay under microsoft/Lens, got {self.repo_id!r}")
        if not self.folder_name.startswith("Lens"):
            raise ValueError(f"Lens folder_name must start with Lens, got {self.folder_name!r}")
        if self.default_steps <= 0:
            raise ValueError(f"{self.variant}.default_steps must be positive")
        if self.default_cfg <= 0.0:
            raise ValueError(f"{self.variant}.default_cfg must be positive")


LENS_VARIANT_SPECS: Mapping[str, LensVariantSpec] = {
    LENS_DEFAULT_VARIANT: LensVariantSpec(
        variant=LENS_DEFAULT_VARIANT,
        repo_id=LENS_DEFAULT_REPO_ID,
        folder_name="Lens",
        default_steps=20,
        default_cfg=5.0,
    ),
    LENS_TURBO_VARIANT: LensVariantSpec(
        variant=LENS_TURBO_VARIANT,
        repo_id=LENS_TURBO_REPO_ID,
        folder_name="Lens-Turbo",
        default_steps=4,
        default_cfg=1.0,
    ),
    LENS_BASE_VARIANT: LensVariantSpec(
        variant=LENS_BASE_VARIANT,
        repo_id=LENS_BASE_REPO_ID,
        folder_name="Lens-Base",
        default_steps=50,
        default_cfg=5.0,
    ),
}


def _variant_error(context: str) -> RuntimeError:
    allowed = ", ".join(LENS_SUPPORTED_VARIANTS)
    return RuntimeError(f"{context} must be one of: {allowed}.")


def require_lens_variant(raw_variant: object, *, context: str = LENS_VARIANT_KEY) -> str:
    if not isinstance(raw_variant, str) or not raw_variant:
        raise _variant_error(context)
    if raw_variant != raw_variant.strip():
        raise _variant_error(context)
    if raw_variant not in LENS_VARIANT_SPECS:
        raise _variant_error(context)
    return raw_variant


def lens_variant_spec(raw_variant: object) -> LensVariantSpec:
    variant = require_lens_variant(raw_variant)
    return LENS_VARIANT_SPECS[variant]


def _quant_policy_error(context: str) -> RuntimeError:
    allowed = ", ".join(LENS_TEXT_ENCODER_QUANT_POLICIES)
    return RuntimeError(f"{context} must be one of: {allowed}.")


def require_lens_text_encoder_quant_policy(
    raw_policy: object,
    *,
    context: str = LENS_TEXT_ENCODER_QUANT_POLICY_KEY,
) -> str:
    if not isinstance(raw_policy, str) or not raw_policy:
        raise _quant_policy_error(context)
    if raw_policy != raw_policy.strip():
        raise _quant_policy_error(context)
    if raw_policy not in LENS_TEXT_ENCODER_QUANT_POLICIES:
        raise _quant_policy_error(context)
    return raw_policy


def lens_options_with_default_quant_policy(options: Mapping[str, object]) -> dict[str, object]:
    """Return a non-mutating Lens option mapping with the effective quant policy."""

    effective_options = dict(options)
    raw_policy = effective_options.get(LENS_TEXT_ENCODER_QUANT_POLICY_KEY)
    if raw_policy is None:
        effective_options[LENS_TEXT_ENCODER_QUANT_POLICY_KEY] = LENS_TEXT_ENCODER_QUANT_POLICY_DEFAULT
    else:
        effective_options[LENS_TEXT_ENCODER_QUANT_POLICY_KEY] = require_lens_text_encoder_quant_policy(
            raw_policy,
            context=f"Lens engine option {LENS_TEXT_ENCODER_QUANT_POLICY_KEY!r}",
        )
    return effective_options


def lens_text_encoder_quant_policy_from_options(options: Mapping[str, object]) -> str:
    return require_lens_text_encoder_quant_policy(
        options.get(LENS_TEXT_ENCODER_QUANT_POLICY_KEY),
        context=f"Lens engine option {LENS_TEXT_ENCODER_QUANT_POLICY_KEY!r}",
    )


__all__ = [
    "LENS_AXES_DIMS_ROPE",
    "LENS_BASE_REPO_ID",
    "LENS_BASE_VARIANT",
    "LENS_DEFAULT_REPO_ID",
    "LENS_DEFAULT_VARIANT",
    "LENS_ENGINE_ID",
    "LENS_EXTRAS_KEY",
    "LENS_MAX_SEQUENCE_LENGTH",
    "LENS_MXFP4_QUANT_METHOD",
    "LENS_NOT_IMPLEMENTED_MESSAGE",
    "LENS_PIPELINE_CLASS",
    "LENS_RESOLUTION_BUCKETS",
    "LENS_SCHEDULER_CLASS",
    "LENS_SELECTED_LAYER_INDEX",
    "LENS_SEQUENCE_LATENT_CHANNELS",
    "LENS_SUPPORTED_ASPECT_RATIOS",
    "LENS_SUPPORTED_BASE_RESOLUTIONS",
    "LENS_SUPPORTED_VARIANTS",
    "LENS_TEXT_ENCODER_CLASS",
    "LENS_TEXT_ENCODER_QUANT_POLICIES",
    "LENS_TEXT_ENCODER_QUANT_POLICY_DEFAULT",
    "LENS_TEXT_ENCODER_QUANT_POLICY_DEQUANT_BF16",
    "LENS_TEXT_ENCODER_QUANT_POLICY_KEY",
    "LENS_TEXT_HIDDEN_DIM",
    "LENS_TEXT_MODEL_TYPE",
    "LENS_TEXT_NUM_HIDDEN_LAYERS",
    "LENS_TEXT_OFFSET",
    "LENS_TOKENIZER_BACKEND",
    "LENS_TOKENIZER_CLASS",
    "LENS_TOKENIZER_CONFIG_CLASS",
    "LENS_TRANSFORMER_ATTENTION_HEAD_DIM",
    "LENS_TRANSFORMER_INNER_DIM",
    "LENS_TRANSFORMER_NUM_ATTENTION_HEADS",
    "LENS_TRANSFORMER_NUM_LAYERS",
    "LENS_TRANSFORMER_OUT_CHANNELS",
    "LENS_TRANSFORMER_PATCH_SIZE",
    "LENS_TURBO_REPO_ID",
    "LENS_TURBO_VARIANT",
    "LENS_VAE_CLASS",
    "LENS_VAE_LATENT_CHANNELS",
    "LENS_VAE_SCALE_FACTOR",
    "LENS_VARIANT_KEY",
    "LENS_VARIANT_SPECS",
    "LensVariantSpec",
    "lens_options_with_default_quant_policy",
    "lens_text_encoder_quant_policy_from_options",
    "lens_variant_spec",
    "require_lens_text_encoder_quant_policy",
    "require_lens_variant",
]

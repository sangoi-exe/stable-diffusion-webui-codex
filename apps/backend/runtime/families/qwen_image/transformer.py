"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image Edit-2511 transformer metadata validation.
Owns the lightweight exact `QwenImageTransformer2DModel` config contract without building or loading the heavyweight MMDiT transformer.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageTransformerConfig` (dataclass): Strict metadata contract for `QwenImageTransformer2DModel`.
- `qwen_image_transformer_config_from_mapping` (function): Validate and convert a transformer config mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import (
    QWEN_IMAGE_CONTEXT_DIM,
    QWEN_IMAGE_EDIT_VARIANT,
    QWEN_IMAGE_LATENT_CHANNELS,
    QWEN_IMAGE_PATCH_SIZE,
    QWEN_IMAGE_TRANSFORMER_IN_CHANNELS,
    require_qwen_image_variant,
)

QWEN_IMAGE_ATTENTION_HEAD_DIM = 128
QWEN_IMAGE_NUM_ATTENTION_HEADS = 24
QWEN_IMAGE_NUM_LAYERS = 60
QWEN_IMAGE_ROPE_AXES_DIMS = (16, 56, 56)


@dataclass(frozen=True, slots=True)
class QwenImageTransformerConfig:
    class_name: str
    variant: str
    attention_head_dim: int
    axes_dims_rope: tuple[int, ...]
    guidance_embeds: bool
    in_channels: int
    joint_attention_dim: int
    num_attention_heads: int
    num_layers: int
    out_channels: int
    patch_size: int
    zero_cond_t: bool | None = None

    def __post_init__(self) -> None:
        if self.class_name != "QwenImageTransformer2DModel":
            raise ValueError("Qwen Image transformer class must be QwenImageTransformer2DModel")
        require_qwen_image_variant(self.variant, context="Qwen Image transformer variant")
        if self.attention_head_dim != QWEN_IMAGE_ATTENTION_HEAD_DIM:
            raise ValueError(f"Qwen Image attention_head_dim must be {QWEN_IMAGE_ATTENTION_HEAD_DIM}")
        if self.axes_dims_rope != QWEN_IMAGE_ROPE_AXES_DIMS:
            raise ValueError(f"Qwen Image axes_dims_rope must be {QWEN_IMAGE_ROPE_AXES_DIMS}")
        if sum(self.axes_dims_rope) != self.attention_head_dim:
            raise ValueError("Qwen Image axes_dims_rope must sum to attention_head_dim")
        if self.guidance_embeds:
            raise ValueError("Qwen Image transformer guidance_embeds must be false in this tranche")
        if self.in_channels != QWEN_IMAGE_TRANSFORMER_IN_CHANNELS:
            raise ValueError(f"Qwen Image transformer in_channels must be {QWEN_IMAGE_TRANSFORMER_IN_CHANNELS}")
        if self.out_channels != QWEN_IMAGE_LATENT_CHANNELS:
            raise ValueError(f"Qwen Image transformer out_channels must be {QWEN_IMAGE_LATENT_CHANNELS}")
        if self.joint_attention_dim != QWEN_IMAGE_CONTEXT_DIM:
            raise ValueError(f"Qwen Image joint_attention_dim must be {QWEN_IMAGE_CONTEXT_DIM}")
        if self.num_attention_heads != QWEN_IMAGE_NUM_ATTENTION_HEADS:
            raise ValueError(f"Qwen Image num_attention_heads must be {QWEN_IMAGE_NUM_ATTENTION_HEADS}")
        if self.num_layers != QWEN_IMAGE_NUM_LAYERS:
            raise ValueError(f"Qwen Image num_layers must be {QWEN_IMAGE_NUM_LAYERS}")
        if self.patch_size != QWEN_IMAGE_PATCH_SIZE:
            raise ValueError(f"Qwen Image transformer patch_size must be {QWEN_IMAGE_PATCH_SIZE}")
        if self.variant != QWEN_IMAGE_EDIT_VARIANT:
            raise ValueError(f"Qwen Image transformer variant must be {QWEN_IMAGE_EDIT_VARIANT!r}")
        if self.zero_cond_t is not True:
            raise ValueError("Qwen Image Edit-2511 transformer must set zero_cond_t=true")


def _int_tuple(values: object, *, field: str, context: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise RuntimeError(f"{context}: {field} must be a sequence of integers.")
    result: list[int] = []
    for index, value in enumerate(values):
        try:
            result.append(int(value))
        except Exception as exc:  # noqa: BLE001 - strict metadata validation
            raise RuntimeError(f"{context}: {field}[{index}] must be an integer; got {value!r}.") from exc
    return tuple(result)


def _optional_zero_cond_t(value: object, *, context: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise RuntimeError(f"{context}: zero_cond_t must be true, false, or absent; got {value!r}.")


def _required_bool(config: Mapping[str, object], key: str, *, context: str) -> bool:
    value = config.get(key)
    if isinstance(value, bool):
        return value
    raise RuntimeError(f"{context}: {key} must be a boolean; got {value!r}.")


def qwen_image_transformer_config_from_mapping(
    config: Mapping[str, object],
    *,
    variant: object,
    context: str = "Qwen Image transformer metadata",
) -> QwenImageTransformerConfig:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{context}: transformer config must be a mapping.")
    variant_value = require_qwen_image_variant(variant, context=f"{context} variant")
    try:
        return QwenImageTransformerConfig(
            class_name=str(config.get("_class_name") or "").strip(),
            variant=variant_value,
            attention_head_dim=int(config.get("attention_head_dim") or 0),
            axes_dims_rope=_int_tuple(config.get("axes_dims_rope"), field="axes_dims_rope", context=context),
            guidance_embeds=_required_bool(config, "guidance_embeds", context=context),
            in_channels=int(config.get("in_channels") or 0),
            joint_attention_dim=int(config.get("joint_attention_dim") or 0),
            num_attention_heads=int(config.get("num_attention_heads") or 0),
            num_layers=int(config.get("num_layers") or 0),
            out_channels=int(config.get("out_channels") or 0),
            patch_size=int(config.get("patch_size") or 0),
            zero_cond_t=_optional_zero_cond_t(config.get("zero_cond_t"), context=context),
        )
    except ValueError as exc:
        raise RuntimeError(f"{context}: {exc}") from exc


__all__ = [
    "QWEN_IMAGE_ATTENTION_HEAD_DIM",
    "QWEN_IMAGE_NUM_ATTENTION_HEADS",
    "QWEN_IMAGE_NUM_LAYERS",
    "QWEN_IMAGE_ROPE_AXES_DIMS",
    "QwenImageTransformerConfig",
    "qwen_image_transformer_config_from_mapping",
]

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image Edit-2511 runtime tensor contracts, latent math, and device-side denoise validity flags.
Owns CPU-staged conditioning/latent value objects, native 2x2 latent pack/unpack, exact true-CFG norm rescaling, and
the private bit contract that defers CUDA validity reporting until the denoise stage's single terminal read.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageConditioning` (dataclass): Positive/negative multimodal embeddings plus explicit attention masks.
- `QwenImageDenoisedLatents` (dataclass): CPU-packed generated latents plus their output grid and seed metadata.
- `QwenImageReferenceLatents` (dataclass): CPU-packed reference-image latents plus their native grid.
- `_QwenImageDenoiseError` (IntFlag): Private device-side denoise validity bit contract.
- `qwen_image_pack_latents` (function): Pack 16-channel latents into native 2x2 image tokens.
- `qwen_image_true_cfg` (function): Return exact norm-rescaled true CFG plus device validity flags.
- `qwen_image_unpack_latents` (function): Restore packed image tokens to one-frame 5D VAE latents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntFlag
from typing import TYPE_CHECKING

import torch

from .config import (
    QWEN_IMAGE_CONTEXT_DIM,
    QWEN_IMAGE_DEFAULT_TRUE_CFG,
    QWEN_IMAGE_LATENT_CHANNELS,
    QWEN_IMAGE_TRANSFORMER_IN_CHANNELS,
)

if TYPE_CHECKING:
    from .scheduler import QwenImageLatentGrid


class _QwenImageDenoiseError(IntFlag):
    NONE = 0
    POSITIVE_PREDICTION_NONFINITE = 1
    CFG_NORM_NONFINITE = 2
    CFG_COMBINED_ZERO_NORM = 4
    CFG_RESULT_NONFINITE = 8
    EULER_RESULT_NONFINITE = 16


def _device_error_flag(
    condition: torch.Tensor,
    flag: _QwenImageDenoiseError,
) -> torch.Tensor:
    if condition.ndim != 0 or condition.dtype is not torch.bool:
        raise RuntimeError(
            "Qwen Image denoise validity conditions must be zero-dimensional bool tensors; "
            f"shape={tuple(condition.shape)} dtype={condition.dtype}."
        )
    return torch.where(
        condition,
        torch.full((), int(flag), device=condition.device, dtype=torch.int32),
        torch.zeros((), device=condition.device, dtype=torch.int32),
    )


def _require_cpu_tensor(value: torch.Tensor, *, label: str) -> None:
    if value.device.type != "cpu":
        raise RuntimeError(f"{label} must be staged on CPU; got device={value.device}.")


def _validate_conditioning_pair(embeddings: torch.Tensor, mask: torch.Tensor, *, label: str) -> None:
    if embeddings.ndim != 3 or tuple(embeddings.shape[:1]) != (1,) or int(embeddings.shape[-1]) != QWEN_IMAGE_CONTEXT_DIM:
        raise RuntimeError(
            f"{label} embeddings must have shape [1,S,{QWEN_IMAGE_CONTEXT_DIM}]; "
            f"got {tuple(embeddings.shape)}."
        )
    if tuple(mask.shape) != tuple(embeddings.shape[:2]):
        raise RuntimeError(
            f"{label} attention mask must match embedding batch/sequence dimensions; "
            f"mask={tuple(mask.shape)} embeddings={tuple(embeddings.shape)}."
        )
    if int(embeddings.shape[1]) <= 0:
        raise RuntimeError(f"{label} embeddings must retain at least one token after template removal.")
    if not bool(mask.to(dtype=torch.bool).all().item()):
        raise RuntimeError(f"{label} explicit attention mask must mark every retained single-image token valid.")
    if not bool(torch.isfinite(embeddings).all().item()):
        raise RuntimeError(f"{label} embeddings contain non-finite values.")
    _require_cpu_tensor(embeddings, label=f"{label} embeddings")
    _require_cpu_tensor(mask, label=f"{label} attention mask")


@dataclass(frozen=True, slots=True)
class QwenImageConditioning:
    positive_embeddings: torch.Tensor
    positive_mask: torch.Tensor
    negative_embeddings: torch.Tensor
    negative_mask: torch.Tensor
    condition_width: int
    condition_height: int

    def __post_init__(self) -> None:
        _validate_conditioning_pair(self.positive_embeddings, self.positive_mask, label="Qwen Image positive")
        _validate_conditioning_pair(self.negative_embeddings, self.negative_mask, label="Qwen Image negative")
        if self.condition_width <= 0 or self.condition_height <= 0:
            raise RuntimeError("Qwen Image condition dimensions must be positive.")


def _validate_packed_latents(
    packed_latents: torch.Tensor,
    grid: QwenImageLatentGrid,
    *,
    label: str,
) -> None:
    expected_shape = (1, int(grid.sequence_length), QWEN_IMAGE_TRANSFORMER_IN_CHANNELS)
    if tuple(packed_latents.shape) != expected_shape:
        raise RuntimeError(
            f"{label} packed latent shape mismatch: got={tuple(packed_latents.shape)} expected={expected_shape}."
        )
    if not bool(torch.isfinite(packed_latents).all().item()):
        raise RuntimeError(f"{label} packed latents contain non-finite values.")
    _require_cpu_tensor(packed_latents, label=f"{label} packed latents")


@dataclass(frozen=True, slots=True)
class QwenImageReferenceLatents:
    packed_latents: torch.Tensor
    grid: QwenImageLatentGrid

    def __post_init__(self) -> None:
        _validate_packed_latents(self.packed_latents, self.grid, label="Qwen Image reference")


@dataclass(frozen=True, slots=True)
class QwenImageDenoisedLatents:
    packed_latents: torch.Tensor
    grid: QwenImageLatentGrid
    seed: int
    steps: int

    def __post_init__(self) -> None:
        _validate_packed_latents(self.packed_latents, self.grid, label="Qwen Image generated")
        if self.steps < 2:
            raise RuntimeError("Qwen Image denoised step count must be at least 2.")


def qwen_image_pack_latents(latents: torch.Tensor) -> torch.Tensor:
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"Qwen Image latents must be a torch.Tensor; got {type(latents).__name__}.")
    if latents.ndim == 4:
        latents_5d = latents.unsqueeze(2)
    elif latents.ndim == 5:
        latents_5d = latents
    else:
        raise RuntimeError(
            "Qwen Image latent packing expects BCHW or BCTHW input; "
            f"got shape={tuple(latents.shape)}."
        )
    if int(latents_5d.shape[0]) != 1 or int(latents_5d.shape[1]) != QWEN_IMAGE_LATENT_CHANNELS:
        raise RuntimeError(
            "Qwen Image latent packing requires shape [1,16,1,H,W]; "
            f"got {tuple(latents_5d.shape)}."
        )
    if int(latents_5d.shape[2]) != 1:
        raise RuntimeError(
            "Qwen Image single-image latent packing requires temporal size 1; "
            f"got {int(latents_5d.shape[2])}."
        )
    height = int(latents_5d.shape[3])
    width = int(latents_5d.shape[4])
    if height <= 0 or width <= 0 or height % 2 != 0 or width % 2 != 0:
        raise RuntimeError(f"Qwen Image latent height/width must be positive even values; got {width}x{height}.")

    latents_4d = latents_5d[:, :, 0]
    packed = latents_4d.reshape(1, QWEN_IMAGE_LATENT_CHANNELS, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    packed = packed.reshape(1, (height // 2) * (width // 2), QWEN_IMAGE_TRANSFORMER_IN_CHANNELS)
    return packed.contiguous()


def qwen_image_unpack_latents(
    packed_latents: torch.Tensor,
    grid: QwenImageLatentGrid,
) -> torch.Tensor:
    if not isinstance(packed_latents, torch.Tensor):
        raise TypeError(
            f"Qwen Image packed latents must be a torch.Tensor; got {type(packed_latents).__name__}."
        )
    expected_shape = (1, int(grid.sequence_length), QWEN_IMAGE_TRANSFORMER_IN_CHANNELS)
    if tuple(packed_latents.shape) != expected_shape:
        raise RuntimeError(
            "Qwen Image latent unpack shape mismatch: "
            f"got={tuple(packed_latents.shape)} expected={expected_shape}."
        )
    unpacked = packed_latents.reshape(
        1,
        int(grid.packed_height),
        int(grid.packed_width),
        QWEN_IMAGE_LATENT_CHANNELS,
        2,
        2,
    )
    unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
    unpacked = unpacked.reshape(
        1,
        QWEN_IMAGE_LATENT_CHANNELS,
        1,
        int(grid.latent_height),
        int(grid.latent_width),
    )
    return unpacked.contiguous()


def qwen_image_true_cfg(
    positive_prediction: torch.Tensor,
    negative_prediction: torch.Tensor,
    *,
    scale: object = QWEN_IMAGE_DEFAULT_TRUE_CFG,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(positive_prediction.shape) != tuple(negative_prediction.shape):
        raise RuntimeError(
            "Qwen Image true CFG requires matching positive/negative prediction shapes; "
            f"positive={tuple(positive_prediction.shape)} negative={tuple(negative_prediction.shape)}."
        )
    if positive_prediction.device != negative_prediction.device:
        raise RuntimeError(
            "Qwen Image true CFG predictions must share one device; "
            f"positive={positive_prediction.device} negative={negative_prediction.device}."
        )
    try:
        resolved_scale = float(scale)
    except Exception as exc:  # noqa: BLE001 - strict runtime validation
        raise RuntimeError("Qwen Image true CFG scale must be numeric.") from exc
    if not math.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise RuntimeError(f"Qwen Image true CFG scale must be finite and positive; got {resolved_scale!r}.")

    combined = negative_prediction + resolved_scale * (positive_prediction - negative_prediction)
    positive_norm = torch.norm(positive_prediction, dim=-1, keepdim=True)
    combined_norm = torch.norm(combined, dim=-1, keepdim=True)
    norm_nonfinite = torch.logical_not(
        torch.logical_and(
            torch.isfinite(positive_norm).all(),
            torch.isfinite(combined_norm).all(),
        )
    )
    zero_combined_norm = (combined_norm == 0).any()
    safe_combined_norm = torch.where(
        combined_norm == 0,
        torch.ones_like(combined_norm),
        combined_norm,
    )
    result = combined * (positive_norm / safe_combined_norm)
    result_nonfinite = torch.logical_not(torch.isfinite(result).all())
    error_flags = torch.bitwise_or(
        _device_error_flag(norm_nonfinite, _QwenImageDenoiseError.CFG_NORM_NONFINITE),
        _device_error_flag(zero_combined_norm, _QwenImageDenoiseError.CFG_COMBINED_ZERO_NORM),
    )
    error_flags = torch.bitwise_or(
        error_flags,
        _device_error_flag(result_nonfinite, _QwenImageDenoiseError.CFG_RESULT_NONFINITE),
    )
    return result, error_flags


__all__ = [
    "QwenImageConditioning",
    "QwenImageDenoisedLatents",
    "QwenImageReferenceLatents",
    "qwen_image_pack_latents",
    "qwen_image_true_cfg",
    "qwen_image_unpack_latents",
]

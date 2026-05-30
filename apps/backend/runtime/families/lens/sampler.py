"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Reserved Microsoft Lens txt2img sampler owner.
Provides the named denoising seam for the parked Lens skeleton and fails loud until the real GPT-OSS/LensTransformer/VAE runtime is implemented.

Symbols (top-level; keep in sync; no ghosts):
- `LensProgressCallback` (type alias): Strict Lens sampler progress callback shape.
- `sample_lens_txt2img_not_implemented` (function): Keyword-only skeleton Lens sampler hook that raises the exact runtime-not-implemented error.
- `validate_lens_decoded_output` (function): Validate decoded Lens hook output before the canonical runner bypasses classic VAE decode.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from apps.backend.runtime.processing.datatypes import PromptContext, SamplingPlan

from .config import LENS_NOT_IMPLEMENTED_MESSAGE

LensProgressCallback = Callable[[Mapping[str, object]], None]


def _require_positive_int(raw_value: object, *, context: str) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise RuntimeError(f"{context} must be an integer.")
    if raw_value <= 0:
        raise RuntimeError(f"{context} must be positive; got {raw_value}.")
    return raw_value


def validate_lens_decoded_output(
    decoded: object,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Validate decoded Lens hook output before returning through `GenerationResult.decoded`."""

    batch_size_int = _require_positive_int(batch_size, context="Lens decoded output batch_size")
    height_int = _require_positive_int(height, context="Lens decoded output height")
    width_int = _require_positive_int(width, context="Lens decoded output width")
    expected_shape = (batch_size_int, 3, height_int, width_int)
    if not isinstance(decoded, torch.Tensor):
        raise RuntimeError(
            "Lens txt2img hook returned invalid decoded output; "
            f"type={type(decoded).__name__} shape={getattr(decoded, 'shape', None)} expected={expected_shape}."
        )
    actual_shape = tuple(int(dim) for dim in decoded.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            "Lens txt2img hook returned invalid decoded output shape; "
            f"shape={actual_shape} expected={expected_shape}."
        )
    return decoded


def sample_lens_txt2img_not_implemented(
    *,
    prompt_context: PromptContext,
    sampling_plan: SamplingPlan,
    width: int,
    height: int,
    batch_size: int,
    iterations: int,
    progress_callback: LensProgressCallback | None,
) -> torch.Tensor:
    del prompt_context, sampling_plan, width, height, batch_size, iterations, progress_callback
    raise NotImplementedError(LENS_NOT_IMPLEMENTED_MESSAGE)


__all__ = ["LensProgressCallback", "sample_lens_txt2img_not_implemented", "validate_lens_decoded_output"]

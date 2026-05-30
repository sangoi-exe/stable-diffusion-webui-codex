"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Microsoft Lens official output-resolution bucket validation and image sequence-length helpers.
Resolves Lens base-resolution/aspect-ratio buckets exactly as upstream (`W:H` keys, `(height, width)` values) without constructing a pipeline.

Symbols (top-level; keep in sync; no ghosts):
- `LensResolutionBucket` (dataclass): Resolved official Lens output bucket.
- `lens_resolution_bucket` (function): Resolve a `(base_resolution, aspect_ratio)` pair to the official bucket.
- `lens_sequence_length` (function): Derive Lens latent image sequence length for a bucketed output size.
- `require_lens_dimensions` (function): Validate that a width/height pair is one of the official Lens buckets.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import LENS_RESOLUTION_BUCKETS, LENS_SUPPORTED_ASPECT_RATIOS, LENS_SUPPORTED_BASE_RESOLUTIONS, LENS_VAE_SCALE_FACTOR


@dataclass(frozen=True, slots=True)
class LensResolutionBucket:
    base_resolution: int
    aspect_ratio: str
    height: int
    width: int

    @property
    def sequence_length(self) -> int:
        return lens_sequence_length(self.width, self.height)


def _require_int_value(raw_value: object, *, context: str) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise RuntimeError(f"{context} must be an integer.")
    return raw_value


def lens_resolution_bucket(base_resolution: object, aspect_ratio: object) -> LensResolutionBucket:
    base_resolution_int = _require_int_value(base_resolution, context="Lens base_resolution")
    if base_resolution_int not in LENS_RESOLUTION_BUCKETS:
        raise RuntimeError(
            "Lens base_resolution must be one of: "
            f"{', '.join(str(value) for value in LENS_SUPPORTED_BASE_RESOLUTIONS)}."
        )
    if not isinstance(aspect_ratio, str) or aspect_ratio not in LENS_SUPPORTED_ASPECT_RATIOS:
        raise RuntimeError(
            "Lens aspect_ratio must be one of: "
            f"{', '.join(LENS_SUPPORTED_ASPECT_RATIOS)}."
        )
    height, width = LENS_RESOLUTION_BUCKETS[base_resolution_int][aspect_ratio]
    return LensResolutionBucket(
        base_resolution=base_resolution_int,
        aspect_ratio=aspect_ratio,
        height=height,
        width=width,
    )


def lens_sequence_length(width: object, height: object) -> int:
    width_int = _require_int_value(width, context="Lens width")
    height_int = _require_int_value(height, context="Lens height")
    if width_int <= 0 or height_int <= 0:
        raise RuntimeError(f"Lens width and height must be positive; got {width_int}x{height_int}.")
    if width_int % LENS_VAE_SCALE_FACTOR or height_int % LENS_VAE_SCALE_FACTOR:
        raise RuntimeError(
            f"Lens width and height must be divisible by {LENS_VAE_SCALE_FACTOR}; got {width_int}x{height_int}."
        )
    return (width_int // LENS_VAE_SCALE_FACTOR) * (height_int // LENS_VAE_SCALE_FACTOR)


def require_lens_dimensions(width: object, height: object, *, context: str = "Lens dimensions") -> LensResolutionBucket:
    width_int = _require_int_value(width, context=f"{context}.width")
    height_int = _require_int_value(height, context=f"{context}.height")
    for base_resolution, buckets in LENS_RESOLUTION_BUCKETS.items():
        for aspect_ratio, dimensions in buckets.items():
            bucket_height, bucket_width = dimensions
            if width_int == bucket_width and height_int == bucket_height:
                return LensResolutionBucket(
                    base_resolution=base_resolution,
                    aspect_ratio=aspect_ratio,
                    height=bucket_height,
                    width=bucket_width,
                )
    raise RuntimeError(f"{context} must match one official Lens bucket; got {width_int}x{height_int}.")


__all__ = [
    "LensResolutionBucket",
    "lens_resolution_bucket",
    "lens_sequence_length",
    "require_lens_dimensions",
]

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Dependency-light tiled scaling utility.
Provides overlap+feather tiling for arbitrary single-image tensor functions (e.g. SR models, VAE encode/decode fallbacks)
with one final output accumulator and a single-channel blend-weight accumulator per active batch.

Symbols (top-level; keep in sync; no ghosts):
- `tiled_scale_multidim` (function): Multi-dim tiled scaling helper with single-channel feather-weight accumulation.
- `get_tiled_scale_steps` (function): Computes expected tile steps for progress.
- `tiled_scale` (function): Convenience wrapper for 2D tiling (tile_x/tile_y).
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Optional, Tuple

import torch


@torch.inference_mode()
def tiled_scale_multidim(
    samples: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    tile: Tuple[int, ...] = (64, 64),
    overlap: int = 8,
    upscale_amount: float = 4.0,
    out_channels: int = 3,
    output_device: str | torch.device = "cpu",
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    dims = len(tile)
    if dims <= 0:
        raise ValueError("tile must have at least one dimension")
    if samples.ndim < 2 + dims:
        raise ValueError(f"samples must have at least {2 + dims} dims for tile={tile}; got shape={tuple(samples.shape)}")

    scaled_spatial = list(map(lambda a: round(a * upscale_amount), samples.shape[2:]))
    output = torch.zeros([samples.shape[0], out_channels] + scaled_spatial, device=output_device)

    total_steps = 0
    try:
        total_steps = int(samples.shape[0]) * math.prod(
            [
                math.ceil((int(samples.shape[d + 2]) / (int(tile[d]) - int(overlap))))
                for d in range(dims)
            ]
        )
    except Exception:
        total_steps = 0

    step = 0
    for b in range(int(samples.shape[0])):
        s = samples[b : b + 1]
        out = output[b : b + 1]
        out_div = torch.zeros([1, 1] + scaled_spatial, device=output_device)

        ranges = []
        for shape_d, tile_d in zip(s.shape[2:], tile):
            step_stride = int(tile_d) - int(overlap)
            if step_stride <= 0:
                raise ValueError("tile size must be larger than overlap")
            ranges.append(range(0, int(shape_d), step_stride))

        for it in itertools.product(*ranges):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(int(s.shape[d + 2]) - int(overlap), int(it[d])))
                length = min(int(tile[d]), int(s.shape[d + 2]) - pos)
                s_in = s_in.narrow(d + 2, pos, length)
                upscaled.append(round(pos * upscale_amount))

            ps = function(s_in).to(output_device)

            mask = torch.ones_like(ps[:, :1])
            feather = round(int(overlap) * float(upscale_amount))
            for t in range(int(feather)):
                for d in range(2, dims + 2):
                    m = mask.narrow(d, t, 1)
                    m *= ((1.0 / feather) * (t + 1))
                    m = mask.narrow(d, int(mask.shape[d]) - 1 - t, 1)
                    m *= ((1.0 / feather) * (t + 1))

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, int(upscaled[d]), int(mask.shape[d + 2]))
                o_d = o_d.narrow(d + 2, int(upscaled[d]), int(mask.shape[d + 2]))

            o += ps * mask
            o_d += mask

            step += 1
            if progress_callback is not None:
                try:
                    progress_callback(step, total_steps)
                except Exception:
                    pass

        out.div_(out_div)

    return output


def get_tiled_scale_steps(width: int, height: int, *, tile_x: int, tile_y: int, overlap: int) -> int:
    stride_x = int(tile_x) - int(overlap)
    stride_y = int(tile_y) - int(overlap)
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError("tile must be larger than overlap")
    return math.ceil(int(height) / stride_y) * math.ceil(int(width) / stride_x)


def tiled_scale(
    samples: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    *,
    tile_x: int = 64,
    tile_y: int = 64,
    overlap: int = 8,
    upscale_amount: float = 4.0,
    out_channels: int = 3,
    output_device: str | torch.device = "cpu",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    return tiled_scale_multidim(
        samples,
        function,
        tile=(int(tile_y), int(tile_x)),
        overlap=int(overlap),
        upscale_amount=float(upscale_amount),
        out_channels=int(out_channels),
        output_device=output_device,
        progress_callback=progress_callback,
    )


__all__ = ["tiled_scale_multidim", "get_tiled_scale_steps", "tiled_scale"]

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native Qwen Image Edit-2511 rotary-position ownership.
Builds centered three-axis complex RoPE for the generated and single reference-image token grids as non-persistent
runtime buffers without adding checkpoint state or changing the checkpoint-owned tensor keyspace.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageRotaryEmbedding` (class): Centered three-axis complex RoPE for generated and reference-image token grids.
- `apply_qwen_image_rotary_embedding` (function): Apply precomputed complex RoPE to one Q/K tensor.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _rope_params(index: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    if int(dim) % 2:
        raise RuntimeError(f"Qwen Image RoPE axis dimension must be even; got {dim}.")
    exponent = torch.arange(0, int(dim), 2, dtype=torch.float32, device=index.device).div(float(dim))
    frequencies = torch.outer(index, 1.0 / torch.pow(float(theta), exponent))
    return torch.polar(torch.ones_like(frequencies), frequencies)


def _validate_shape_triplet(raw_shape: Sequence[object], *, context: str) -> tuple[int, int, int]:
    if len(raw_shape) != 3:
        raise RuntimeError(f"{context} must contain exactly (frames, height, width); got {raw_shape!r}.")
    try:
        frames, height, width = (int(value) for value in raw_shape)
    except Exception as exc:  # noqa: BLE001 - strict runtime shape contract
        raise RuntimeError(f"{context} must contain integer dimensions; got {raw_shape!r}.") from exc
    if frames <= 0 or height <= 0 or width <= 0:
        raise RuntimeError(f"{context} dimensions must be positive; got {(frames, height, width)!r}.")
    return frames, height, width


class QwenImageRotaryEmbedding(nn.Module):
    """Centered Qwen three-axis RoPE for one fixed-shape edit batch."""

    def __init__(
        self,
        *,
        theta: int = 10_000,
        axes_dim: Sequence[int] = (16, 56, 56),
        scale_rope: bool = True,
    ) -> None:
        super().__init__()
        self.theta = int(theta)
        self.axes_dim = tuple(int(value) for value in axes_dim)
        if len(self.axes_dim) != 3:
            raise RuntimeError(f"Qwen Image RoPE requires three axis dimensions; got {self.axes_dim!r}.")
        if any(value <= 0 or value % 2 for value in self.axes_dim):
            raise RuntimeError(f"Qwen Image RoPE axis dimensions must be positive and even; got {self.axes_dim!r}.")
        self.scale_rope = bool(scale_rope)

        positive_index = torch.arange(4096, dtype=torch.float32)
        negative_index = torch.arange(4096, dtype=torch.float32).flip(0) * -1 - 1
        self.register_buffer(
            "pos_freqs",
            torch.cat(
                tuple(_rope_params(positive_index, axis_dim, self.theta) for axis_dim in self.axes_dim),
                dim=1,
            ),
            persistent=False,
        )
        self.register_buffer(
            "neg_freqs",
            torch.cat(
                tuple(_rope_params(negative_index, axis_dim, self.theta) for axis_dim in self.axes_dim),
                dim=1,
            ),
            persistent=False,
        )

    def _image_freqs(
        self,
        shape: tuple[int, int, int],
        *,
        layer_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        frames, height, width = shape
        positive = self.pos_freqs.to(device=device)
        negative = self.neg_freqs.to(device=device)
        positive_axes = positive.split(tuple(value // 2 for value in self.axes_dim), dim=1)
        negative_axes = negative.split(tuple(value // 2 for value in self.axes_dim), dim=1)

        if layer_index + frames > int(positive_axes[0].shape[0]):
            raise RuntimeError(
                "Qwen Image RoPE frame/layer index exceeds the supported table: "
                f"layer_index={layer_index} frames={frames}."
            )
        frame_freqs = positive_axes[0][layer_index : layer_index + frames]
        frame_freqs = frame_freqs.view(frames, 1, 1, -1).expand(frames, height, width, -1)

        if self.scale_rope:
            if height > 4096 or width > 4096:
                raise RuntimeError(
                    "Qwen Image RoPE spatial dimensions exceed the supported table; "
                    f"got height={height} width={width}."
                )
            height_freqs = torch.cat(
                (
                    negative_axes[1][-(height - height // 2) :],
                    positive_axes[1][: height // 2],
                ),
                dim=0,
            )
            width_freqs = torch.cat(
                (
                    negative_axes[2][-(width - width // 2) :],
                    positive_axes[2][: width // 2],
                ),
                dim=0,
            )
        else:
            height_freqs = positive_axes[1][:height]
            width_freqs = positive_axes[2][:width]

        height_freqs = height_freqs.view(1, height, 1, -1).expand(frames, height, width, -1)
        width_freqs = width_freqs.view(1, 1, width, -1).expand(frames, height, width, -1)
        return torch.cat((frame_freqs, height_freqs, width_freqs), dim=-1).reshape(
            frames * height * width,
            -1,
        ).contiguous()

    def forward(
        self,
        image_shapes: Sequence[Sequence[Sequence[object]]],
        *,
        text_sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(image_shapes, Sequence) or isinstance(image_shapes, (str, bytes, bytearray)):
            raise RuntimeError("Qwen Image RoPE requires a batch sequence of image-shape sequences.")
        if not image_shapes:
            raise RuntimeError("Qwen Image RoPE image_shapes must not be empty.")

        normalized_batch: list[tuple[tuple[int, int, int], ...]] = []
        for batch_index, sample_shapes in enumerate(image_shapes):
            if not isinstance(sample_shapes, Sequence) or isinstance(sample_shapes, (str, bytes, bytearray)):
                raise RuntimeError(
                    f"Qwen Image RoPE image_shapes[{batch_index}] must be a sequence of shape triplets."
                )
            normalized = tuple(
                _validate_shape_triplet(raw_shape, context=f"Qwen Image RoPE image_shapes[{batch_index}][{shape_index}]")
                for shape_index, raw_shape in enumerate(sample_shapes)
            )
            if not normalized:
                raise RuntimeError(f"Qwen Image RoPE image_shapes[{batch_index}] must not be empty.")
            normalized_batch.append(normalized)

        first_shapes = normalized_batch[0]
        if any(sample_shapes != first_shapes for sample_shapes in normalized_batch[1:]):
            raise RuntimeError(
                "Qwen Image Edit-2511 does not support variable image-token geometry within one batch."
            )

        image_frequency_parts = [
            self._image_freqs(shape, layer_index=shape_index, device=device)
            for shape_index, shape in enumerate(first_shapes)
        ]
        image_frequencies = torch.cat(image_frequency_parts, dim=0)
        max_image_index = max(
            max(height // 2, width // 2) if self.scale_rope else max(height, width)
            for _frames, height, width in first_shapes
        )
        text_length = int(text_sequence_length)
        if text_length <= 0:
            raise RuntimeError(f"Qwen Image text_sequence_length must be positive; got {text_length}.")
        if max_image_index + text_length > int(self.pos_freqs.shape[0]):
            raise RuntimeError(
                "Qwen Image RoPE text span exceeds the supported table: "
                f"start={max_image_index} length={text_length} capacity={int(self.pos_freqs.shape[0])}."
            )
        text_frequencies = self.pos_freqs.to(device=device)[
            max_image_index : max_image_index + text_length
        ].contiguous()
        return image_frequencies, text_frequencies


def apply_qwen_image_rotary_embedding(
    hidden_states: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.ndim != 4:
        raise RuntimeError(
            "Qwen Image RoPE expects hidden states [B,S,H,D]; "
            f"got shape={tuple(hidden_states.shape)}."
        )
    if int(hidden_states.shape[1]) != int(frequencies.shape[0]):
        raise RuntimeError(
            "Qwen Image RoPE sequence mismatch: "
            f"hidden={int(hidden_states.shape[1])} frequencies={int(frequencies.shape[0])}."
        )
    if int(hidden_states.shape[-1]) % 2:
        raise RuntimeError(
            "Qwen Image RoPE head dimension must be even; "
            f"got {int(hidden_states.shape[-1])}."
        )
    complex_states = torch.view_as_complex(
        hidden_states.to(torch.float32).reshape(*hidden_states.shape[:-1], -1, 2)
    )
    rotated = complex_states * frequencies.unsqueeze(1)
    return torch.view_as_real(rotated).flatten(3).to(dtype=hidden_states.dtype)


__all__ = [
    "QwenImageRotaryEmbedding",
    "apply_qwen_image_rotary_embedding",
]

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image Edit-2511 transformer metadata validation and native streamed runtime model.
Owns the exact `QwenImageTransformer2DModel` config contract plus the Diffusers-free 60-block implementation whose
module names bind directly to the native 1,933-key GGUF checkpoint without changing stored tensor names, including
the shared per-block sampling progress callback, full-forward streamed execution lease, and one-block residency
context seams. Request-static image geometry, modulation indices, and rotary tensors are prepared once per distinct
text sequence length and validated at every forward boundary.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageTransformerConfig` (dataclass): Strict metadata contract for `QwenImageTransformer2DModel`.
- `QwenImageTransformerOutput` (dataclass): Structured native transformer output.
- `QwenImageTransformerRequestContext` (dataclass): Immutable request-static geometry/modulation/RoPE tensors.
- `QwenImageTransformer2DModel` (class): Native 60-block Edit-2511 dual-stream transformer.
- `qwen_image_transformer_config_from_mapping` (function): Validate and convert a transformer config mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import torch
from torch import nn

from apps.backend.runtime.sampling.block_progress import resolve_block_progress_callback

from .config import (
    QWEN_IMAGE_CONTEXT_DIM,
    QWEN_IMAGE_EDIT_VARIANT,
    QWEN_IMAGE_LATENT_CHANNELS,
    QWEN_IMAGE_PATCH_SIZE,
    QWEN_IMAGE_TRANSFORMER_IN_CHANNELS,
    require_qwen_image_variant,
)
from .transformer_layers import (
    QwenImageAdaLayerNormContinuous,
    QwenImageRMSNorm,
    QwenImageTimestepEmbedding,
    QwenImageTransformerBlock,
)
from .transformer_rope import QwenImageRotaryEmbedding

if TYPE_CHECKING:
    from .streaming import QwenImageStreamedCoreRuntime

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


@dataclass(frozen=True, slots=True)
class QwenImageTransformerOutput:
    sample: torch.Tensor


@dataclass(frozen=True, slots=True)
class QwenImageTransformerRequestContext:
    image_shapes: tuple[tuple[tuple[int, int, int], ...], ...]
    expected_image_tokens: tuple[int, ...]
    batch_size: int
    text_sequence_length: int
    compute_device: torch.device
    modulate_index: torch.Tensor
    rotary_embedding: tuple[torch.Tensor, torch.Tensor]

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or len(self.image_shapes) != self.batch_size:
            raise RuntimeError(
                "Qwen Image transformer request context batch mismatch: "
                f"batch={self.batch_size} shapes={len(self.image_shapes)}."
            )
        if len(self.expected_image_tokens) != self.batch_size:
            raise RuntimeError(
                "Qwen Image transformer request context token-count batch mismatch: "
                f"tokens={len(self.expected_image_tokens)} batch={self.batch_size}."
            )
        derived_image_tokens: list[int] = []
        for batch_index, sample_shapes in enumerate(self.image_shapes):
            if len(sample_shapes) != 2:
                raise RuntimeError(
                    "Qwen Image transformer request context requires exactly two image grids per sample; "
                    f"got {len(sample_shapes)} at batch index {batch_index}."
                )
            if any(len(shape) != 3 or any(value <= 0 for value in shape) for shape in sample_shapes):
                raise RuntimeError(
                    "Qwen Image transformer request context image shapes must be positive triplets; "
                    f"got {sample_shapes!r} at batch index {batch_index}."
                )
            derived_image_tokens.append(sum(math.prod(shape) for shape in sample_shapes))
        if tuple(derived_image_tokens) != self.expected_image_tokens:
            raise RuntimeError(
                "Qwen Image transformer request context image-token counts do not match image_shapes: "
                f"derived={tuple(derived_image_tokens)!r} declared={self.expected_image_tokens!r}."
            )
        if self.text_sequence_length <= 0:
            raise RuntimeError(
                "Qwen Image transformer request context text_sequence_length must be positive; "
                f"got {self.text_sequence_length}."
            )
        if not isinstance(self.compute_device, torch.device):
            raise TypeError(
                "Qwen Image transformer request context compute_device must be torch.device; "
                f"got {type(self.compute_device).__name__}."
            )
        if any(token_count <= 0 for token_count in self.expected_image_tokens):
            raise RuntimeError(
                "Qwen Image transformer request context image-token counts must be positive; "
                f"got {self.expected_image_tokens!r}."
            )
        if len(set(self.expected_image_tokens)) != 1:
            raise RuntimeError(
                "Qwen Image transformer request context requires equal image-token counts within one batch; "
                f"got {self.expected_image_tokens!r}."
            )
        image_token_count = self.expected_image_tokens[0]
        if tuple(self.modulate_index.shape) != (self.batch_size, image_token_count):
            raise RuntimeError(
                "Qwen Image transformer request modulation index shape mismatch: "
                f"got={tuple(self.modulate_index.shape)} expected={(self.batch_size, image_token_count)}."
            )
        if self.modulate_index.dtype is not torch.int64 or self.modulate_index.device != self.compute_device:
            raise RuntimeError(
                "Qwen Image transformer request modulation index must use int64 on the compute device; "
                f"dtype={self.modulate_index.dtype} device={self.modulate_index.device} "
                f"expected_device={self.compute_device}."
            )
        if not isinstance(self.rotary_embedding, tuple) or len(self.rotary_embedding) != 2:
            raise RuntimeError("Qwen Image transformer request rotary embedding must be a two-tensor tuple.")
        image_frequencies, text_frequencies = self.rotary_embedding
        if not isinstance(image_frequencies, torch.Tensor) or not isinstance(text_frequencies, torch.Tensor):
            raise RuntimeError("Qwen Image transformer request rotary embedding entries must be tensors.")
        if int(image_frequencies.shape[0]) != image_token_count:
            raise RuntimeError(
                "Qwen Image transformer request image RoPE length mismatch: "
                f"got={int(image_frequencies.shape[0])} expected={image_token_count}."
            )
        if int(text_frequencies.shape[0]) != self.text_sequence_length:
            raise RuntimeError(
                "Qwen Image transformer request text RoPE length mismatch: "
                f"got={int(text_frequencies.shape[0])} expected={self.text_sequence_length}."
            )
        for tensor_name, tensor in (
            ("image rotary frequencies", image_frequencies),
            ("text rotary frequencies", text_frequencies),
        ):
            if tensor.device != self.compute_device:
                raise RuntimeError(
                    f"Qwen Image transformer request {tensor_name} must be on {self.compute_device}; "
                    f"got {tensor.device}."
                )


def _normalized_image_shapes(
    image_shapes: object,
    *,
    batch_size: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    if not isinstance(image_shapes, Sequence) or isinstance(image_shapes, (str, bytes, bytearray)):
        raise RuntimeError("Qwen Image transformer request image_shapes must be a batch sequence.")
    if len(image_shapes) != int(batch_size):
        raise RuntimeError(
            "Qwen Image transformer request image_shapes batch mismatch: "
            f"got={len(image_shapes)} expected={int(batch_size)}."
        )
    normalized_batch: list[tuple[tuple[int, int, int], ...]] = []
    for batch_index, raw_sample in enumerate(image_shapes):
        if not isinstance(raw_sample, Sequence) or isinstance(raw_sample, (str, bytes, bytearray)):
            raise RuntimeError(
                f"Qwen Image transformer request image_shapes[{batch_index}] must be a sequence of shape triplets."
            )
        if len(raw_sample) != 2:
            raise RuntimeError(
                "Qwen Image Edit-2511 requires exactly two image-token grids per sample "
                f"(generated + one reference); got {len(raw_sample)} at batch index {batch_index}."
            )
        normalized_sample: list[tuple[int, int, int]] = []
        for shape_index, raw_shape in enumerate(raw_sample):
            if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes, bytearray)):
                raise RuntimeError(
                    "Qwen Image transformer request "
                    f"image_shapes[{batch_index}][{shape_index}] must be a shape triplet."
                )
            if len(raw_shape) != 3:
                raise RuntimeError(
                    "Qwen Image transformer request "
                    f"image_shapes[{batch_index}][{shape_index}] must contain three values."
                )
            try:
                shape = tuple(int(value) for value in raw_shape)
            except Exception as exc:  # noqa: BLE001 - strict runtime shape contract
                raise RuntimeError(
                    "Qwen Image transformer request "
                    f"image_shapes[{batch_index}][{shape_index}] must contain integers."
                ) from exc
            if any(value <= 0 for value in shape):
                raise RuntimeError(
                    f"Qwen Image transformer image-token shape must be positive; got {shape!r}."
                )
            normalized_sample.append(shape)
        normalized_batch.append(tuple(normalized_sample))
    return tuple(normalized_batch)


class QwenImageTransformer2DModel(nn.Module):
    """Native Qwen Image Edit-2511 transformer with exact checkpoint key ownership."""

    def __init__(self, config: QwenImageTransformerConfig) -> None:
        super().__init__()
        if not isinstance(config, QwenImageTransformerConfig):
            raise TypeError(
                "QwenImageTransformer2DModel requires QwenImageTransformerConfig; "
                f"got {type(config).__name__}."
            )
        self.config = config
        self.out_channels = int(config.out_channels)
        self.inner_dim = int(config.num_attention_heads) * int(config.attention_head_dim)
        self.zero_cond_t = bool(config.zero_cond_t)

        self.pos_embed = QwenImageRotaryEmbedding(
            theta=10_000,
            axes_dim=config.axes_dims_rope,
            scale_rope=True,
        )
        self.time_text_embed = QwenImageTimestepEmbedding(self.inner_dim)
        self.txt_norm = QwenImageRMSNorm(int(config.joint_attention_dim), eps=1e-6)
        self.img_in = nn.Linear(int(config.in_channels), self.inner_dim, bias=True)
        self.txt_in = nn.Linear(int(config.joint_attention_dim), self.inner_dim, bias=True)
        self.transformer_blocks = nn.ModuleList(
            QwenImageTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=int(config.num_attention_heads),
                attention_head_dim=int(config.attention_head_dim),
                zero_cond_t=self.zero_cond_t,
            )
            for _ in range(int(config.num_layers))
        )
        self.norm_out = QwenImageAdaLayerNormContinuous(self.inner_dim, eps=1e-6)
        self.proj_out = nn.Linear(
            self.inner_dim,
            int(config.patch_size) * int(config.patch_size) * self.out_channels,
            bias=True,
        )
        self._streamed_core_runtime: QwenImageStreamedCoreRuntime | None = None

    def set_streamed_core_runtime(self, runtime: QwenImageStreamedCoreRuntime) -> None:
        from .streaming import QwenImageStreamedCoreRuntime

        if not isinstance(runtime, QwenImageStreamedCoreRuntime):
            raise TypeError(
                "Qwen Image transformer streamed runtime must be QwenImageStreamedCoreRuntime; "
                f"got {type(runtime).__name__}."
            )
        if self._streamed_core_runtime is not None and self._streamed_core_runtime is not runtime:
            raise RuntimeError("Qwen Image transformer already owns a different streamed core runtime.")
        self._streamed_core_runtime = runtime

    @torch.inference_mode()
    def prepare_request_context(
        self,
        image_shapes: Sequence[Sequence[Sequence[object]]],
        *,
        batch_size: int,
        text_sequence_length: int,
        compute_device: torch.device,
    ) -> QwenImageTransformerRequestContext:
        if type(batch_size) is not int or batch_size <= 0:
            raise RuntimeError(
                "Qwen Image transformer request batch_size must be a positive exact int; "
                f"got {batch_size!r}."
            )
        if type(text_sequence_length) is not int or text_sequence_length <= 0:
            raise RuntimeError(
                "Qwen Image transformer request text_sequence_length must be a positive exact int; "
                f"got {text_sequence_length!r}."
            )
        if not isinstance(compute_device, torch.device):
            raise TypeError(
                "Qwen Image transformer request compute_device must be torch.device; "
                f"got {type(compute_device).__name__}."
            )
        normalized_shapes = _normalized_image_shapes(image_shapes, batch_size=batch_size)
        expected_image_tokens = tuple(
            sum(math.prod(shape) for shape in sample_shapes)
            for sample_shapes in normalized_shapes
        )
        if len(set(expected_image_tokens)) != 1:
            raise RuntimeError(
                "Qwen Image transformer request requires equal image-token counts within one batch; "
                f"got {expected_image_tokens!r}."
            )
        streamed_runtime = self._streamed_core_runtime
        if streamed_runtime is None:
            raise RuntimeError(
                "Qwen Image Edit-2511 transformer requires its explicit streamed core runtime before request prep."
            )
        with streamed_runtime.transformer_execution_lease() as residency:
            if residency.compute_device != compute_device:
                raise RuntimeError(
                    "Qwen Image transformer request compute device must match streamed residency; "
                    f"request={compute_device} residency={residency.compute_device}."
                )
            modulation_rows = [
                [0] * math.prod(sample_shapes[0])
                + [1] * sum(math.prod(shape) for shape in sample_shapes[1:])
                for sample_shapes in normalized_shapes
            ]
            modulate_index = torch.tensor(
                modulation_rows,
                device=compute_device,
                dtype=torch.int64,
            )
            rotary_embedding = self.pos_embed(
                normalized_shapes,
                text_sequence_length=text_sequence_length,
                device=compute_device,
            )
        return QwenImageTransformerRequestContext(
            image_shapes=normalized_shapes,
            expected_image_tokens=expected_image_tokens,
            batch_size=batch_size,
            text_sequence_length=text_sequence_length,
            compute_device=compute_device,
            modulate_index=modulate_index,
            rotary_embedding=rotary_embedding,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None,
        timestep: torch.Tensor,
        request_context: QwenImageTransformerRequestContext,
        guidance: torch.Tensor | None = None,
        attention_kwargs: Mapping[str, object] | None = None,
        transformer_options: Mapping[str, object] | None = None,
        controlnet_block_samples: object = None,
        additional_t_cond: object = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | QwenImageTransformerOutput:
        if guidance is not None:
            raise NotImplementedError("Qwen Image Edit-2511 guidance embeddings are not supported.")
        if attention_kwargs:
            raise NotImplementedError("Qwen Image Edit-2511 attention kwargs are not supported.")
        if controlnet_block_samples is not None:
            raise NotImplementedError("Qwen Image Edit-2511 ControlNet is not supported.")
        if additional_t_cond is not None:
            raise NotImplementedError("Qwen Image Edit-2511 additional timestep conditioning is not supported.")
        if hidden_states.ndim != 3 or int(hidden_states.shape[-1]) != int(self.config.in_channels):
            raise RuntimeError(
                "Qwen Image transformer hidden_states must have shape [B,S,64]; "
                f"got {tuple(hidden_states.shape)}."
            )
        if (
            encoder_hidden_states.ndim != 3
            or int(encoder_hidden_states.shape[0]) != int(hidden_states.shape[0])
            or int(encoder_hidden_states.shape[-1]) != int(self.config.joint_attention_dim)
        ):
            raise RuntimeError(
                "Qwen Image transformer encoder_hidden_states must have shape [B,T,3584] "
                f"with matching batch; got {tuple(encoder_hidden_states.shape)}."
            )
        if not isinstance(request_context, QwenImageTransformerRequestContext):
            raise TypeError(
                "Qwen Image transformer request_context must be QwenImageTransformerRequestContext; "
                f"got {type(request_context).__name__}."
            )

        batch_size = int(hidden_states.shape[0])
        if request_context.batch_size != batch_size:
            raise RuntimeError(
                "Qwen Image transformer request-context batch mismatch: "
                f"context={request_context.batch_size} hidden={batch_size}."
            )
        if any(
            token_count != int(hidden_states.shape[1])
            for token_count in request_context.expected_image_tokens
        ):
            raise RuntimeError(
                "Qwen Image transformer image-token geometry mismatch: "
                f"hidden_tokens={int(hidden_states.shape[1])} "
                f"shape_tokens={request_context.expected_image_tokens!r}."
            )
        text_sequence_length = int(encoder_hidden_states.shape[1])
        if request_context.text_sequence_length != text_sequence_length:
            raise RuntimeError(
                "Qwen Image transformer request-context text length mismatch: "
                f"context={request_context.text_sequence_length} hidden={text_sequence_length}."
            )
        if encoder_hidden_states_mask is None:
            raise RuntimeError(
                "Qwen Image transformer requires the explicit CPU-staged all-valid conditioning mask."
            )
        if tuple(encoder_hidden_states_mask.shape) != (batch_size, text_sequence_length):
            raise RuntimeError(
                "Qwen Image transformer encoder mask shape mismatch: "
                f"got={tuple(encoder_hidden_states_mask.shape)} "
                f"expected={(batch_size, text_sequence_length)}."
            )
        if encoder_hidden_states_mask.device.type != "cpu":
            raise RuntimeError(
                "Qwen Image transformer encoder mask validity must be proven by the CPU-staged "
                f"conditioning contract; got device={encoder_hidden_states_mask.device}."
            )

        if timestep.ndim == 0:
            timestep = timestep.reshape(1)
        if timestep.ndim != 1 or int(timestep.shape[0]) != batch_size:
            raise RuntimeError(
                "Qwen Image transformer timestep must have shape [B]; "
                f"got {tuple(timestep.shape)} for batch={batch_size}."
            )

        streamed_runtime = self._streamed_core_runtime
        if streamed_runtime is None:
            raise RuntimeError(
                "Qwen Image Edit-2511 transformer requires its explicit streamed core runtime before forward."
            )
        with streamed_runtime.transformer_execution_lease() as residency:
            if request_context.compute_device != residency.compute_device:
                raise RuntimeError(
                    "Qwen Image transformer request context no longer matches streamed compute residency: "
                    f"context={request_context.compute_device} residency={residency.compute_device}."
                )
            for tensor_name, tensor in (
                ("hidden_states", hidden_states),
                ("encoder_hidden_states", encoder_hidden_states),
            ):
                if tensor.device != residency.compute_device:
                    raise RuntimeError(
                        f"Qwen Image transformer {tensor_name} must be on the streamed compute device "
                        f"{residency.compute_device}; got {tensor.device}."
                    )

            image_hidden_states = self.img_in(hidden_states)
            timestep = timestep.to(device=image_hidden_states.device, dtype=image_hidden_states.dtype)
            generated_timestep = timestep
            if self.zero_cond_t:
                timestep = torch.cat((generated_timestep, torch.zeros_like(generated_timestep)), dim=0)
                modulate_index = request_context.modulate_index
            else:
                raise RuntimeError("Qwen Image Edit-2511 request context requires zero_cond_t=true.")

            text_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
            timestep_embedding = self.time_text_embed(
                timestep,
                hidden_dtype=image_hidden_states.dtype,
            )
            block_progress_callback = resolve_block_progress_callback(transformer_options)
            total_blocks = int(len(self.transformer_blocks))
            if block_progress_callback is not None and total_blocks <= 0:
                raise RuntimeError("Qwen Image transformer block progress requires at least one transformer block.")
            for block_index, block in enumerate(self.transformer_blocks):
                if block_progress_callback is not None:
                    block_progress_callback(int(block_index + 1), total_blocks)
                with streamed_runtime.activate_block(block_index):
                    text_hidden_states, image_hidden_states = block(
                        image_hidden_states,
                        text_hidden_states,
                        timestep_embedding=timestep_embedding,
                        rotary_embedding=request_context.rotary_embedding,
                        attention_mask=None,
                        modulate_index=modulate_index,
                    )

            if self.zero_cond_t:
                timestep_embedding = timestep_embedding.chunk(2, dim=0)[0]
            output = self.proj_out(self.norm_out(image_hidden_states, timestep_embedding))
            if not return_dict:
                return (output,)
            return QwenImageTransformerOutput(sample=output)


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
    "QwenImageTransformer2DModel",
    "QwenImageTransformerOutput",
    "QwenImageTransformerRequestContext",
    "qwen_image_transformer_config_from_mapping",
]

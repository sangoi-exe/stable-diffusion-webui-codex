"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image Qwen2.5-VL metadata, processor-batch, and base-model runtime helpers.
Validates the lightweight Qwen2.5-VL config contract, builds the exact Edit-2511 prompt template, and owns the typed
processor/base-model boundary without importing `transformers` at module import time.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageProcessorBatch` (dataclass): Exact four-tensor Qwen2-VL processor output used by Edit-2511.
- `QwenImagePromptPlan` (dataclass): Rendered prompt plus template-drop/max-sequence metadata for text encoding.
- `QwenImageTextEncoderConfig` (dataclass): Strict Qwen2.5-VL text-encoder metadata contract.
- `QwenImageTextEncoderRuntime` (class): Bound processor plus `Qwen2_5_VLModel` forward seam.
- `QwenImageVisionConfig` (dataclass): Strict Qwen2.5-VL visual-tower metadata contract.
- `qwen_image_prompt_plan` (function): Render a prompt through the variant-owned Qwen Image template.
- `qwen_image_text_encoder_config_from_mapping` (function): Validate and convert a text-encoder config mapping.
- `qwen_image_validate_max_sequence_length` (function): Enforce Qwen Image tokenizer max sequence length.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from .config import QWEN_IMAGE_CONTEXT_DIM, QWEN_IMAGE_TOKENIZER_MAX_LENGTH, qwen_image_variant_spec


class _QwenImageProcessor(Protocol):
    def __call__(
        self,
        *,
        text: list[str],
        images: object,
        padding: bool,
        return_tensors: str,
    ) -> Mapping[str, torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class QwenImageVisionConfig:
    model_type: str
    out_hidden_size: int
    patch_size: int

    def __post_init__(self) -> None:
        if self.model_type != "qwen2_5_vl":
            raise ValueError("Qwen Image vision config model_type must be qwen2_5_vl")
        if self.out_hidden_size != QWEN_IMAGE_CONTEXT_DIM:
            raise ValueError(f"Qwen Image vision out_hidden_size must be {QWEN_IMAGE_CONTEXT_DIM}")
        if self.patch_size != 14:
            raise ValueError("Qwen Image vision patch_size must be 14")


@dataclass(frozen=True, slots=True)
class QwenImageTextEncoderConfig:
    model_type: str
    hidden_size: int
    vision: QwenImageVisionConfig

    def __post_init__(self) -> None:
        if self.model_type != "qwen2_5_vl":
            raise ValueError("Qwen Image text encoder model_type must be qwen2_5_vl")
        if self.hidden_size != QWEN_IMAGE_CONTEXT_DIM:
            raise ValueError(f"Qwen Image text encoder hidden_size must be {QWEN_IMAGE_CONTEXT_DIM}")
        if self.vision.out_hidden_size != self.hidden_size:
            raise ValueError("Qwen Image vision projection must match text encoder hidden_size")


@dataclass(frozen=True, slots=True)
class QwenImagePromptPlan:
    prompt: str
    rendered_prompt: str
    template_start_idx: int
    max_sequence_length: int

    def __post_init__(self) -> None:
        if self.template_start_idx < 0:
            raise ValueError("Qwen Image prompt template_start_idx must be non-negative")
        qwen_image_validate_max_sequence_length(self.max_sequence_length)


@dataclass(frozen=True, slots=True)
class QwenImageProcessorBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor

    def __post_init__(self) -> None:
        for field_name, value in (
            ("input_ids", self.input_ids),
            ("attention_mask", self.attention_mask),
            ("pixel_values", self.pixel_values),
            ("image_grid_thw", self.image_grid_thw),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Qwen Image processor {field_name} must be a torch.Tensor.")
        if self.input_ids.ndim != 2 or int(self.input_ids.shape[0]) != 1:
            raise RuntimeError(
                "Qwen Image processor input_ids must have shape [1,S]; "
                f"got {tuple(self.input_ids.shape)}."
            )
        if tuple(self.attention_mask.shape) != tuple(self.input_ids.shape):
            raise RuntimeError(
                "Qwen Image processor attention_mask must match input_ids; "
                f"got mask={tuple(self.attention_mask.shape)} ids={tuple(self.input_ids.shape)}."
            )
        if self.pixel_values.ndim != 2 or int(self.pixel_values.shape[1]) != 1176:
            raise RuntimeError(
                "Qwen Image processor pixel_values must have shape [P,1176]; "
                f"got {tuple(self.pixel_values.shape)}."
            )
        if self.image_grid_thw.ndim != 2 or tuple(self.image_grid_thw.shape) != (1, 3):
            raise RuntimeError(
                "Qwen Image processor image_grid_thw must have shape [1,3]; "
                f"got {tuple(self.image_grid_thw.shape)}."
            )

    def to_model_inputs(
        self,
        *,
        device: torch.device,
        pixel_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids.to(device=device, dtype=torch.long),
            "attention_mask": self.attention_mask.to(device=device, dtype=torch.long),
            "pixel_values": self.pixel_values.to(device=device, dtype=pixel_dtype),
            "image_grid_thw": self.image_grid_thw.to(device=device, dtype=torch.long),
        }


class QwenImageTextEncoderRuntime:
    """Exact Qwen2.5-VL processor and base-model runtime boundary."""

    def __init__(
        self,
        *,
        model: nn.Module,
        processor: _QwenImageProcessor,
        compute_dtype: torch.dtype,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"Qwen Image text encoder model must be nn.Module; got {type(model).__name__}.")
        if not callable(processor):
            raise TypeError(
                f"Qwen Image processor must be callable; got {type(processor).__name__}."
            )
        if compute_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise RuntimeError(
                "Qwen Image text encoder compute dtype must be bf16, fp16, or fp32; "
                f"got {compute_dtype}."
            )
        self.model = model.eval()
        self.processor = processor
        self.compute_dtype = compute_dtype

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("Qwen Image text encoder model has no parameters.") from exc

    def prepare_processor_batch(
        self,
        *,
        prompt_plan: QwenImagePromptPlan,
        image: object,
    ) -> QwenImageProcessorBatch:
        if not isinstance(prompt_plan, QwenImagePromptPlan):
            raise TypeError(
                "Qwen Image processor requires QwenImagePromptPlan; "
                f"got {type(prompt_plan).__name__}."
            )
        raw_batch = self.processor(
            text=[prompt_plan.rendered_prompt],
            images=image,
            padding=True,
            return_tensors="pt",
        )
        if not isinstance(raw_batch, Mapping):
            raise RuntimeError(
                "Qwen Image processor must return a mapping; "
                f"got {type(raw_batch).__name__}."
            )
        expected_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        actual_keys = {str(key) for key in raw_batch.keys()}
        if actual_keys != expected_keys:
            raise RuntimeError(
                "Qwen Image processor output key mismatch. "
                f"missing={sorted(expected_keys - actual_keys)} extra={sorted(actual_keys - expected_keys)}."
            )
        values: dict[str, torch.Tensor] = {}
        for key in sorted(expected_keys):
            value = raw_batch[key]
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"Qwen Image processor output {key!r} must be a torch.Tensor; "
                    f"got {type(value).__name__}."
                )
            values[key] = value
        batch = QwenImageProcessorBatch(
            input_ids=values["input_ids"],
            attention_mask=values["attention_mask"],
            pixel_values=values["pixel_values"],
            image_grid_thw=values["image_grid_thw"],
        )
        sequence_length = int(batch.input_ids.shape[1])
        if sequence_length > prompt_plan.max_sequence_length:
            raise RuntimeError(
                "Qwen Image processor sequence exceeds the selected max_sequence_length. "
                f"got={sequence_length} max={prompt_plan.max_sequence_length}."
            )
        if prompt_plan.template_start_idx >= sequence_length:
            raise RuntimeError(
                "Qwen Image prompt template drop index must be smaller than the encoded sequence length. "
                f"drop={prompt_plan.template_start_idx} sequence={sequence_length}."
            )
        return batch

    @torch.inference_mode()
    def forward_processor_batch(self, batch: QwenImageProcessorBatch) -> torch.Tensor:
        if not isinstance(batch, QwenImageProcessorBatch):
            raise TypeError(
                "Qwen Image text encoder requires QwenImageProcessorBatch; "
                f"got {type(batch).__name__}."
            )
        model_inputs = batch.to_model_inputs(
            device=self.device,
            pixel_dtype=self.compute_dtype,
        )
        output = self.model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pixel_values=model_inputs["pixel_values"],
            image_grid_thw=model_inputs["image_grid_thw"],
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                "Qwen2_5_VLModel output must expose tensor last_hidden_state; "
                f"got {type(hidden_states).__name__}."
            )
        expected_shape = (
            int(batch.input_ids.shape[0]),
            int(batch.input_ids.shape[1]),
            QWEN_IMAGE_CONTEXT_DIM,
        )
        if tuple(hidden_states.shape) != expected_shape:
            raise RuntimeError(
                "Qwen Image text encoder hidden-state shape mismatch. "
                f"got={tuple(hidden_states.shape)} expected={expected_shape}."
            )
        if not bool(torch.isfinite(hidden_states).all().item()):
            raise RuntimeError("Qwen Image text encoder produced non-finite hidden states.")
        return hidden_states


def _require_mapping(value: object, *, field: str, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context}: {field} must be an object.")
    return value


def qwen_image_text_encoder_config_from_mapping(
    config: Mapping[str, object],
    *,
    context: str = "Qwen Image text encoder metadata",
) -> QwenImageTextEncoderConfig:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{context}: text encoder config must be a mapping.")
    vision_config = _require_mapping(config.get("vision_config"), field="vision_config", context=context)
    try:
        return QwenImageTextEncoderConfig(
            model_type=str(config.get("model_type") or "").strip(),
            hidden_size=int(config.get("hidden_size") or 0),
            vision=QwenImageVisionConfig(
                model_type=str(vision_config.get("model_type") or "").strip(),
                out_hidden_size=int(vision_config.get("out_hidden_size") or 0),
                patch_size=int(vision_config.get("patch_size") or 0),
            ),
        )
    except ValueError as exc:
        raise RuntimeError(f"{context}: {exc}") from exc


def qwen_image_validate_max_sequence_length(value: object) -> int:
    try:
        length = int(value)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - strict runtime validation
        raise RuntimeError("Qwen Image max_sequence_length must be an integer.") from exc
    if length <= 0:
        raise RuntimeError(f"Qwen Image max_sequence_length must be positive; got {length}.")
    if length > QWEN_IMAGE_TOKENIZER_MAX_LENGTH:
        raise RuntimeError(
            f"Qwen Image max_sequence_length cannot exceed {QWEN_IMAGE_TOKENIZER_MAX_LENGTH}; got {length}."
        )
    return length


def qwen_image_prompt_plan(
    prompt: object,
    *,
    variant: object,
    max_sequence_length: object = QWEN_IMAGE_TOKENIZER_MAX_LENGTH,
) -> QwenImagePromptPlan:
    spec = qwen_image_variant_spec(variant)
    prompt_text = "" if prompt is None else str(prompt)
    sequence_length = qwen_image_validate_max_sequence_length(max_sequence_length)
    return QwenImagePromptPlan(
        prompt=prompt_text,
        rendered_prompt=spec.prompt_template.format(prompt_text),
        template_start_idx=spec.prompt_template_start_idx,
        max_sequence_length=sequence_length,
    )


__all__ = [
    "QwenImageProcessorBatch",
    "QwenImagePromptPlan",
    "QwenImageTextEncoderConfig",
    "QwenImageTextEncoderRuntime",
    "QwenImageVisionConfig",
    "qwen_image_prompt_plan",
    "qwen_image_text_encoder_config_from_mapping",
    "qwen_image_validate_max_sequence_length",
]

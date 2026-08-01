"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image Qwen2.5-VL metadata, shared-image conditioning batches, and base-model runtime helpers.
Validates the lightweight Qwen2.5-VL config contract, builds the exact Edit-2511 prompt template, preprocesses one
condition image for positive/negative prompts, and reuses one visual-tower result across both language-model forwards
without importing `transformers` at module import time.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageProcessorBatch` (dataclass): Exact four-tensor Qwen2-VL processor output used by Edit-2511.
- `QwenImagePromptPlan` (dataclass): Rendered prompt plus template-drop/max-sequence metadata for text encoding.
- `QwenImageTextEncoderConfig` (dataclass): Strict Qwen2.5-VL text-encoder metadata contract.
- `QwenImageTextEncoderRuntime` (class): Shared-image processor plus one-vision/two-language-forward runtime seam.
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


class _QwenImageImageProcessor(Protocol):
    merge_size: int


class _QwenImageTokenizer(Protocol):
    def __call__(
        self,
        text: list[str],
        *,
        padding: bool,
        return_tensors: str,
    ) -> Mapping[str, torch.Tensor]: ...


class _QwenImageProcessor(Protocol):
    image_processor: _QwenImageImageProcessor
    image_token: str
    tokenizer: _QwenImageTokenizer

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

    @staticmethod
    def _tensor_mapping(
        value: object,
        *,
        expected_keys: set[str],
        context: str,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{context} must return a mapping; got {type(value).__name__}.")
        actual_keys = {str(key) for key in value.keys()}
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"{context} key mismatch. "
                f"missing={sorted(expected_keys - actual_keys)} extra={sorted(actual_keys - expected_keys)}."
            )
        tensors: dict[str, torch.Tensor] = {}
        for key in sorted(expected_keys):
            tensor = value[key]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"{context} output {key!r} must be a torch.Tensor; "
                    f"got {type(tensor).__name__}."
                )
            tensors[key] = tensor
        return tensors

    @staticmethod
    def _validate_prompt_batch(
        batch: QwenImageProcessorBatch,
        *,
        prompt_plan: QwenImagePromptPlan,
        label: str,
    ) -> None:
        sequence_length = int(batch.input_ids.shape[1])
        if sequence_length > prompt_plan.max_sequence_length:
            raise RuntimeError(
                f"{label} sequence exceeds the selected max_sequence_length. "
                f"got={sequence_length} max={prompt_plan.max_sequence_length}."
            )
        if prompt_plan.template_start_idx >= sequence_length:
            raise RuntimeError(
                f"{label} template drop index must be smaller than the encoded sequence length. "
                f"drop={prompt_plan.template_start_idx} sequence={sequence_length}."
            )

    def _tokenize_prompt_with_image_grid(
        self,
        *,
        prompt_plan: QwenImagePromptPlan,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_token = str(self.processor.image_token)
        if not image_token:
            raise RuntimeError("Qwen Image processor image_token must be non-empty.")
        image_count = prompt_plan.rendered_prompt.count(image_token)
        grid_count = int(image_grid_thw.shape[0])
        if image_count != grid_count:
            raise RuntimeError(
                "Qwen Image prompt/image-grid count mismatch during shared-image tokenization: "
                f"prompt_images={image_count} grid_rows={grid_count}."
            )
        merge_size = int(self.processor.image_processor.merge_size)
        if merge_size <= 0:
            raise RuntimeError(f"Qwen Image processor merge_size must be positive; got {merge_size}.")
        merge_length = merge_size * merge_size
        rendered_prompt = prompt_plan.rendered_prompt
        placeholder = "<|placeholder|>"
        for grid_index in range(grid_count):
            patch_count = int(image_grid_thw[grid_index].prod().item())
            if patch_count <= 0 or patch_count % merge_length != 0:
                raise RuntimeError(
                    "Qwen Image image-grid patch count must be positive and divisible by merge_size squared; "
                    f"patches={patch_count} merge_length={merge_length}."
                )
            image_token_count = patch_count // merge_length
            rendered_prompt = rendered_prompt.replace(
                image_token,
                placeholder * image_token_count,
                1,
            )
        rendered_prompt = rendered_prompt.replace(placeholder, image_token)
        tokenized = self._tensor_mapping(
            self.processor.tokenizer(
                [rendered_prompt],
                padding=True,
                return_tensors="pt",
            ),
            expected_keys={"input_ids", "attention_mask"},
            context="Qwen Image shared-image tokenizer",
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def prepare_conditioning_batches(
        self,
        *,
        positive_prompt_plan: QwenImagePromptPlan,
        negative_prompt_plan: QwenImagePromptPlan,
        image: object,
    ) -> tuple[QwenImageProcessorBatch, QwenImageProcessorBatch]:
        if not isinstance(positive_prompt_plan, QwenImagePromptPlan):
            raise TypeError(
                "Qwen Image positive processor plan must be QwenImagePromptPlan; "
                f"got {type(positive_prompt_plan).__name__}."
            )
        if not isinstance(negative_prompt_plan, QwenImagePromptPlan):
            raise TypeError(
                "Qwen Image negative processor plan must be QwenImagePromptPlan; "
                f"got {type(negative_prompt_plan).__name__}."
            )
        raw_batch = self.processor(
            text=[positive_prompt_plan.rendered_prompt],
            images=image,
            padding=True,
            return_tensors="pt",
        )
        expected_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        values = self._tensor_mapping(
            raw_batch,
            expected_keys=expected_keys,
            context="Qwen Image processor",
        )
        positive_batch = QwenImageProcessorBatch(
            input_ids=values["input_ids"],
            attention_mask=values["attention_mask"],
            pixel_values=values["pixel_values"],
            image_grid_thw=values["image_grid_thw"],
        )
        negative_input_ids, negative_attention_mask = self._tokenize_prompt_with_image_grid(
            prompt_plan=negative_prompt_plan,
            image_grid_thw=positive_batch.image_grid_thw,
        )
        negative_batch = QwenImageProcessorBatch(
            input_ids=negative_input_ids,
            attention_mask=negative_attention_mask,
            pixel_values=positive_batch.pixel_values,
            image_grid_thw=positive_batch.image_grid_thw,
        )
        self._validate_prompt_batch(
            positive_batch,
            prompt_plan=positive_prompt_plan,
            label="Qwen Image positive processor",
        )
        self._validate_prompt_batch(
            negative_batch,
            prompt_plan=negative_prompt_plan,
            label="Qwen Image negative tokenizer",
        )
        if negative_batch.pixel_values is not positive_batch.pixel_values:
            raise RuntimeError("Qwen Image conditioning batches must share the exact pixel_values tensor object.")
        if negative_batch.image_grid_thw is not positive_batch.image_grid_thw:
            raise RuntimeError("Qwen Image conditioning batches must share the exact image_grid_thw tensor object.")
        return positive_batch, negative_batch

    def _forward_text_batch(
        self,
        batch: QwenImageProcessorBatch,
        *,
        image_features: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = batch.input_ids.to(device=self.device, dtype=torch.long)
        attention_mask = batch.attention_mask.to(device=self.device, dtype=torch.long)
        get_input_embeddings = getattr(self.model, "get_input_embeddings", None)
        if not callable(get_input_embeddings):
            raise RuntimeError("Qwen2_5_VLModel must expose callable get_input_embeddings().")
        input_embedding = get_input_embeddings()
        if not callable(input_embedding):
            raise RuntimeError("Qwen2_5_VLModel input embedding must be callable.")
        inputs_embeds = input_embedding(input_ids)
        if not isinstance(inputs_embeds, torch.Tensor):
            raise RuntimeError(
                "Qwen2_5_VLModel input embedding must return a tensor; "
                f"got {type(inputs_embeds).__name__}."
            )
        batch_image_features = image_features.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        get_placeholder_mask = getattr(self.model, "get_placeholder_mask", None)
        if not callable(get_placeholder_mask):
            raise RuntimeError("Qwen2_5_VLModel must expose callable get_placeholder_mask().")
        placeholder_masks = get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=batch_image_features,
        )
        if not isinstance(placeholder_masks, tuple) or len(placeholder_masks) != 2:
            raise RuntimeError("Qwen2_5_VLModel placeholder-mask result must be a two-tensor tuple.")
        image_mask = placeholder_masks[0]
        if not isinstance(image_mask, torch.Tensor) or tuple(image_mask.shape) != tuple(inputs_embeds.shape):
            raise RuntimeError(
                "Qwen2_5_VLModel image placeholder mask must match inputs_embeds; "
                f"mask={getattr(image_mask, 'shape', None)} embeds={tuple(inputs_embeds.shape)}."
            )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, batch_image_features)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=None,
            image_grid_thw=image_grid_thw,
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

    @torch.inference_mode()
    def forward_conditioning_batches(
        self,
        positive_batch: QwenImageProcessorBatch,
        negative_batch: QwenImageProcessorBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(positive_batch, QwenImageProcessorBatch):
            raise TypeError(
                "Qwen Image text encoder positive batch must be QwenImageProcessorBatch; "
                f"got {type(positive_batch).__name__}."
            )
        if not isinstance(negative_batch, QwenImageProcessorBatch):
            raise TypeError(
                "Qwen Image text encoder negative batch must be QwenImageProcessorBatch; "
                f"got {type(negative_batch).__name__}."
            )
        if positive_batch.pixel_values is not negative_batch.pixel_values:
            raise RuntimeError("Qwen Image text encoder batches must share the exact pixel_values tensor object.")
        if positive_batch.image_grid_thw is not negative_batch.image_grid_thw:
            raise RuntimeError("Qwen Image text encoder batches must share the exact image_grid_thw tensor object.")
        pixel_values = positive_batch.pixel_values.to(
            device=self.device,
            dtype=self.compute_dtype,
        )
        image_grid_thw = positive_batch.image_grid_thw.to(device=self.device, dtype=torch.long)
        get_image_features = getattr(self.model, "get_image_features", None)
        if not callable(get_image_features):
            raise RuntimeError("Qwen2_5_VLModel must expose callable get_image_features().")
        raw_image_features = get_image_features(pixel_values, image_grid_thw)
        if not isinstance(raw_image_features, (tuple, list)) or not raw_image_features:
            raise RuntimeError("Qwen2_5_VLModel image features must be a non-empty tensor sequence.")
        image_feature_parts: list[torch.Tensor] = []
        for feature_index, feature in enumerate(raw_image_features):
            if not isinstance(feature, torch.Tensor):
                raise RuntimeError(
                    "Qwen2_5_VLModel image feature parts must be tensors; "
                    f"index={feature_index} got={type(feature).__name__}."
                )
            image_feature_parts.append(feature)
        image_features = torch.cat(tuple(image_feature_parts), dim=0)
        positive_hidden_states = self._forward_text_batch(
            positive_batch,
            image_features=image_features,
            image_grid_thw=image_grid_thw,
        )
        negative_hidden_states = self._forward_text_batch(
            negative_batch,
            image_features=image_features,
            image_grid_thw=image_grid_thw,
        )
        return positive_hidden_states, negative_hidden_states


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

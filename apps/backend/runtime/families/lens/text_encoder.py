"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Microsoft Lens GPT-OSS tokenizer, selected-layer text encoder, and prompt-feature helpers.
Loads the vendored GPT-OSS tokenizer sidecars directly through `PreTrainedTokenizerFast`, exposes a Lens-owned GPT-OSS selected-layer subclass, and validates offset/padding/repeat text-feature contracts without constructing Lens transformer/VAE/sampler runtime.

Symbols (top-level; keep in sync; no ghosts):
- `LensAlignedTextFeatures` (dataclass): Positive/negative text features padded to a shared Lens text length.
- `LensGptOssEncoder` (class): GPT-OSS subclass with Lens selected-layer feature extraction helpers.
- `LensRenderedPrompt` (dataclass): Rendered Lens chat-template text and tokenization prefix.
- `LensTextFeatures` (dataclass): Per-selected-layer feature tensors plus boolean text mask.
- `LensTokenizerBundle` (dataclass): Direct tokenizer sidecar load result.
- `align_lens_text_features` (function): Pad positive/negative Lens text features to a common length.
- `apply_lens_text_offset` (function): Apply Lens text offset `97` to selected-layer features and masks.
- `empty_negative_lens_text_features` (function): Build Lens empty-negative zero features without encoder execution.
- `lens_mxfp4_dequant_config` (function): Build the supported dequantized BF16 MXFP4 config.
- `load_lens_gpt_oss_encoder_dequant_bf16` (function): Load Lens GPT-OSS through local HF machinery with the internal dequant BF16 policy.
- `load_lens_tokenizer` (function): Build the Lens tokenizer directly from `tokenizer.json` plus sidecar metadata.
- `render_lens_chat_prompt` (function): Render the Lens system/user/assistant chat scaffold and split before `<|return|>`.
- `repeat_lens_text_features` (function): Repeat prompt-major features with `repeat_interleave`.
- `tokenize_lens_prompt_texts` (function): Tokenize rendered Lens prompt text with the fixed 512-token contract.
- `validate_lens_selected_layers` (function): Prove selected layers are unique and in range for the GPT-OSS config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from apps.backend.runtime.families.lens.config import (
    LENS_MAX_SEQUENCE_LENGTH,
    LENS_SELECTED_LAYER_INDEX,
    LENS_TEXT_OFFSET,
)

try:  # Keep import errors reportable by the bootstrap probe instead of hiding them.
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM as _GptOssForCausalLM
except Exception as _gpt_oss_import_error:  # noqa: BLE001 - stored for explicit bootstrap diagnostics
    _GPT_OSS_IMPORT_ERROR: Exception | None = _gpt_oss_import_error
    _GptOssForCausalLM = object  # type: ignore[assignment]
else:
    _GPT_OSS_IMPORT_ERROR = None

_CHAT_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
_CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
_LENS_RETURN_MARKER = "<|return|>"


@dataclass(frozen=True, slots=True)
class LensTokenizerBundle:
    tokenizer: Any
    tokenizer_dir: Path
    tokenizer_json_path: Path
    tokenizer_config_path: Path
    chat_template_path: Path
    clean_up_tokenization_spaces: bool


@dataclass(frozen=True, slots=True)
class LensRenderedPrompt:
    prompt: str
    rendered_text: str
    tokenization_text: str
    contains_return_marker: bool


@dataclass(frozen=True, slots=True)
class LensTextFeatures:
    features: tuple[torch.Tensor, ...]
    mask: torch.Tensor

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("Lens text features must contain at least one selected layer.")
        first_shape = self.features[0].shape
        if len(first_shape) != 3:
            raise ValueError(f"Lens text feature tensors must be [batch, seq, hidden], got {tuple(first_shape)}.")
        for layer_index, feature in enumerate(self.features):
            if feature.ndim != 3:
                raise ValueError(
                    f"Lens text feature layer {layer_index} must be rank 3, got rank {feature.ndim}."
                )
            if feature.shape[:2] != first_shape[:2]:
                raise ValueError(
                    f"Lens text feature layer {layer_index} batch/sequence shape {tuple(feature.shape[:2])} "
                    f"does not match layer 0 {tuple(first_shape[:2])}."
                )
        expected_mask_shape = first_shape[:2]
        if tuple(self.mask.shape) != tuple(expected_mask_shape):
            raise ValueError(
                f"Lens text mask shape {tuple(self.mask.shape)} does not match features {tuple(expected_mask_shape)}."
            )
        if self.mask.dtype != torch.bool:
            raise ValueError("Lens text mask must be boolean.")


@dataclass(frozen=True, slots=True)
class LensAlignedTextFeatures:
    positive: LensTextFeatures
    negative: LensTextFeatures


def _require_gpt_oss_available() -> None:
    if _GPT_OSS_IMPORT_ERROR is not None:
        raise RuntimeError("Lens GPT-OSS text encoder requires local Transformers GPT-OSS support.") from _GPT_OSS_IMPORT_ERROR


class LensGptOssEncoder(_GptOssForCausalLM):  # type: ignore[misc, valid-type]
    """GPT-OSS text encoder subclass for Lens selected-layer extraction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _require_gpt_oss_available()
        super().__init__(*args, **kwargs)
        self.set_selected_layers(LENS_SELECTED_LAYER_INDEX)

    def set_selected_layers(self, layer_indices: Sequence[int]) -> None:
        layer_values = validate_lens_selected_layers(len(self.model.layers), selected_layers=layer_indices)
        self._lens_selected_layers = layer_values
        self._lens_last_selected_layer = max(layer_values)

    @torch.no_grad()
    def forward(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if input_ids is None or attention_mask is None or args or kwargs or not hasattr(self, "_lens_selected_layers"):
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
        return self._selected_layer_forward(input_ids=input_ids, attention_mask=attention_mask)

    def _selected_layer_forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        try:
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        except Exception as exc:  # noqa: BLE001 - dependency gate should report local API drift
            raise RuntimeError("Lens GPT-OSS selected-layer extraction requires local Transformers masking utilities.") from exc

        model = self.model
        device = model.embed_tokens.weight.device
        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)
        hidden_states = model.embed_tokens(input_ids)
        cache_position = torch.arange(0, hidden_states.shape[1], device=device)
        position_ids = cache_position.unsqueeze(0)
        mask_inputs = {
            "config": model.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
        }
        attention_masks = {
            "full_attention": create_causal_mask(**mask_inputs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_inputs),
        }
        rotary_embeddings = model.rotary_emb(hidden_states, position_ids)
        selected_lookup = {layer_number: output_index for output_index, layer_number in enumerate(self._lens_selected_layers)}
        captured: list[torch.Tensor | None] = [None for _ in self._lens_selected_layers]
        for layer_number, decoder_layer in enumerate(model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_masks[decoder_layer.attention_type],
                position_embeddings=rotary_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
            )
            output_index = selected_lookup.get(layer_number)
            if output_index is not None:
                captured[output_index] = hidden_states
            if layer_number >= self._lens_last_selected_layer:
                break
        missing_layers = [str(layer) for layer, value in zip(self._lens_selected_layers, captured) if value is None]
        if missing_layers:
            raise RuntimeError(f"Lens GPT-OSS failed to capture selected layer(s): {', '.join(missing_layers)}.")
        return [value for value in captured if value is not None]

    def encode_layers(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        if not hasattr(self, "_lens_selected_layers"):
            raise RuntimeError("Lens GPT-OSS selected layers must be configured before encode_layers(...).")
        return self(input_ids=input_ids, attention_mask=attention_mask)


def validate_lens_selected_layers(
    num_hidden_layers: object,
    *,
    selected_layers: Sequence[int] = LENS_SELECTED_LAYER_INDEX,
) -> tuple[int, ...]:
    try:
        layer_count = int(num_hidden_layers)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - strict runtime validation
        raise RuntimeError("Lens GPT-OSS num_hidden_layers must be an integer.") from exc
    if layer_count <= 0:
        raise RuntimeError(f"Lens GPT-OSS num_hidden_layers must be positive; got {layer_count}.")
    layer_values = tuple(int(layer_index) for layer_index in selected_layers)
    if not layer_values:
        raise RuntimeError("Lens selected-layer contract must contain at least one layer.")
    if len(set(layer_values)) != len(layer_values):
        raise RuntimeError(f"Lens selected layers must be unique; got {layer_values!r}.")
    if min(layer_values) < 0 or max(layer_values) >= layer_count:
        raise RuntimeError(
            f"Lens selected layers {layer_values!r} are out of range for GPT-OSS layer count {layer_count}."
        )
    return layer_values


def _read_json_mapping(path: Path, *, context: str) -> Mapping[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - path/context surfaced in the raised error
        raise RuntimeError(f"{context}: failed to read JSON at {path}.") from exc
    if not isinstance(data, Mapping):
        raise RuntimeError(f"{context}: expected JSON object at {path}.")
    return data


def _require_string(config: Mapping[str, object], key: str, *, context: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context}: tokenizer_config field {key!r} must be a non-empty string.")
    return value


def load_lens_tokenizer(tokenizer_dir: str | Path) -> LensTokenizerBundle:
    try:
        from transformers import PreTrainedTokenizerFast
    except Exception as exc:  # noqa: BLE001 - explicit dependency gate
        raise RuntimeError("Lens tokenizer bootstrap requires transformers.PreTrainedTokenizerFast.") from exc

    directory = Path(tokenizer_dir).expanduser()
    tokenizer_json_path = directory / "tokenizer.json"
    tokenizer_config_path = directory / "tokenizer_config.json"
    chat_template_path = directory / "chat_template.jinja"
    if not tokenizer_json_path.is_file():
        raise RuntimeError(f"Lens tokenizer JSON not found: {tokenizer_json_path}")
    if not tokenizer_config_path.is_file():
        raise RuntimeError(f"Lens tokenizer config not found: {tokenizer_config_path}")
    if not chat_template_path.is_file():
        raise RuntimeError(f"Lens chat template not found: {chat_template_path}")

    tokenizer_config = _read_json_mapping(tokenizer_config_path, context="Lens tokenizer config")
    clean_up = bool(tokenizer_config.get("clean_up_tokenization_spaces", False))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path),
        bos_token=_require_string(tokenizer_config, "bos_token", context="Lens tokenizer config"),
        eos_token=_require_string(tokenizer_config, "eos_token", context="Lens tokenizer config"),
        pad_token=_require_string(tokenizer_config, "pad_token", context="Lens tokenizer config"),
        clean_up_tokenization_spaces=clean_up,
    )
    tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = LENS_MAX_SEQUENCE_LENGTH
    return LensTokenizerBundle(
        tokenizer=tokenizer,
        tokenizer_dir=directory,
        tokenizer_json_path=tokenizer_json_path,
        tokenizer_config_path=tokenizer_config_path,
        chat_template_path=chat_template_path,
        clean_up_tokenization_spaces=clean_up,
    )


def render_lens_chat_prompt(tokenizer: Any, prompt: object) -> LensRenderedPrompt:
    prompt_text = "" if prompt is None else str(prompt)
    conversation = [
        {"role": "system", "content": _CHAT_SYSTEM_PROMPT, "thinking": None},
        {"role": "user", "content": prompt_text, "thinking": None},
        {"role": "assistant", "thinking": _CHAT_ASSISTANT_THINKING, "content": ""},
    ]
    try:
        rendered_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    except Exception as exc:  # noqa: BLE001 - keep Lens context on tokenizer failures
        raise RuntimeError("Lens tokenizer failed to render the chat template.") from exc
    if not isinstance(rendered_text, str):
        raise RuntimeError("Lens tokenizer chat template returned a non-string result.")
    return LensRenderedPrompt(
        prompt=prompt_text,
        rendered_text=rendered_text,
        tokenization_text=rendered_text.split(_LENS_RETURN_MARKER, 1)[0],
        contains_return_marker=_LENS_RETURN_MARKER in rendered_text,
    )


def tokenize_lens_prompt_texts(tokenizer: Any, prompt_texts: Sequence[str]) -> Mapping[str, torch.Tensor]:
    if not prompt_texts:
        raise RuntimeError("Lens prompt tokenization requires at least one rendered prompt.")
    encoded = tokenizer(
        list(prompt_texts),
        padding=True,
        truncation=True,
        max_length=LENS_MAX_SEQUENCE_LENGTH,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise RuntimeError("Lens prompt tokenization did not return tensor input_ids and attention_mask.")
    if input_ids.shape[1] > LENS_MAX_SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Lens tokenized prompt length {input_ids.shape[1]} exceeds {LENS_MAX_SEQUENCE_LENGTH}."
        )
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def apply_lens_text_offset(
    layer_outputs: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    offset: int = LENS_TEXT_OFFSET,
) -> LensTextFeatures:
    if offset < 0:
        raise RuntimeError(f"Lens text offset must be non-negative; got {offset}.")
    if not layer_outputs:
        raise RuntimeError("Lens selected-layer outputs must not be empty.")
    base = layer_outputs[0]
    if base.ndim != 3:
        raise RuntimeError(f"Lens selected-layer outputs must be [batch, seq, hidden], got {tuple(base.shape)}.")
    if attention_mask.shape != base.shape[:2]:
        raise RuntimeError(
            f"Lens attention mask shape {tuple(attention_mask.shape)} does not match feature shape {tuple(base.shape[:2])}."
        )
    for layer_position, feature in enumerate(layer_outputs):
        if feature.ndim != 3 or feature.shape[:2] != base.shape[:2]:
            raise RuntimeError(
                f"Lens selected-layer output {layer_position} shape {tuple(feature.shape)} does not match layer 0."
            )
    if base.shape[1] > offset:
        features = tuple(feature[:, offset:, :].contiguous() for feature in layer_outputs)
        mask = attention_mask[:, offset:].bool()
    else:
        batch_size = base.shape[0]
        features = tuple(feature.new_zeros((batch_size, 0, feature.shape[-1])) for feature in layer_outputs)
        mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=base.device)
    return LensTextFeatures(features=features, mask=mask)


def empty_negative_lens_text_features(positive: LensTextFeatures) -> LensTextFeatures:
    return LensTextFeatures(
        features=tuple(feature.new_zeros(feature.shape) for feature in positive.features),
        mask=torch.zeros_like(positive.mask, dtype=torch.bool),
    )


def repeat_lens_text_features(text_features: LensTextFeatures, num_images_per_prompt: object) -> LensTextFeatures:
    try:
        repeat_count = int(num_images_per_prompt)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - strict validation
        raise RuntimeError("Lens num_images_per_prompt must be an integer.") from exc
    if repeat_count <= 0:
        raise RuntimeError(f"Lens num_images_per_prompt must be positive; got {repeat_count}.")
    if repeat_count == 1:
        return text_features
    return LensTextFeatures(
        features=tuple(feature.repeat_interleave(repeat_count, dim=0) for feature in text_features.features),
        mask=text_features.mask.repeat_interleave(repeat_count, dim=0),
    )


def _pad_lens_text_features(text_features: LensTextFeatures, target_length: int) -> LensTextFeatures:
    current_length = text_features.features[0].shape[1]
    if current_length == target_length:
        return text_features
    if current_length > target_length:
        raise RuntimeError("Lens text feature padding target cannot be shorter than the current length.")
    pad_length = target_length - current_length
    padded_features = tuple(
        torch.cat([feature, feature.new_zeros((feature.shape[0], pad_length, feature.shape[-1]))], dim=1)
        for feature in text_features.features
    )
    padded_mask = torch.cat(
        [
            text_features.mask,
            torch.zeros(
                (text_features.mask.shape[0], pad_length),
                dtype=torch.bool,
                device=text_features.mask.device,
            ),
        ],
        dim=1,
    )
    return LensTextFeatures(features=padded_features, mask=padded_mask)


def align_lens_text_features(positive: LensTextFeatures, negative: LensTextFeatures) -> LensAlignedTextFeatures:
    if len(positive.features) != len(negative.features):
        raise RuntimeError(
            "Lens positive and negative text features must have the same number of selected layers; "
            f"got {len(positive.features)} and {len(negative.features)}."
        )
    if positive.features[0].shape[0] != negative.features[0].shape[0]:
        raise RuntimeError(
            "Lens positive and negative text features must have the same batch size; "
            f"got {positive.features[0].shape[0]} and {negative.features[0].shape[0]}."
        )
    target_length = max(positive.features[0].shape[1], negative.features[0].shape[1])
    return LensAlignedTextFeatures(
        positive=_pad_lens_text_features(positive, target_length),
        negative=_pad_lens_text_features(negative, target_length),
    )


def lens_mxfp4_dequant_config() -> Any:
    try:
        from transformers import Mxfp4Config
    except Exception as exc:  # noqa: BLE001 - explicit dependency gate
        raise RuntimeError("Lens dequant BF16 text encoder requires transformers.Mxfp4Config.") from exc
    return Mxfp4Config(dequantize=True)


def load_lens_gpt_oss_encoder_dequant_bf16(model_ref: str | Path) -> LensGptOssEncoder:
    _require_gpt_oss_available()
    return LensGptOssEncoder.from_pretrained(
        str(Path(model_ref).expanduser()),
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
        quantization_config=lens_mxfp4_dequant_config(),
    )


__all__ = [
    "LensAlignedTextFeatures",
    "LensGptOssEncoder",
    "LensRenderedPrompt",
    "LensTextFeatures",
    "LensTokenizerBundle",
    "align_lens_text_features",
    "apply_lens_text_offset",
    "empty_negative_lens_text_features",
    "lens_mxfp4_dequant_config",
    "load_lens_gpt_oss_encoder_dequant_bf16",
    "load_lens_tokenizer",
    "render_lens_chat_prompt",
    "repeat_lens_text_features",
    "tokenize_lens_prompt_texts",
    "validate_lens_selected_layers",
]

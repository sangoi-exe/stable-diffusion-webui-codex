"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Microsoft Lens Diffusers-folder metadata validator for the parked engine skeleton.
Validates JSON/config/index structure in `metadata_only` and `runtime_ready` modes without importing upstream `lens`, constructing Diffusers pipelines, parsing tokenizer JSON, or reading tensor payloads.

Symbols (top-level; keep in sync; no ghosts):
- `LensFolderMetadata` (dataclass): Summarized validated Lens folder metadata.
- `LensSafeTensorsIndexSummary` (dataclass): Header/index-only SafeTensors shard summary.
- `lens_variant_from_ref_identity` (function): Infer a trusted Lens variant only from repo/path identity.
- `validate_lens_folder` (function): Validate a Lens Diffusers-style folder in `metadata_only` or `runtime_ready` mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from apps.backend.runtime.families.lens.config import (
    LENS_AXES_DIMS_ROPE,
    LENS_MXFP4_QUANT_METHOD,
    LENS_PIPELINE_CLASS,
    LENS_SCHEDULER_CLASS,
    LENS_SELECTED_LAYER_INDEX,
    LENS_SEQUENCE_LATENT_CHANNELS,
    LENS_TEXT_ENCODER_CLASS,
    LENS_TEXT_HIDDEN_DIM,
    LENS_TEXT_MODEL_TYPE,
    LENS_TEXT_NUM_HIDDEN_LAYERS,
    LENS_TOKENIZER_BACKEND,
    LENS_TOKENIZER_CLASS,
    LENS_TOKENIZER_CONFIG_CLASS,
    LENS_TRANSFORMER_CLASS,
    LENS_TRANSFORMER_ATTENTION_HEAD_DIM,
    LENS_TRANSFORMER_INNER_DIM,
    LENS_TRANSFORMER_NUM_ATTENTION_HEADS,
    LENS_TRANSFORMER_NUM_LAYERS,
    LENS_TRANSFORMER_OUT_CHANNELS,
    LENS_TRANSFORMER_PATCH_SIZE,
    LENS_VAE_CLASS,
    LENS_VAE_LATENT_CHANNELS,
    LENS_VARIANT_SPECS,
    require_lens_variant,
)
from apps.backend.runtime.families.lens.scheduler import lens_scheduler_config_from_mapping

LensFolderValidationMode = Literal["metadata_only", "runtime_ready"]


@dataclass(frozen=True, slots=True)
class LensSafeTensorsIndexSummary:
    index_path: str
    total_size: int
    total_parameters: int | None
    tensor_count: int
    shards: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LensFolderMetadata:
    root: Path
    variant: str
    validation_mode: LensFolderValidationMode
    trusted_identity_variant: str | None
    text_encoder_index: LensSafeTensorsIndexSummary
    transformer_index: LensSafeTensorsIndexSummary
    tokenizer_json_path: Path
    chat_template_path: Path

    @property
    def total_indexed_weight_size(self) -> int:
        return self.text_encoder_index.total_size + self.transformer_index.total_size


_REQUIRED_METADATA_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/chat_template.jinja",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
)

_MODEL_INDEX_COMPONENTS: Mapping[str, tuple[str, str]] = {
    "scheduler": ("diffusers", LENS_SCHEDULER_CLASS),
    "vae": ("diffusers", LENS_VAE_CLASS),
    "text_encoder": ("transformers", LENS_TEXT_ENCODER_CLASS),
    "tokenizer": ("transformers", LENS_TOKENIZER_CLASS),
    "transformer": ("diffusers", LENS_TRANSFORMER_CLASS),
}


def _require_validation_mode(raw_mode: object) -> LensFolderValidationMode:
    if raw_mode not in {"metadata_only", "runtime_ready"}:
        raise RuntimeError("Lens folder validation mode must be 'metadata_only' or 'runtime_ready'.")
    return raw_mode  # type: ignore[return-value]


def _read_json_mapping(path: Path, *, context: str) -> Mapping[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - path/context is surfaced below
        raise RuntimeError(f"{context}: failed to read JSON at {path}.") from exc
    if not isinstance(data, Mapping):
        raise RuntimeError(f"{context}: expected JSON object at {path}.")
    return data


def _require_file(root: Path, relative_path: str, *, context: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise RuntimeError(f"{context}: missing required file '{relative_path}'.")
    return path


def _require_non_empty_file(root: Path, relative_path: str, *, context: str) -> Path:
    path = _require_file(root, relative_path, context=context)
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{context}: required file '{relative_path}' is empty.")
    return path


def _require_equal(config: Mapping[str, object], key: str, expected: object, *, context: str) -> object:
    actual = config.get(key)
    if actual != expected:
        raise RuntimeError(f"{context}: field {key!r} expected {expected!r}, got {actual!r}.")
    return actual


def _require_int_field(config: Mapping[str, object], key: str, expected: int, *, context: str) -> int:
    actual = config.get(key)
    if isinstance(actual, bool) or actual != expected:
        raise RuntimeError(f"{context}: field {key!r} expected {expected!r}, got {actual!r}.")
    return expected


def _validate_model_index(model_index: Mapping[str, object], *, context: str) -> None:
    _require_equal(model_index, "_class_name", LENS_PIPELINE_CLASS, context=context)
    for component_name, expected in _MODEL_INDEX_COMPONENTS.items():
        actual = model_index.get(component_name)
        if not isinstance(actual, list) or tuple(actual) != expected:
            raise RuntimeError(f"{context}: component {component_name!r} expected {expected!r}, got {actual!r}.")


def _validate_transformer_config(config: Mapping[str, object], *, context: str) -> None:
    _require_equal(config, "_class_name", LENS_TRANSFORMER_CLASS, context=context)
    _require_int_field(config, "patch_size", LENS_TRANSFORMER_PATCH_SIZE, context=context)
    _require_int_field(config, "in_channels", LENS_SEQUENCE_LATENT_CHANNELS, context=context)
    _require_int_field(config, "out_channels", LENS_TRANSFORMER_OUT_CHANNELS, context=context)
    _require_int_field(config, "num_layers", LENS_TRANSFORMER_NUM_LAYERS, context=context)
    _require_int_field(config, "attention_head_dim", LENS_TRANSFORMER_ATTENTION_HEAD_DIM, context=context)
    _require_int_field(config, "num_attention_heads", LENS_TRANSFORMER_NUM_ATTENTION_HEADS, context=context)
    _require_int_field(config, "inner_dim", LENS_TRANSFORMER_INNER_DIM, context=context)
    _require_int_field(config, "enc_hidden_dim", LENS_TEXT_HIDDEN_DIM, context=context)
    _require_equal(config, "axes_dims_rope", list(LENS_AXES_DIMS_ROPE), context=context)
    _require_equal(config, "selected_layer_index", list(LENS_SELECTED_LAYER_INDEX), context=context)
    _require_equal(config, "multi_layer_encoder_feature", True, context=context)
    _require_equal(config, "rms_norm", True, context=context)


def _validate_text_encoder_config(config: Mapping[str, object], *, context: str) -> None:
    _require_equal(config, "architectures", ["GptOssForCausalLM"], context=context)
    _require_equal(config, "model_type", LENS_TEXT_MODEL_TYPE, context=context)
    _require_int_field(config, "hidden_size", LENS_TEXT_HIDDEN_DIM, context=context)
    _require_int_field(config, "num_hidden_layers", LENS_TEXT_NUM_HIDDEN_LAYERS, context=context)
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, Mapping):
        raise RuntimeError(f"{context}: quantization_config must be an object.")
    _require_equal(quantization_config, "quant_method", LENS_MXFP4_QUANT_METHOD, context=context)


def _validate_tokenizer_config(config: Mapping[str, object], *, context: str) -> None:
    _require_equal(config, "backend", LENS_TOKENIZER_BACKEND, context=context)
    _require_equal(config, "tokenizer_class", LENS_TOKENIZER_CONFIG_CLASS, context=context)
    _require_equal(config, "model_input_names", ["input_ids", "attention_mask"], context=context)


def _validate_vae_config(config: Mapping[str, object], *, context: str) -> None:
    _require_equal(config, "_class_name", LENS_VAE_CLASS, context=context)
    _require_int_field(config, "latent_channels", LENS_VAE_LATENT_CHANNELS, context=context)
    _require_equal(config, "patch_size", [2, 2], context=context)
    _require_equal(config, "batch_norm_eps", 0.0001, context=context)
    _require_equal(config, "force_upcast", True, context=context)


def _require_index_int(metadata: Mapping[str, object], key: str, *, context: str, required: bool) -> int | None:
    actual = metadata.get(key)
    if actual is None and not required:
        return None
    if isinstance(actual, bool) or not isinstance(actual, int) or actual <= 0:
        raise RuntimeError(f"{context}: metadata.{key} must be a positive integer.")
    return actual


def _validate_shard_name(raw_shard: object, *, context: str) -> str:
    if not isinstance(raw_shard, str) or not raw_shard.strip():
        raise RuntimeError(f"{context}: shard path must be a non-empty string.")
    shard = raw_shard.strip()
    shard_path = Path(shard)
    if shard_path.is_absolute() or ".." in shard_path.parts:
        raise RuntimeError(f"{context}: shard path must be relative and stay inside the component folder: {shard!r}.")
    if shard_path.suffix != ".safetensors":
        raise RuntimeError(f"{context}: shard path must end with .safetensors: {shard!r}.")
    return shard


def _validate_safetensors_index(
    index_path: Path,
    *,
    context: str,
    validation_mode: LensFolderValidationMode,
) -> LensSafeTensorsIndexSummary:
    data = _read_json_mapping(index_path, context=context)
    metadata = data.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"{context}: metadata must be an object.")
    total_size = _require_index_int(metadata, "total_size", context=context, required=True)
    if total_size is None:
        raise RuntimeError(f"{context}: metadata.total_size is required.")
    total_parameters = _require_index_int(metadata, "total_parameters", context=context, required=False)
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise RuntimeError(f"{context}: weight_map must be a non-empty object.")

    shard_names: set[str] = set()
    for tensor_name, raw_shard in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise RuntimeError(f"{context}: weight_map tensor names must be non-empty strings.")
        shard = _validate_shard_name(raw_shard, context=f"{context}.weight_map[{tensor_name!r}]")
        shard_names.add(shard)

    if validation_mode == "runtime_ready":
        for shard in sorted(shard_names):
            shard_path = index_path.parent / shard
            if not shard_path.is_file():
                raise RuntimeError(f"{context}: runtime_ready validation missing shard '{shard}'.")

    return LensSafeTensorsIndexSummary(
        index_path=str(index_path),
        total_size=total_size,
        total_parameters=total_parameters,
        tensor_count=len(weight_map),
        shards=tuple(sorted(shard_names)),
    )


def lens_variant_from_ref_identity(model_ref: object) -> str | None:
    raw_ref = str(model_ref or "").strip()
    if not raw_ref:
        return None
    normalized_ref = raw_ref.replace("\\", "/").rstrip("/")
    ref_parts = tuple(part for part in normalized_ref.split("/") if part)
    for spec in LENS_VARIANT_SPECS.values():
        repo_parts = tuple(spec.repo_id.split("/"))
        if normalized_ref == spec.repo_id:
            return spec.variant
        if len(ref_parts) >= 2 and ref_parts[-2:] == repo_parts:
            return spec.variant
        if ref_parts and ref_parts[-1] == spec.folder_name:
            return spec.variant
    return None


def _resolve_folder_root(model_ref: object) -> Path:
    if not isinstance(model_ref, str) or not model_ref.strip():
        raise RuntimeError("Lens model_ref must be a non-empty Diffusers folder path.")
    root = Path(model_ref.strip()).expanduser()
    if not root.is_dir():
        raise RuntimeError(f"Lens model_ref must be a local Diffusers folder path: {root}")
    return root


def validate_lens_folder(
    model_ref: object,
    *,
    variant: object,
    validation_mode: LensFolderValidationMode = "metadata_only",
) -> LensFolderMetadata:
    mode = _require_validation_mode(validation_mode)
    variant_value = require_lens_variant(variant, context="Lens folder variant")
    root = _resolve_folder_root(model_ref)
    context = f"Lens folder {root}"

    trusted_identity_variant = lens_variant_from_ref_identity(str(root))
    if trusted_identity_variant is not None and trusted_identity_variant != variant_value:
        raise RuntimeError(
            f"{context}: requested variant {variant_value!r} conflicts with folder identity {trusted_identity_variant!r}."
        )

    for relative_path in _REQUIRED_METADATA_FILES:
        _require_file(root, relative_path, context=context)
    chat_template_path = _require_non_empty_file(root, "tokenizer/chat_template.jinja", context=context)
    tokenizer_json_path = _require_non_empty_file(root, "tokenizer/tokenizer.json", context=context)

    model_index = _read_json_mapping(root / "model_index.json", context=f"{context}.model_index")
    _validate_model_index(model_index, context=f"{context}.model_index")

    scheduler_config = _read_json_mapping(root / "scheduler/scheduler_config.json", context=f"{context}.scheduler")
    lens_scheduler_config_from_mapping(scheduler_config, context=f"{context}.scheduler")

    transformer_config = _read_json_mapping(root / "transformer/config.json", context=f"{context}.transformer")
    _validate_transformer_config(transformer_config, context=f"{context}.transformer")

    text_encoder_config = _read_json_mapping(root / "text_encoder/config.json", context=f"{context}.text_encoder")
    _validate_text_encoder_config(text_encoder_config, context=f"{context}.text_encoder")

    tokenizer_config = _read_json_mapping(root / "tokenizer/tokenizer_config.json", context=f"{context}.tokenizer")
    _validate_tokenizer_config(tokenizer_config, context=f"{context}.tokenizer")

    vae_config = _read_json_mapping(root / "vae/config.json", context=f"{context}.vae")
    _validate_vae_config(vae_config, context=f"{context}.vae")

    text_encoder_index = _validate_safetensors_index(
        root / "text_encoder/model.safetensors.index.json",
        context=f"{context}.text_encoder.index",
        validation_mode=mode,
    )
    transformer_index = _validate_safetensors_index(
        root / "transformer/diffusion_pytorch_model.safetensors.index.json",
        context=f"{context}.transformer.index",
        validation_mode=mode,
    )

    return LensFolderMetadata(
        root=root,
        variant=variant_value,
        validation_mode=mode,
        trusted_identity_variant=trusted_identity_variant,
        text_encoder_index=text_encoder_index,
        transformer_index=transformer_index,
        tokenizer_json_path=tokenizer_json_path,
        chat_template_path=chat_template_path,
    )


__all__ = [
    "LensFolderMetadata",
    "LensFolderValidationMode",
    "LensSafeTensorsIndexSummary",
    "lens_variant_from_ref_identity",
    "validate_lens_folder",
]

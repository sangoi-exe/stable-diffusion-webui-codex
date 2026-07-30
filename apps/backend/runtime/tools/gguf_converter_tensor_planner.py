"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tensor planning helpers for the GGUF converter.
Plans per-tensor quantization/storage settings for the source/native keyspace emitted by the converter and exposes lightweight config
classifiers/metadata normalizers for supported component families.

Symbols (top-level; keep in sync; no ghosts):
- `TensorPlan` (dataclass): Planned tensor conversion entry (name/shape/source dtype/type + storage strategy).
- `tensor_storage_layout` (function): Computes GGUF storage dtype/shape/byte-count for a logical tensor type.
- `retype_tensor_plan` (function): Returns a TensorPlan with a different GGML type and recalculated storage layout.
- `plan_tensors` (function): Plan per-tensor conversion settings for a safetensors source.
- `_source_ggml_type_from_slice` (function): Reads a SafeTensors slice dtype into the matching float GGML type when supported.
- `_rule_matches` (function): Returns whether a compiled dtype rule matches source or destination tensor names.
- `_resolve_rule_ggml_type` (function): Resolves a compiled dtype rule, including source-dtype preservation actions.
- `_select_ggml_type` (function): Selects an effective GGML type for a tensor (requested + per-name override rules).
- `is_zimage_transformer_config` (function): Returns True when a config.json represents a Z-Image transformer export.
- `normalize_zimage_transformer_metadata_config` (function): Adapts Z-Image transformer config fields to metadata helper inputs (variant-neutral; no Turbo defaulting).
- `is_zimage_l2p_denoiser_config` (function): Returns True when a config.json represents the L2P pixel-space denoiser.
- `normalize_zimage_l2p_denoiser_metadata_config` (function): Adapts L2P denoiser config fields to metadata helper inputs.
- `is_zimage_l2p_text_encoder_config` (function): Returns True when a config.json represents exact Qwen3-4B for L2P.
- `normalize_zimage_l2p_text_encoder_metadata_config` (function): Adapts exact Qwen3-4B config fields to metadata helper inputs.
- `is_flux_transformer_config` (function): Returns True when a config.json represents a Flux transformer export.
- `normalize_flux_transformer_metadata_config` (function): Adapts Flux transformer config fields to metadata helper inputs.
- `_is_qwen_image_canonical_repo_id` (function): Returns True when a model-name candidate is the canonical Qwen Image repo id for the variant.
- `_is_path_like_model_name` (function): Returns True when a model-name candidate looks like a local filesystem path.
- `_safe_qwen_image_model_name` (function): Selects a stable Qwen model name without leaking local filesystem paths.
- `is_qwen_image_transformer_config` (function): Returns True when a config.json represents a Qwen Image transformer export.
- `normalize_qwen_image_transformer_metadata_config` (function): Adapts Qwen Image transformer config fields to metadata helper inputs.
- `_require_exact_int` (function): Requires an exact integer config value for strict profile identity checks.
- `_require_exact_float` (function): Requires an exact numeric config value for strict profile identity checks.
- `_qwen_image_text_encoder_identity_config` (function): Validates exact Qwen Image Qwen2.5-VL text-encoder identity metadata.
- `_safe_qwen_image_text_encoder_model_name` (function): Selects a stable Qwen text-encoder model name without leaking local paths.
- `is_qwen_image_text_encoder_config` (function): Returns True when a config.json represents a Qwen Image Qwen2.5-VL text encoder.
- `normalize_qwen_image_text_encoder_metadata_config` (function): Adapts Qwen Image text-encoder config fields to metadata helper inputs.
- `is_wan22_transformer_config` (function): Returns True when a config.json represents a WAN22 transformer export.
- `normalize_wan22_transformer_metadata_config` (function): Adapts WAN22 transformer config fields to metadata helper inputs.
- `is_ltx2_transformer_config` (function): Returns True when a config.json represents an LTX2 transformer export.
- `normalize_ltx2_transformer_metadata_config` (function): Adapts LTX2 transformer config fields to metadata helper inputs.
- `is_gemma3_text_encoder_config` (function): Returns True when a config.json represents a Gemma3 text encoder export.
- `normalize_gemma3_text_encoder_metadata_config` (function): Adapts Gemma3 config fields to metadata helper inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from apps.backend.quantization.gguf import GGMLQuantizationType
from apps.backend.quantization.gguf.quant_shapes import quant_shape_to_byte_shape
from apps.backend.runtime.families.qwen_image.config import QWEN_IMAGE_EDIT_VARIANT
from apps.backend.runtime.families.qwen_image.text_encoder import qwen_image_text_encoder_config_from_mapping
from apps.backend.runtime.families.qwen_image.transformer import qwen_image_transformer_config_from_mapping
from apps.backend.runtime.tools.gguf_converter_quantization import select_tensor_ggml_type
from apps.backend.runtime.tools.gguf_converter_specs import CompiledTensorTypeRule

_ZIMAGE_PAD_TOKENS = {"x_pad_token", "cap_pad_token"}
_QWEN_IMAGE_TEXT_ENCODER_FALLBACK_NAME = "qwen_image_qwen2_5_vl_text_encoder"
_QWEN_IMAGE_TEXT_ENCODER_CANONICAL_REPO_IDS = frozenset(
    {
        "Qwen/Qwen-Image-Edit-2511",
    }
)
_SOURCE_FLOAT_GGML_TYPES: dict[str, GGMLQuantizationType] = {
    "F16": GGMLQuantizationType.F16,
    "BF16": GGMLQuantizationType.BF16,
    "F32": GGMLQuantizationType.F32,
}


@dataclass(frozen=True, slots=True)
class TensorPlan:
    src_name: str
    gguf_name: str
    raw_shape: tuple[int, ...]
    source_ggml_type: GGMLQuantizationType | None
    ggml_type: GGMLQuantizationType
    stored_shape: tuple[int, ...]
    stored_dtype: np.dtype
    stored_nbytes: int
    op: str
    src_names: tuple[str, ...]


def tensor_storage_layout(
    raw_shape: tuple[int, ...],
    ggml_type: GGMLQuantizationType,
) -> tuple[tuple[int, ...], np.dtype, int]:
    if ggml_type == GGMLQuantizationType.F16:
        stored_dtype = np.dtype(np.float16)
        stored_shape = raw_shape
        stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 2)
    elif ggml_type == GGMLQuantizationType.BF16:
        stored_dtype = np.dtype(np.uint16)
        stored_shape = raw_shape
        stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 2)
    elif ggml_type == GGMLQuantizationType.F32:
        stored_dtype = np.dtype(np.float32)
        stored_shape = raw_shape
        stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 4)
    else:
        stored_dtype = np.dtype(np.uint8)
        stored_shape = quant_shape_to_byte_shape(raw_shape, ggml_type)
        stored_nbytes = int(np.prod(stored_shape, dtype=np.int64))

    return stored_shape, stored_dtype, stored_nbytes


def retype_tensor_plan(plan: TensorPlan, ggml_type: GGMLQuantizationType) -> TensorPlan:
    stored_shape, stored_dtype, stored_nbytes = tensor_storage_layout(plan.raw_shape, ggml_type)
    return TensorPlan(
        src_name=plan.src_name,
        gguf_name=plan.gguf_name,
        raw_shape=plan.raw_shape,
        source_ggml_type=plan.source_ggml_type,
        ggml_type=ggml_type,
        stored_shape=stored_shape,
        stored_dtype=stored_dtype,
        stored_nbytes=stored_nbytes,
        op=plan.op,
        src_names=plan.src_names,
    )


def plan_tensors(
    tensor_names: list[str],
    safetensors_handle: Any,
    key_mapping: dict[str, str],
    requested: GGMLQuantizationType,
    overrides: list[CompiledTensorTypeRule],
) -> list[TensorPlan]:
    plans: list[TensorPlan] = []

    for src_name in tensor_names:
        sl = safetensors_handle.get_slice(src_name)
        raw_shape = tuple(int(x) for x in sl.get_shape())
        source_ggml_type, source_dtype_name = _source_ggml_type_from_slice(src_name, sl)
        gguf_name = key_mapping.get(src_name, src_name)

        desired = requested
        for rule in overrides:
            if _rule_matches(rule, src_name=src_name, gguf_name=gguf_name):
                desired = _resolve_rule_ggml_type(
                    rule,
                    src_name=src_name,
                    source_ggml_type=source_ggml_type,
                    source_dtype_name=source_dtype_name,
                )
        ggml_type = _select_ggml_type(raw_shape=raw_shape, gguf_name=gguf_name, desired=desired)

        stored_shape, stored_dtype, stored_nbytes = tensor_storage_layout(raw_shape, ggml_type)

        plans.append(
            TensorPlan(
                src_name=src_name,
                gguf_name=gguf_name,
                raw_shape=raw_shape,
                source_ggml_type=source_ggml_type,
                ggml_type=ggml_type,
                stored_shape=stored_shape,
                stored_dtype=stored_dtype,
                stored_nbytes=stored_nbytes,
                op="copy",
                src_names=(src_name,),
            )
        )

    return plans


def _source_ggml_type_from_slice(src_name: str, safetensors_slice: Any) -> tuple[GGMLQuantizationType | None, str]:
    get_dtype = getattr(safetensors_slice, "get_dtype", None)
    if not callable(get_dtype):
        return None, "unavailable"
    try:
        dtype_name = str(get_dtype()).strip().upper()
    except Exception as exc:
        raise RuntimeError(f"Failed to read source dtype metadata for tensor {src_name!r}: {exc}") from exc
    return _SOURCE_FLOAT_GGML_TYPES.get(dtype_name), dtype_name or "empty"


def _rule_matches(rule: CompiledTensorTypeRule, *, src_name: str, gguf_name: str) -> bool:
    return (rule.apply_to.matches_src() and rule.pattern.search(src_name) is not None) or (
        rule.apply_to.matches_dst() and rule.pattern.search(gguf_name) is not None
    )


def _resolve_rule_ggml_type(
    rule: CompiledTensorTypeRule,
    *,
    src_name: str,
    source_ggml_type: GGMLQuantizationType | None,
    source_dtype_name: str,
) -> GGMLQuantizationType:
    if rule.preserve_source_dtype:
        if source_ggml_type is None:
            raise RuntimeError(
                f"Source-dtype preservation rule matched tensor {src_name!r}, but its SafeTensors dtype "
                f"{source_dtype_name!r} is not one of F16, BF16, or F32."
            )
        return source_ggml_type
    if rule.ggml_type is None:
        raise RuntimeError(f"Compiled dtype rule for tensor {src_name!r} has no target GGML type.")
    return rule.ggml_type


def _select_ggml_type(
    *,
    raw_shape: tuple[int, ...],
    gguf_name: str,
    desired: GGMLQuantizationType,
) -> GGMLQuantizationType:
    ggml_type = select_tensor_ggml_type(raw_shape, desired)

    if gguf_name in _ZIMAGE_PAD_TOKENS:
        if ggml_type == GGMLQuantizationType.BF16:
            raise RuntimeError(
                f"BF16 override is not supported for Z-Image pad token {gguf_name!r}; use F16/F32 or auto."
            )
        if ggml_type not in {GGMLQuantizationType.F16, GGMLQuantizationType.F32}:
            return GGMLQuantizationType.F16

    return ggml_type


def is_zimage_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") == "ZImageTransformer2DModel"


def _config_int(config: Mapping[str, Any], key: str, default: int) -> int:
    raw = config.get(key)
    if isinstance(raw, bool):
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _first_config_int(config: Mapping[str, Any], keys: Sequence[str], default: int) -> int:
    for key in keys:
        if key in config:
            return _config_int(config, key, default)
    return int(default)


def _first_list_int(config: Mapping[str, Any], key: str, default: int) -> int:
    raw = config.get(key)
    if isinstance(raw, (list, tuple)) and raw:
        try:
            return int(raw[0])
        except Exception:
            return int(default)
    return _config_int(config, key, default)


def is_zimage_l2p_denoiser_config(config: Mapping[str, Any]) -> bool:
    class_name = str(config.get("_class_name") or "").strip()
    if class_name not in {"ZImageL2PDiT", "ZImageDiT"}:
        return False
    return (
        _first_config_int(config, ("in_channels", "in_chans"), 3) == 3
        and _first_list_int(config, "all_patch_size", _config_int(config, "patch_size", 16)) == 16
        and _first_list_int(config, "all_f_patch_size", _config_int(config, "frame_patch_size", 1)) == 1
        and _first_config_int(config, ("dim", "hidden_dim", "hidden_size"), 3840) == 3840
        and _first_config_int(config, ("n_layers", "num_layers", "num_hidden_layers"), 30) == 30
        and _first_config_int(config, ("n_refiner_layers", "num_refiner_layers"), 2) == 2
        and _first_config_int(config, ("n_heads", "num_heads", "num_attention_heads"), 30) == 30
        and _first_config_int(config, ("cap_feat_dim", "context_dim"), 2560) == 2560
    )


def is_zimage_l2p_text_encoder_config(config: Mapping[str, Any]) -> bool:
    model_type = str(config.get("model_type") or "").strip().lower()
    architectures_raw = config.get("architectures")
    if isinstance(architectures_raw, str):
        architectures = (architectures_raw,)
    elif isinstance(architectures_raw, list):
        architectures = tuple(str(value) for value in architectures_raw)
    else:
        architectures = ()
    if model_type != "qwen3" and "Qwen3ForCausalLM" not in architectures:
        return False
    return (
        _config_int(config, "vocab_size", 151936) == 151936
        and _config_int(config, "hidden_size", 2560) == 2560
        and _config_int(config, "intermediate_size", 9728) == 9728
        and _config_int(config, "num_hidden_layers", 36) == 36
        and _config_int(config, "num_attention_heads", 32) == 32
        and _config_int(config, "num_key_value_heads", 8) == 8
    )


def is_flux_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") == "FluxTransformer2DModel"


def _is_qwen_image_canonical_repo_id(value: str, *, fallback: str) -> bool:
    return str(value or "").strip() == fallback


def _is_path_like_model_name(value: str, *, fallback: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False
    if candidate.startswith(("/", "\\", ".", "~")):
        return True
    if re.match(r"^[A-Za-z]:", candidate):
        return True
    if "/" in candidate or "\\" in candidate:
        return not _is_qwen_image_canonical_repo_id(candidate, fallback=fallback)
    return False


def _safe_qwen_image_model_name(config: Mapping[str, Any], *, fallback: str) -> str:
    for key in ("_name_or_path", "name"):
        value = str(config.get(key) or "").strip()
        if not value or _is_path_like_model_name(value, fallback=fallback):
            continue
        return value
    return fallback


def is_qwen_image_transformer_config(config: Mapping[str, Any]) -> bool:
    if str(config.get("_class_name") or "") != "QwenImageTransformer2DModel":
        return False
    try:
        qwen_image_transformer_config_from_mapping(
            config,
            variant=QWEN_IMAGE_EDIT_VARIANT,
            context="Qwen Image GGUF converter profile detection",
        )
    except RuntimeError:
        return False
    return True


def _require_exact_int(config: Mapping[str, Any], key: str, expected: int, *, context: str) -> int:
    raw = config.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RuntimeError(f"{context}: {key} must be integer {expected}.")
    value = int(raw)
    if value != expected:
        raise RuntimeError(f"{context}: {key} must be {expected}; got {value}.")
    return value


def _require_exact_float(config: Mapping[str, Any], key: str, expected: float, *, context: str) -> float:
    raw = config.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RuntimeError(f"{context}: {key} must be numeric {expected}.")
    value = float(raw)
    if abs(value - expected) > 1e-12:
        raise RuntimeError(f"{context}: {key} must be {expected}; got {value}.")
    return value


def _qwen_image_text_encoder_identity_config(config: Mapping[str, Any], *, context: str) -> dict[str, int | float]:
    qwen_image_text_encoder_config_from_mapping(config, context=context)
    return {
        "num_hidden_layers": _require_exact_int(config, "num_hidden_layers", 28, context=context),
        "num_attention_heads": _require_exact_int(config, "num_attention_heads", 28, context=context),
        "num_key_value_heads": _require_exact_int(config, "num_key_value_heads", 4, context=context),
        "max_position_embeddings": _require_exact_int(config, "max_position_embeddings", 128000, context=context),
        "rope_theta": _require_exact_float(config, "rope_theta", 1000000.0, context=context),
        "rms_norm_eps": _require_exact_float(config, "rms_norm_eps", 1e-6, context=context),
    }


def _safe_qwen_image_text_encoder_model_name(config: Mapping[str, Any]) -> str:
    candidates: list[Mapping[str, Any]] = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping):
        candidates.append(text_config)

    for candidate_config in candidates:
        for key in ("_name_or_path", "name"):
            value = str(candidate_config.get(key) or "").strip()
            if not value:
                continue
            if value in _QWEN_IMAGE_TEXT_ENCODER_CANONICAL_REPO_IDS:
                return value
            if _is_path_like_model_name(value, fallback=_QWEN_IMAGE_TEXT_ENCODER_FALLBACK_NAME):
                continue
            return value
    return _QWEN_IMAGE_TEXT_ENCODER_FALLBACK_NAME


def is_qwen_image_text_encoder_config(config: Mapping[str, Any]) -> bool:
    try:
        _qwen_image_text_encoder_identity_config(
            config,
            context="Qwen Image text-encoder GGUF converter profile detection",
        )
    except RuntimeError:
        return False
    return True


def is_wan22_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") in {"WanTransformer3DModel", "WanModel"}


def is_ltx2_transformer_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("_class_name") or "") == "LTX2VideoTransformer3DModel"


def is_gemma3_text_encoder_config(config: Mapping[str, Any]) -> bool:
    return str(config.get("model_type") or "").strip() == "gemma3"


def normalize_flux_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Flux transformer config keys into the metadata helper's expected fields."""

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    num_heads = _as_int(config.get("num_attention_heads"), 24)
    head_dim = _as_int(config.get("attention_head_dim"), 128)
    hidden = max(1, num_heads) * max(1, head_dim)

    double_layers = _as_int(config.get("num_layers"), 19)
    single_layers = _as_int(config.get("num_single_layers"), 38)

    return {
        "model_type": "flux",
        "num_hidden_layers": max(1, double_layers + single_layers),
        "hidden_size": hidden,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "_name_or_path": str(
            config.get("_name_or_path") or config.get("name") or "black-forest-labs/FLUX.1-Kontext-dev"
        ),
    }


def normalize_qwen_image_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Qwen Image transformer config keys into the metadata helper's expected fields."""

    qwen_config = qwen_image_transformer_config_from_mapping(
        config,
        variant=QWEN_IMAGE_EDIT_VARIANT,
        context="Qwen Image GGUF converter metadata",
    )
    hidden = int(qwen_config.num_attention_heads * qwen_config.attention_head_dim)
    fallback_name = "Qwen/Qwen-Image-Edit-2511"
    name = _safe_qwen_image_model_name(config, fallback=fallback_name)

    return {
        "model_type": "qwen_image",
        "num_hidden_layers": qwen_config.num_layers,
        "hidden_size": hidden,
        "num_attention_heads": qwen_config.num_attention_heads,
        "num_key_value_heads": qwen_config.num_attention_heads,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "_name_or_path": name,
        "codex.qwen_image.variant": qwen_config.variant,
        "codex.qwen_image.zero_cond_t": bool(qwen_config.zero_cond_t),
        "codex.qwen_image.joint_attention_dim": qwen_config.joint_attention_dim,
        "codex.qwen_image.in_channels": qwen_config.in_channels,
        "codex.qwen_image.out_channels": qwen_config.out_channels,
        "codex.qwen_image.patch_size": qwen_config.patch_size,
        "codex.qwen_image.axes_dims_rope": list(qwen_config.axes_dims_rope),
    }


def normalize_qwen_image_text_encoder_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Qwen Image Qwen2.5-VL text-encoder config keys into metadata helper inputs."""

    identity = _qwen_image_text_encoder_identity_config(
        config,
        context="Qwen Image text-encoder GGUF converter metadata",
    )

    return {
        "model_type": "qwen2_5_vl",
        "num_hidden_layers": identity["num_hidden_layers"],
        "hidden_size": 3584,
        "num_attention_heads": identity["num_attention_heads"],
        "num_key_value_heads": identity["num_key_value_heads"],
        "max_position_embeddings": identity["max_position_embeddings"],
        "rope_theta": identity["rope_theta"],
        "rms_norm_eps": identity["rms_norm_eps"],
        "_name_or_path": _safe_qwen_image_text_encoder_model_name(config),
    }


def normalize_wan22_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt WAN22 transformer config keys into the metadata helper's expected fields."""

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    class_name = str(config.get("_class_name") or "")
    if class_name == "WanModel":
        hidden = _as_int(config.get("dim"), 4096)
        num_layers = _as_int(config.get("num_layers"), 32)
        num_heads = _as_int(config.get("num_heads"), 32)
        max_pos = _as_int(config.get("text_len"), 512)
        eps = _as_float(config.get("eps"), 1e-6)
        model_type = str(config.get("model_type") or "").strip().lower()
        name = "Wan-AI/Wan2.2"
        if model_type == "i2v":
            name = "Wan-AI/Wan2.2-I2V-A14B"
        elif model_type == "t2v":
            name = "Wan-AI/Wan2.2-T2V-A14B"
        return {
            "model_type": "wan22",
            "num_hidden_layers": max(1, num_layers),
            "hidden_size": max(1, hidden),
            "num_attention_heads": max(1, num_heads),
            "num_key_value_heads": max(1, num_heads),
            "max_position_embeddings": max(1, max_pos),
            "rope_theta": 10000.0,
            "rms_norm_eps": eps,
            "_name_or_path": str(config.get("_name_or_path") or config.get("name") or name),
        }

    num_heads = _as_int(config.get("num_attention_heads"), 40)
    head_dim = _as_int(config.get("attention_head_dim"), 128)
    hidden = max(1, num_heads) * max(1, head_dim)

    num_layers = _as_int(config.get("num_layers"), 40)
    max_pos = _as_int(config.get("rope_max_seq_len"), 1024)
    eps = _as_float(config.get("eps"), 1e-6)

    name = "Wan-AI/Wan2.2"
    in_channels = _as_int(config.get("in_channels"), 16)
    if in_channels == 36:
        name = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    elif in_channels == 16:
        name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    return {
        "model_type": "wan22",
        "num_hidden_layers": max(1, num_layers),
        "hidden_size": hidden,
        "num_attention_heads": max(1, num_heads),
        "num_key_value_heads": max(1, num_heads),
        "max_position_embeddings": max(1, max_pos),
        "rope_theta": 10000.0,
        "rms_norm_eps": eps,
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or name),
    }


def normalize_ltx2_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt LTX2 transformer config keys into the metadata helper's expected fields."""

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    num_heads = _as_int(config.get("num_attention_heads"), 32)
    head_dim = _as_int(config.get("attention_head_dim"), 128)
    hidden = max(1, num_heads) * max(1, head_dim)
    num_layers = _as_int(config.get("num_layers"), 48)
    rope_theta = _as_float(config.get("rope_theta"), 10000.0)
    norm_eps = _as_float(config.get("norm_eps"), 1e-6)

    return {
        "model_type": "ltx2",
        "num_hidden_layers": max(1, num_layers),
        "hidden_size": hidden,
        "num_attention_heads": max(1, num_heads),
        "num_key_value_heads": max(1, num_heads),
        "max_position_embeddings": 4096,
        "rope_theta": rope_theta,
        "rms_norm_eps": norm_eps,
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or "Lightricks/LTX-2"),
    }


def normalize_gemma3_text_encoder_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Gemma3 text encoder config keys into the metadata helper's expected fields.

    Gemma3 `config.json` used by Diffusers is a multimodal envelope that contains the LLM config under `text_config`.
    """

    text_cfg = config.get("text_config")
    if not isinstance(text_cfg, dict):
        raise ValueError("Gemma3 metadata normalize: expected `text_config` dict in config.json")

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    num_layers = _as_int(text_cfg.get("num_hidden_layers"), 48)
    hidden = _as_int(text_cfg.get("hidden_size"), 3840)
    num_heads = _as_int(text_cfg.get("num_attention_heads"), 16)
    num_kv = _as_int(text_cfg.get("num_key_value_heads"), 8)
    max_pos = _as_int(text_cfg.get("max_position_embeddings"), 131072)
    rope_theta = _as_float(text_cfg.get("rope_theta"), 1000000.0)
    eps = _as_float(text_cfg.get("rms_norm_eps"), 1e-6)

    return {
        "model_type": "gemma3",
        "num_hidden_layers": max(1, num_layers),
        "hidden_size": max(1, hidden),
        "num_attention_heads": max(1, num_heads),
        "num_key_value_heads": max(1, num_kv),
        "max_position_embeddings": max(1, max_pos),
        "rope_theta": rope_theta,
        "rms_norm_eps": eps,
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or "google/gemma-3"),
    }


def normalize_zimage_transformer_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Z-Image transformer config keys into the metadata helper's expected fields.

    The GGUF metadata helper is LLM-shaped (hidden_size, num_hidden_layers, etc.).
    Z-Image transformer configs use Diffusers keys (`dim`, `n_layers`, ...).
    """

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    patch_size = config.get("all_patch_size")
    if isinstance(patch_size, list) and patch_size:
        patch_size = patch_size[0]
    patch = _as_int(patch_size, 2)

    dim = _as_int(config.get("dim"), 3840)
    n_layers = _as_int(config.get("n_layers"), 30)
    n_heads = _as_int(config.get("n_heads"), 30)
    n_kv = _as_int(config.get("n_kv_heads"), n_heads)

    axes_lens = config.get("axes_lens")
    if isinstance(axes_lens, list) and len(axes_lens) == 3:
        max_pos = max(_as_int(v, 0) for v in axes_lens)
    else:
        max_pos = 4096

    return {
        "model_type": "zimage",
        "num_hidden_layers": n_layers,
        "hidden_size": dim,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "max_position_embeddings": max_pos * max(1, patch),
        "rope_theta": _as_float(config.get("rope_theta"), 256.0),
        "rms_norm_eps": _as_float(config.get("norm_eps"), 1e-5),
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or "zimage"),
    }


def normalize_zimage_l2p_denoiser_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt L2P denoiser config keys into metadata helper inputs."""

    if not is_zimage_l2p_denoiser_config(config):
        raise ValueError("Z-Image L2P denoiser metadata normalize: config does not match exact L2P 1K identity")

    dim = _first_config_int(config, ("dim", "hidden_dim", "hidden_size"), 3840)
    n_layers = _first_config_int(config, ("n_layers", "num_layers", "num_hidden_layers"), 30)
    n_heads = _first_config_int(config, ("n_heads", "num_heads", "num_attention_heads"), 30)
    patch = _first_list_int(config, "all_patch_size", _config_int(config, "patch_size", 16))
    f_patch = _first_list_int(config, "all_f_patch_size", _config_int(config, "frame_patch_size", 1))
    axes_dims = config.get("axes_dims")
    axes_lens = config.get("axes_lens")
    if not isinstance(axes_dims, list) or len(axes_dims) != 3:
        axes_dims = [32, 48, 48]
    if not isinstance(axes_lens, list) or len(axes_lens) != 3:
        axes_lens = [1024, 512, 512]

    return {
        "model_type": "zimage_l2p",
        "num_hidden_layers": n_layers,
        "hidden_size": dim,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_heads,
        "max_position_embeddings": max(int(value) for value in axes_lens) * max(1, patch),
        "rope_theta": float(config.get("rope_theta") or 256.0),
        "rms_norm_eps": float(config.get("norm_eps") or config.get("rms_norm_eps") or 1e-5),
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or "zhen-nan/L2P"),
        "codex.zimage_l2p.profile_id": "zimage_l2p_denoiser",
        "codex.zimage_l2p.component": "denoiser",
        "codex.zimage_l2p.family": "zimage_l2p",
        "codex.zimage_l2p.pixel_space": True,
        "codex.zimage_l2p.patch_size": int(patch),
        "codex.zimage_l2p.frame_patch_size": int(f_patch),
        "codex.zimage_l2p.in_channels": int(_first_config_int(config, ("in_channels", "in_chans"), 3)),
        "codex.zimage_l2p.context_dim": int(_first_config_int(config, ("cap_feat_dim", "context_dim"), 2560)),
        "codex.zimage_l2p.num_refiner_layers": int(_first_config_int(config, ("n_refiner_layers", "num_refiner_layers"), 2)),
        "codex.zimage_l2p.axes_dims": [int(value) for value in axes_dims],
        "codex.zimage_l2p.axes_lens": [int(value) for value in axes_lens],
        "codex.zimage_l2p.requires_vae": False,
        "codex.zimage_l2p.local_decoder": True,
    }


def normalize_zimage_l2p_text_encoder_metadata_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt exact Qwen3-4B config keys into metadata helper inputs for L2P."""

    if not is_zimage_l2p_text_encoder_config(config):
        raise ValueError("Z-Image L2P TEnc metadata normalize: config does not match exact Qwen3-4B identity")

    return {
        "model_type": "qwen3",
        "num_hidden_layers": 36,
        "hidden_size": 2560,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": _config_int(config, "max_position_embeddings", 40960),
        "rope_theta": float(config.get("rope_theta") or 1000000.0),
        "rms_norm_eps": float(config.get("rms_norm_eps") or 1e-6),
        "_name_or_path": str(config.get("_name_or_path") or config.get("name") or "Qwen/Qwen3-4B"),
        "codex.zimage_l2p.profile_id": "zimage_l2p_tenc",
        "codex.zimage_l2p.component": "tenc",
        "codex.zimage_l2p.family": "zimage_l2p",
        "codex.zimage_l2p.pixel_space": True,
        "codex.zimage_l2p.requires_vae": False,
        "codex.zimage_l2p.tenc_slot": "qwen3_4b",
        "codex.zimage_l2p.qwen_hidden_size": 2560,
        "codex.zimage_l2p.qwen_layers": 36,
        "codex.zimage_l2p.qwen_heads": 32,
        "codex.zimage_l2p.qwen_kv_heads": 8,
        "codex.zimage_l2p.qwen_vocab": int(config.get("vocab_size") or 151936),
    }


__all__ = [
    "TensorPlan",
    "is_flux_transformer_config",
    "is_gemma3_text_encoder_config",
    "is_ltx2_transformer_config",
    "is_qwen_image_text_encoder_config",
    "is_qwen_image_transformer_config",
    "is_wan22_transformer_config",
    "is_zimage_l2p_denoiser_config",
    "is_zimage_l2p_text_encoder_config",
    "is_zimage_transformer_config",
    "normalize_flux_transformer_metadata_config",
    "normalize_gemma3_text_encoder_metadata_config",
    "normalize_ltx2_transformer_metadata_config",
    "normalize_qwen_image_text_encoder_metadata_config",
    "normalize_qwen_image_transformer_metadata_config",
    "normalize_wan22_transformer_metadata_config",
    "normalize_zimage_l2p_denoiser_metadata_config",
    "normalize_zimage_l2p_text_encoder_metadata_config",
    "normalize_zimage_transformer_metadata_config",
    "plan_tensors",
]

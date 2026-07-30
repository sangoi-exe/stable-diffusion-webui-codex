"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native Qwen Image Edit-2511 component assembly.
Builds and strictly binds the exact transformer GGUF, Qwen2.5-VL GGUF, Codex3D VAE SafeTensors, offline processor,
and scheduler metadata while preserving lazy checkpoint keyspaces, canonical patcher ownership, and deterministic
posterior-mode reference encoding.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageComponentAssembly` (dataclass): Strictly loaded native components plus patcher/runtime ownership.
- `load_qwen_image_components` (function): Assemble the exact Edit-2511 transformer, text encoder, VAE, and scheduler.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from apps.backend.patchers.base import ModelPatcher
from apps.backend.patchers.denoiser import DenoiserPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.checkpoint.io import load_gguf_state_dict, load_torch_file
from apps.backend.runtime.common.vae_codex3d import (
    AutoencoderCodex3D,
    resolve_codex3d_vae_keyspace,
    sanitize_codex3d_vae_config,
)
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.state_dict.keymap_qwen_image_transformer import (
    resolve_qwen_image_edit_transformer_keyspace,
)
from apps.backend.runtime.state_dict.keymap_qwen_text_encoder import (
    resolve_qwen2_5_vl_multimodal_keyspace,
)

from .config import (
    QWEN_IMAGE_EDIT_REPO_ID,
    QWEN_IMAGE_EDIT_VARIANT,
    QWEN_IMAGE_VARIANT_KEY,
    require_qwen_image_variant,
)
from .scheduler import QwenImageSchedulerConfig, qwen_image_scheduler_config_from_mapping
from .text_encoder import (
    QwenImageTextEncoderConfig,
    QwenImageTextEncoderRuntime,
    qwen_image_text_encoder_config_from_mapping,
)
from .transformer import (
    QwenImageTransformer2DModel,
    QwenImageTransformerConfig,
    qwen_image_transformer_config_from_mapping,
)
from .vae import QwenImageVaeConfig, qwen_image_vae_config_from_mapping

if TYPE_CHECKING:
    from apps.backend.runtime.models.loader import DiffusionModelBundle


_METADATA_ROOT = (
    Path(__file__).resolve().parents[3]
    / "huggingface"
    / "Qwen"
    / "Qwen-Image-Edit-2511"
)
_TRANSFORMER_CONFIG_PATH = _METADATA_ROOT / "transformer" / "config.json"
_TEXT_ENCODER_CONFIG_PATH = _METADATA_ROOT / "text_encoder" / "config.json"
_PROCESSOR_PATH = _METADATA_ROOT / "processor"
_VAE_CONFIG_PATH = _METADATA_ROOT / "vae" / "config.json"
_SCHEDULER_CONFIG_PATH = _METADATA_ROOT / "scheduler" / "scheduler_config.json"
_TRANSFORMER_EXECUTABLE_KEYS = 1933
_TEXT_ENCODER_SOURCE_KEYS = 729
_TEXT_ENCODER_EXECUTABLE_KEYS = 728
_VAE_EXECUTABLE_KEYS = 194
_LM_HEAD_LOGICAL_SHAPE = (152064, 3584)
_LM_HEAD_QTYPE = "Q8_0"
_LM_HEAD_PACKED_BYTES = 579_059_712


def _posterior_mode(posterior: object) -> torch.Tensor:
    mode = getattr(posterior, "mode", None)
    if not callable(mode):
        raise RuntimeError(
            "Qwen Image VAE posterior must expose callable mode(); "
            f"got {type(posterior).__name__}."
        )
    value = mode()
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            "Qwen Image VAE posterior mode() must return a torch.Tensor; "
            f"got {type(value).__name__}."
        )
    return value


@dataclass(frozen=True, slots=True)
class QwenImageComponentAssembly:
    variant: str
    metadata_root: Path
    transformer_config: QwenImageTransformerConfig
    transformer: QwenImageTransformer2DModel
    denoiser: DenoiserPatcher
    text_encoder_config: QwenImageTextEncoderConfig
    text_encoder: QwenImageTextEncoderRuntime
    text_encoder_patcher: ModelPatcher
    vae_config: QwenImageVaeConfig
    vae_model: AutoencoderCodex3D
    vae: VAE
    scheduler_config: QwenImageSchedulerConfig


def _read_json_object(path: Path, *, context: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{context}: required vendored metadata file not found: {path}") from exc
    except Exception as exc:  # noqa: BLE001 - strict JSON boundary
        raise RuntimeError(f"{context}: invalid JSON at {path}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise RuntimeError(f"{context}: expected a JSON object at {path}.")
    return parsed


def _resolved_path(raw_path: object, *, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError(f"Qwen Image component assembly requires non-empty {field}.")
    path = Path(raw_path.strip()).expanduser()
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Qwen Image component assembly {field} not found: {path}") from exc


def _require_same_path(left: object, right: object, *, field: str) -> Path:
    left_path = _resolved_path(left, field=f"engine option {field}")
    right_path = _resolved_path(right, field=f"bundle metadata {field}")
    if left_path != right_path:
        raise RuntimeError(
            f"Qwen Image {field} mismatch between engine options and loader bundle: "
            f"option={left_path} bundle={right_path}."
        )
    return left_path


def _require_clean_module_load(
    model: nn.Module,
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    expected_keys: int,
    context: str,
) -> None:
    if len(state_dict) != int(expected_keys):
        raise RuntimeError(
            f"{context}: executable key count mismatch before load. "
            f"got={len(state_dict)} expected={int(expected_keys)}."
        )
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(result, "missing_keys", ()))
    unexpected = list(getattr(result, "unexpected_keys", ()))
    if missing or unexpected:
        raise RuntimeError(
            f"{context}: strict executable binding failed. "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}."
        )
    runtime_keys = set(model.state_dict().keys())
    source_keys = set(state_dict.keys())
    if runtime_keys != source_keys:
        raise RuntimeError(
            f"{context}: post-load runtime/source keysets differ. "
            f"missing_runtime={sorted(source_keys - runtime_keys)[:10]} "
            f"extra_runtime={sorted(runtime_keys - source_keys)[:10]}."
        )


def _validate_tenc_lm_head(state_dict: MutableMapping[str, torch.Tensor]) -> None:
    if len(state_dict) != _TEXT_ENCODER_SOURCE_KEYS:
        raise RuntimeError(
            "Qwen Image text encoder source key count mismatch before lm_head validation. "
            f"got={len(state_dict)} expected={_TEXT_ENCODER_SOURCE_KEYS}."
        )
    try:
        tensor = state_dict["lm_head.weight"]
    except KeyError as exc:
        raise RuntimeError("Qwen Image text encoder is missing required stored lm_head.weight.") from exc
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            "Qwen Image stored lm_head.weight must be a tensor descriptor; "
            f"got {type(tensor).__name__}."
        )
    real_shape = tuple(int(dim) for dim in getattr(tensor, "real_shape", tensor.shape))
    if real_shape != _LM_HEAD_LOGICAL_SHAPE:
        raise RuntimeError(
            "Qwen Image stored lm_head.weight logical shape mismatch. "
            f"got={real_shape} expected={_LM_HEAD_LOGICAL_SHAPE}."
        )
    qtype = getattr(getattr(tensor, "qtype", None), "name", None)
    if qtype != _LM_HEAD_QTYPE:
        raise RuntimeError(
            "Qwen Image stored lm_head.weight quantization mismatch. "
            f"got={qtype!r} expected={_LM_HEAD_QTYPE!r}."
        )
    packed_bytes = int(tensor.data.numel()) * int(tensor.data.element_size())
    if packed_bytes != _LM_HEAD_PACKED_BYTES:
        raise RuntimeError(
            "Qwen Image stored lm_head.weight packed payload mismatch. "
            f"got={packed_bytes} expected={_LM_HEAD_PACKED_BYTES}."
        )


def _load_transformer(
    *,
    state_dict: MutableMapping[str, torch.Tensor],
    config: QwenImageTransformerConfig,
    compute_dtype: torch.dtype,
) -> tuple[QwenImageTransformer2DModel, DenoiserPatcher]:
    mount_device = memory_management.manager.mount_device()
    if mount_device.type != "cpu":
        raise RuntimeError(
            "Qwen Image Edit-2511 transformer requires CPU mount/storage for operation-granular GGUF execution; "
            f"got mount_device={mount_device}."
        )
    resolved = resolve_qwen_image_edit_transformer_keyspace(state_dict)
    with using_codex_operations(
        device=mount_device,
        dtype=compute_dtype,
        manual_cast_enabled=True,
        weight_format="gguf",
    ):
        transformer = QwenImageTransformer2DModel(config)
    _require_clean_module_load(
        transformer,
        resolved.view,
        expected_keys=_TRANSFORMER_EXECUTABLE_KEYS,
        context="Qwen Image transformer",
    )
    transformer.eval()
    transformer.requires_grad_(False)
    transformer.storage_dtype = "gguf"
    transformer.computation_dtype = compute_dtype
    transformer.load_device = mount_device
    transformer.offload_device = mount_device
    transformer.initial_device = mount_device
    denoiser = DenoiserPatcher(
        transformer,
        load_device=mount_device,
        offload_device=mount_device,
        current_device=mount_device,
    )
    return transformer, denoiser


def _load_text_encoder(
    *,
    path: Path,
    config_mapping: Mapping[str, object],
    compute_dtype: torch.dtype,
) -> tuple[QwenImageTextEncoderRuntime, ModelPatcher]:
    from transformers import Qwen2VLProcessor, Qwen2_5_VLConfig, Qwen2_5_VLModel
    from transformers.modeling_utils import no_init_weights

    cpu_device = memory_management.manager.cpu_device
    raw_state_dict = load_gguf_state_dict(
        str(path),
        dequantize=False,
        computation_dtype=compute_dtype,
        device=cpu_device,
    )
    if not isinstance(raw_state_dict, MutableMapping):
        raise RuntimeError(
            "Qwen Image text encoder GGUF loader must return a mutable mapping; "
            f"got {type(raw_state_dict).__name__}."
        )
    metadata = getattr(raw_state_dict, "gguf_metadata", None)
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Qwen Image text encoder GGUF is missing metadata.")
    expected_metadata: Mapping[str, object] = {
        "codex.quant_policy": "qwen_image_tenc_mq_v2",
        "codex.quant_policy_preset": "MQ",
        "codex.quant_base_type": "Q8_0",
        "codex.quant_recipe": "Q8_0",
        "general.file_type": 7,
        "model.architecture": "qwen2_5_vl",
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Qwen Image text encoder GGUF metadata mismatch: {mismatches}.")
    _validate_tenc_lm_head(raw_state_dict)
    resolved = resolve_qwen2_5_vl_multimodal_keyspace(raw_state_dict)

    hf_config = Qwen2_5_VLConfig.from_dict(dict(config_mapping))
    with no_init_weights(), using_codex_operations(
        device=cpu_device,
        dtype=compute_dtype,
        manual_cast_enabled=True,
        weight_format="gguf",
    ):
        text_model = Qwen2_5_VLModel(hf_config)
    _require_clean_module_load(
        text_model,
        resolved.view,
        expected_keys=_TEXT_ENCODER_EXECUTABLE_KEYS,
        context="Qwen Image Qwen2.5-VL base model",
    )
    text_model.eval()
    text_model.requires_grad_(False)

    processor = Qwen2VLProcessor.from_pretrained(str(_PROCESSOR_PATH), local_files_only=True)
    runtime = QwenImageTextEncoderRuntime(
        model=text_model,
        processor=processor,
        compute_dtype=compute_dtype,
    )
    load_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
    offload_device = memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER)
    patcher = ModelPatcher(
        text_model,
        load_device=load_device,
        offload_device=offload_device,
        current_device=cpu_device,
    )
    return runtime, patcher


def _load_vae(
    *,
    path: Path,
    config_mapping: Mapping[str, object],
    compute_dtype: torch.dtype,
) -> tuple[AutoencoderCodex3D, VAE]:
    cpu_device = memory_management.manager.cpu_device
    raw_state_dict = load_torch_file(str(path), device=cpu_device)
    if not isinstance(raw_state_dict, MutableMapping):
        raise RuntimeError(
            "Qwen Image VAE SafeTensors loader must return a mutable mapping; "
            f"got {type(raw_state_dict).__name__}."
        )
    _style, resolved_view = resolve_codex3d_vae_keyspace(raw_state_dict)
    native_config = sanitize_codex3d_vae_config(config_mapping)
    with using_codex_operations(
        device=cpu_device,
        dtype=compute_dtype,
        manual_cast_enabled=True,
    ):
        vae_model = AutoencoderCodex3D(**native_config)
    if len(resolved_view) != _VAE_EXECUTABLE_KEYS:
        raise RuntimeError(
            "Qwen Image VAE executable key count mismatch before load. "
            f"got={len(resolved_view)} expected={_VAE_EXECUTABLE_KEYS}."
        )
    missing, unexpected = safe_load_state_dict(
        vae_model,
        resolved_view,
        log_name="Qwen Image Edit-2511 VAE",
    )
    if missing or unexpected:
        raise RuntimeError(
            "Qwen Image VAE strict binding failed. "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}."
        )
    runtime_keys = set(vae_model.state_dict().keys())
    source_keys = set(resolved_view.keys())
    if runtime_keys != source_keys:
        raise RuntimeError(
            "Qwen Image VAE post-load runtime/source keysets differ. "
            f"missing_runtime={sorted(source_keys - runtime_keys)[:10]} "
            f"extra_runtime={sorted(runtime_keys - source_keys)[:10]}."
        )
    vae_model.eval()
    vae_model.requires_grad_(False)
    vae_load_device = memory_management.manager.get_device(DeviceRole.VAE)
    vae = VAE(
        model=vae_model,
        device=vae_load_device,
        dtype=compute_dtype,
        family=ModelFamily.QWEN_IMAGE,
    )
    vae.patcher.set_model_vae_regulation(_posterior_mode)
    return vae_model, vae


def load_qwen_image_components(
    bundle: DiffusionModelBundle,
    *,
    options: Mapping[str, object],
) -> QwenImageComponentAssembly:
    if bundle.family is not ModelFamily.QWEN_IMAGE:
        raise RuntimeError(
            "Qwen Image component loader received the wrong family: "
            f"{getattr(bundle.family, 'value', bundle.family)!r}."
        )
    if bundle.source != "state_dict":
        raise RuntimeError(
            "Qwen Image Edit-2511 requires the exact core GGUF state-dict source; "
            f"got source={bundle.source!r}."
        )
    if options.get("model_format") != "gguf":
        raise RuntimeError("Qwen Image Edit-2511 requires model_format='gguf'.")
    if options.get("checkpoint_core_only") is not True:
        raise RuntimeError("Qwen Image Edit-2511 requires checkpoint_core_only=True.")
    if options.get("tenc_source") != "external":
        raise RuntimeError("Qwen Image Edit-2511 requires tenc_source='external'.")
    if options.get("vae_source") != "external":
        raise RuntimeError("Qwen Image Edit-2511 requires vae_source='external'.")

    variant = require_qwen_image_variant(
        options.get(QWEN_IMAGE_VARIANT_KEY),
        context=f"Qwen Image component option {QWEN_IMAGE_VARIANT_KEY!r}",
    )
    bundle_variant = require_qwen_image_variant(
        bundle.metadata.get(QWEN_IMAGE_VARIANT_KEY),
        context=f"Qwen Image bundle metadata {QWEN_IMAGE_VARIANT_KEY!r}",
    )
    if variant != bundle_variant or variant != QWEN_IMAGE_EDIT_VARIANT:
        raise RuntimeError(
            "Qwen Image component variant mismatch. "
            f"option={variant!r} bundle={bundle_variant!r} expected={QWEN_IMAGE_EDIT_VARIANT!r}."
        )
    metadata_root = _resolved_path(bundle.metadata.get("metadata_root"), field="metadata_root")
    if metadata_root != _METADATA_ROOT.resolve(strict=True):
        raise RuntimeError(
            "Qwen Image bundle metadata root does not match the vendored Edit-2511 owner. "
            f"got={metadata_root} expected={_METADATA_ROOT.resolve(strict=True)}."
        )
    if metadata_root.name != QWEN_IMAGE_EDIT_REPO_ID.rsplit("/", 1)[-1]:
        raise RuntimeError(f"Qwen Image metadata root has unexpected repository name: {metadata_root}.")

    text_encoder_path = _require_same_path(
        options.get("tenc_path"),
        bundle.metadata.get("tenc_path"),
        field="tenc_path",
    )
    vae_path = _require_same_path(
        options.get("vae_path"),
        bundle.metadata.get("vae_path"),
        field="vae_path",
    )

    transformer_state = bundle.components.get("transformer")
    if not isinstance(transformer_state, MutableMapping):
        raise RuntimeError(
            "Qwen Image bundle must expose a mutable lazy transformer mapping at components['transformer']; "
            f"got {type(transformer_state).__name__}."
        )

    transformer_mapping = _read_json_object(
        _TRANSFORMER_CONFIG_PATH,
        context="Qwen Image transformer config",
    )
    text_encoder_mapping = _read_json_object(
        _TEXT_ENCODER_CONFIG_PATH,
        context="Qwen Image text encoder config",
    )
    vae_mapping = _read_json_object(
        _VAE_CONFIG_PATH,
        context="Qwen Image VAE config",
    )
    scheduler_mapping = _read_json_object(
        _SCHEDULER_CONFIG_PATH,
        context="Qwen Image scheduler config",
    )
    transformer_config = qwen_image_transformer_config_from_mapping(
        transformer_mapping,
        variant=variant,
        context=str(_TRANSFORMER_CONFIG_PATH),
    )
    text_encoder_config = qwen_image_text_encoder_config_from_mapping(
        text_encoder_mapping,
        context=str(_TEXT_ENCODER_CONFIG_PATH),
    )
    vae_config = qwen_image_vae_config_from_mapping(
        vae_mapping,
        context=str(_VAE_CONFIG_PATH),
    )
    scheduler_config = qwen_image_scheduler_config_from_mapping(
        scheduler_mapping,
        context=str(_SCHEDULER_CONFIG_PATH),
    )

    compute_dtype = torch.bfloat16
    transformer, denoiser = _load_transformer(
        state_dict=transformer_state,
        config=transformer_config,
        compute_dtype=compute_dtype,
    )
    text_encoder, text_encoder_patcher = _load_text_encoder(
        path=text_encoder_path,
        config_mapping=text_encoder_mapping,
        compute_dtype=compute_dtype,
    )
    vae_model, vae = _load_vae(
        path=vae_path,
        config_mapping=vae_mapping,
        compute_dtype=compute_dtype,
    )
    return QwenImageComponentAssembly(
        variant=variant,
        metadata_root=metadata_root,
        transformer_config=transformer_config,
        transformer=transformer,
        denoiser=denoiser,
        text_encoder_config=text_encoder_config,
        text_encoder=text_encoder,
        text_encoder_patcher=text_encoder_patcher,
        vae_config=vae_config,
        vae_model=vae_model,
        vae=vae,
        scheduler_config=scheduler_config,
    )


__all__ = ["QwenImageComponentAssembly", "load_qwen_image_components"]

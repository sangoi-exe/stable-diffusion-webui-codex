"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Microsoft Lens runtime bootstrap probe for dependency, tokenizer, config, and shard readiness.
Returns metadata-only readiness status for the parked `lens` engine without importing upstream `lens`, constructing Diffusers pipelines, loading SafeTensors payloads, or claiming runnable generation support.

Symbols (top-level; keep in sync; no ghosts):
- `LensDependencyStatus` (dataclass): Local GPT-OSS/MXFP4/kernels dependency probe result.
- `LensRuntimeBootstrapStatus` (dataclass): Aggregate Lens bootstrap readiness status.
- `LensSelectedLayerStatus` (dataclass): Config-only selected-layer bounds proof.
- `LensTextEncoderConfigStatus` (dataclass): GPT-OSS text-encoder config proof.
- `LensTokenizerBootstrapStatus` (dataclass): Direct tokenizer/chat-template bootstrap proof.
- `probe_lens_dependencies` (function): Probe local Transformers GPT-OSS, MXFP4 config, and kernels availability.
- `probe_lens_runtime_bootstrap` (function): Validate a Lens folder and return component-specific readiness status.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from apps.backend.engines.lens.factory import LensSafeTensorsIndexSummary, validate_lens_folder
from apps.backend.runtime.families.lens.config import (
    LENS_MAX_SEQUENCE_LENGTH,
    LENS_MXFP4_QUANT_METHOD,
    LENS_SELECTED_LAYER_INDEX,
    LENS_TEXT_HIDDEN_DIM,
    LENS_TEXT_MODEL_TYPE,
    LENS_TEXT_NUM_HIDDEN_LAYERS,
    require_lens_variant,
)
from apps.backend.runtime.families.lens.text_encoder import (
    load_lens_tokenizer,
    render_lens_chat_prompt,
    tokenize_lens_prompt_texts,
    validate_lens_selected_layers,
)


@dataclass(frozen=True, slots=True)
class LensDependencyStatus:
    gpt_oss_available: bool
    mxfp4_config_available: bool
    kernels_available: bool
    native_mxfp4_supported: bool
    transformers_version: str | None
    tokenizers_version: str | None
    kernels_version: str | None
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LensTokenizerBootstrapStatus:
    loaded: bool
    tokenizer_dir: str
    max_sequence_length: int
    bos_token: str | None
    eos_token: str | None
    pad_token: str | None
    clean_up_tokenization_spaces: bool
    contains_return_marker: bool
    encoded_sequence_length: int


@dataclass(frozen=True, slots=True)
class LensTextEncoderConfigStatus:
    loaded: bool
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    quant_method: str


@dataclass(frozen=True, slots=True)
class LensSelectedLayerStatus:
    valid: bool
    selected_layers: tuple[int, ...]
    num_hidden_layers: int


@dataclass(frozen=True, slots=True)
class LensRuntimeBootstrapStatus:
    root: str
    variant: str
    trusted_identity_variant: str | None
    metadata_validation_mode: str
    indexed_weight_size: int
    dependencies: LensDependencyStatus
    tokenizer: LensTokenizerBootstrapStatus
    text_encoder_config: LensTextEncoderConfigStatus
    selected_layers: LensSelectedLayerStatus
    text_encoder_shards_present: bool
    text_encoder_headers_valid: bool
    transformer_shards_present: bool
    full_lens_runtime_ready: bool
    runtime_ready_status: str
    runtime_ready_error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _version_or_none(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_lens_dependencies() -> LensDependencyStatus:
    errors: list[str] = []
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - surfaced as dependency status
        gpt_oss_available = False
        errors.append(f"gpt_oss:{type(exc).__name__}:{exc}")
    else:
        gpt_oss_available = True

    try:
        from transformers import Mxfp4Config

        Mxfp4Config(dequantize=True)
    except Exception as exc:  # noqa: BLE001 - surfaced as dependency status
        mxfp4_config_available = False
        errors.append(f"mxfp4_config:{type(exc).__name__}:{exc}")
    else:
        mxfp4_config_available = True

    kernels_available = importlib.util.find_spec("kernels") is not None
    native_mxfp4_supported = kernels_available and mxfp4_config_available
    return LensDependencyStatus(
        gpt_oss_available=gpt_oss_available,
        mxfp4_config_available=mxfp4_config_available,
        kernels_available=kernels_available,
        native_mxfp4_supported=native_mxfp4_supported,
        transformers_version=_version_or_none("transformers"),
        tokenizers_version=_version_or_none("tokenizers"),
        kernels_version=_version_or_none("kernels"),
        errors=tuple(errors),
    )


def _quant_method_from_config(config: object) -> str:
    raw_quantization = getattr(config, "quantization_config", None)
    if isinstance(raw_quantization, Mapping):
        raw_method = raw_quantization.get("quant_method")
    else:
        raw_method = getattr(raw_quantization, "quant_method", None)
    return "" if raw_method is None else str(raw_method)


def _probe_text_encoder_config(root: Path) -> LensTextEncoderConfigStatus:
    try:
        from transformers import GptOssConfig
    except Exception as exc:  # noqa: BLE001 - explicit dependency gate
        raise RuntimeError("Lens text encoder config proof requires transformers.GptOssConfig.") from exc
    config = GptOssConfig.from_pretrained(str(root / "text_encoder"), local_files_only=True)
    model_type = str(getattr(config, "model_type", ""))
    hidden_size = int(getattr(config, "hidden_size", 0))
    num_hidden_layers = int(getattr(config, "num_hidden_layers", 0))
    quant_method = _quant_method_from_config(config)
    if model_type != LENS_TEXT_MODEL_TYPE:
        raise RuntimeError(f"Lens text encoder model_type expected {LENS_TEXT_MODEL_TYPE!r}, got {model_type!r}.")
    if hidden_size != LENS_TEXT_HIDDEN_DIM:
        raise RuntimeError(f"Lens text encoder hidden_size expected {LENS_TEXT_HIDDEN_DIM}, got {hidden_size}.")
    if num_hidden_layers != LENS_TEXT_NUM_HIDDEN_LAYERS:
        raise RuntimeError(
            f"Lens text encoder num_hidden_layers expected {LENS_TEXT_NUM_HIDDEN_LAYERS}, got {num_hidden_layers}."
        )
    if quant_method != LENS_MXFP4_QUANT_METHOD:
        raise RuntimeError(f"Lens text encoder quant method expected {LENS_MXFP4_QUANT_METHOD!r}, got {quant_method!r}.")
    return LensTextEncoderConfigStatus(
        loaded=True,
        model_type=model_type,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        quant_method=quant_method,
    )


def _probe_tokenizer(root: Path) -> LensTokenizerBootstrapStatus:
    bundle = load_lens_tokenizer(root / "tokenizer")
    rendered = render_lens_chat_prompt(bundle.tokenizer, "a cat")
    encoded = tokenize_lens_prompt_texts(bundle.tokenizer, [rendered.tokenization_text])
    input_ids = encoded["input_ids"]
    return LensTokenizerBootstrapStatus(
        loaded=True,
        tokenizer_dir=str(bundle.tokenizer_dir),
        max_sequence_length=int(bundle.tokenizer.model_max_length),
        bos_token=bundle.tokenizer.bos_token,
        eos_token=bundle.tokenizer.eos_token,
        pad_token=bundle.tokenizer.pad_token,
        clean_up_tokenization_spaces=bundle.clean_up_tokenization_spaces,
        contains_return_marker=rendered.contains_return_marker,
        encoded_sequence_length=int(input_ids.shape[1]),
    )


def _component_shards_present(index: LensSafeTensorsIndexSummary) -> bool:
    index_path = Path(index.index_path)
    return all((index_path.parent / shard).is_file() for shard in index.shards)


def _component_headers_valid(index: LensSafeTensorsIndexSummary) -> bool:
    if not _component_shards_present(index):
        return False
    try:
        from safetensors import safe_open
    except Exception as exc:  # noqa: BLE001 - explicit dependency gate
        raise RuntimeError("Lens SafeTensors header validation requires safetensors.safe_open.") from exc
    index_path = Path(index.index_path)
    for shard in index.shards:
        shard_path = index_path.parent / shard
        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                tuple(handle.keys())
        except Exception as exc:  # noqa: BLE001 - keep shard path context
            raise RuntimeError(f"Lens SafeTensors header validation failed for {shard_path}.") from exc
    return True


def _runtime_ready_probe(root: Path, *, variant: str) -> tuple[str, str | None]:
    try:
        validate_lens_folder(str(root), variant=variant, validation_mode="runtime_ready")
    except RuntimeError as exc:
        detail = str(exc)
        if "runtime_ready validation missing shard" in detail:
            return "weights_missing", detail
        return "invalid", detail
    return "ready", None


def probe_lens_runtime_bootstrap(model_ref: str | Path, *, variant: object) -> LensRuntimeBootstrapStatus:
    variant_value = require_lens_variant(variant, context="Lens bootstrap variant")
    root = Path(model_ref).expanduser()
    metadata = validate_lens_folder(str(root), variant=variant_value, validation_mode="metadata_only")
    dependencies = probe_lens_dependencies()
    tokenizer_status = _probe_tokenizer(root)
    config_status = _probe_text_encoder_config(root)
    selected_layers = validate_lens_selected_layers(
        config_status.num_hidden_layers,
        selected_layers=LENS_SELECTED_LAYER_INDEX,
    )
    selected_status = LensSelectedLayerStatus(
        valid=True,
        selected_layers=selected_layers,
        num_hidden_layers=config_status.num_hidden_layers,
    )
    text_encoder_shards_present = _component_shards_present(metadata.text_encoder_index)
    transformer_shards_present = _component_shards_present(metadata.transformer_index)
    text_encoder_headers_valid = _component_headers_valid(metadata.text_encoder_index)
    runtime_ready_status, runtime_ready_error = _runtime_ready_probe(root, variant=variant_value)
    full_lens_runtime_ready = (
        runtime_ready_status == "ready"
        and text_encoder_shards_present
        and text_encoder_headers_valid
        and transformer_shards_present
    )
    if tokenizer_status.max_sequence_length != LENS_MAX_SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Lens tokenizer max length expected {LENS_MAX_SEQUENCE_LENGTH}, got {tokenizer_status.max_sequence_length}."
        )
    return LensRuntimeBootstrapStatus(
        root=str(metadata.root),
        variant=variant_value,
        trusted_identity_variant=metadata.trusted_identity_variant,
        metadata_validation_mode=metadata.validation_mode,
        indexed_weight_size=metadata.total_indexed_weight_size,
        dependencies=dependencies,
        tokenizer=tokenizer_status,
        text_encoder_config=config_status,
        selected_layers=selected_status,
        text_encoder_shards_present=text_encoder_shards_present,
        text_encoder_headers_valid=text_encoder_headers_valid,
        transformer_shards_present=transformer_shards_present,
        full_lens_runtime_ready=full_lens_runtime_ready,
        runtime_ready_status=runtime_ready_status,
        runtime_ready_error=runtime_ready_error,
    )


__all__ = [
    "LensDependencyStatus",
    "LensRuntimeBootstrapStatus",
    "LensSelectedLayerStatus",
    "LensTextEncoderConfigStatus",
    "LensTokenizerBootstrapStatus",
    "probe_lens_dependencies",
    "probe_lens_runtime_bootstrap",
]

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Central model loader for diffusion engines (checkpoint/diffusers parsing, component assembly, and runtime-friendly overrides).
Resolves TE/VAE overrides (`tenc_path` shorthand), applies family-scoped keyspace interpretation plus source-key rewrite guards where needed, and selects storage/compute dtypes
(storage defaults to weights primary SafeTensors dtype when detectable; compute defaults to fp32 for stability unless overridden).
Includes core-only families (e.g., Anima and Z-Image L2P) that are not diffusers repositories: the loader returns a minimal bundle and leaves external asset loading to engines.
Partial-metadata fallback stays structural only; loader compatibility no longer invents inpaint-model semantics from channels or name hints.
NF4/FP4 is not supported (fail loud); GGUF is the only supported pre-quant format.
WAN22 variants use explicit families (`WAN22_5B`, `WAN22_14B`, `WAN22_ANIMATE`) with no shared family alias bucket.
SDXL loads are strict: missing/unexpected keys are fatal to surface drift early.
Flux T5 component loading now guarantees model construction before state-dict load for both GGUF and non-GGUF paths, and delegates T5 keyspace interpretation to a canonical keymap module.
FLUX.2 Klein 4B/base-4B expected-family loads use vendored HF metadata plus family-scoped keyspace resolution so native/source GGUF keys and
legacy fused core slices land in the Diffusers `Flux2Transformer2DModel` lookup space without mutating the checkpoint; unsupported FLUX.2
variants/configs fail loud.
Qwen Image is admitted only as the exact core-only Edit-2511 GGUF plus one exact external Qwen2.5-VL GGUF and VAE SafeTensors asset.
HF-style Qwen metadata directories remain validation inputs only and fail loud when selected as executable models.
SDXL VAE conversion now preflights canonical projection keys after keyspace resolution so projection-lane shape violations surface explicitly (instead of collapsing into generic missing-key noise).
GGUF smart-offload staging for large transformer classes now emits canonical INFO audit events via `backend.smart_offload`.
Those staging events are tagged via the canonical `SmartOffloadAction.STAGE_LOAD` enum action.

Symbols (top-level; keep in sync; no ghosts):
- `ParsedCheckpoint` (dataclass): Parsed checkpoint bundle (primary path + optional additional modules + extracted configs/metadata).
- `CheckpointInspection` (dataclass): Checkpoint inspection envelope (format, key count, optional safetensors header/shape metadata).
- `DiffusionModelBundle` (dataclass): Loaded model components and configs (UNet/VAE/text encoders + signature + quant/layout info).
- `_supported_inference_dtypes` (function): Returns supported inference dtypes for a given model family.
- `_prediction_type_value` (function): Converts a `PredictionKind` into the string value expected by configs/pipelines.
- `_clip_layout_cache_key` (function): Builds the canonical `(family component)` cache key used for SHA layout metadata lookups.
- `_clip_layout_metadata_from_cache` (function): Parses cached layout metadata payloads into typed CLIP layout metadata.
- `_clip_layout_metadata_to_cache` (function): Serializes resolved CLIP layout metadata for registry cache persistence.
- `_clip_layout_hint_from_cache` (function): Applies precedence rules to decide when cached layout metadata may be consumed.
- `_projection_module_layout` (function): Resolves the projection module layout (`linear`/`matmul`) from resolved CLIP metadata.
- `_coerce_safetensors_shape` (function): Normalizes safetensors header shape payloads into integer tuples.
- `_inspect_checkpoint` (function): Inspects checkpoint metadata before loading (safetensors header-first; non-safetensors format probe).
- `_attach_checkpoint_inspection` (function): Wires safetensors inspection metadata onto lazy mappings when supported.
- `_load_state_dict` (function): Loads a state dict from disk using inspection metadata and emits materialization events for non-safetensors.
- `_read_json` (function): Reads a required JSON metadata file with explicit errors (used by vendored-HF signature builders).
- `_load_diffusers_model_index` (function): Reads a diffusers `model_index.json` natively from disk (no Diffusers config helper).
- `_zimage_signature_from_vendored_hf` (function): Builds a Z-Image `ModelSignature` from vendored HF metadata (`Tongyi-MAI/Z-Image-Turbo` layout; no state-dict detector).
- `_flux_signature_from_vendored_hf` (function): Builds a Flux `ModelSignature` from vendored HF metadata (no state-dict detector).
- `_resolve_flux2_repo_id_from_path` (function): Chooses the supported FLUX.2 4B/base-4B vendored repo id from a checkpoint path hint.
- `_validate_supported_flux2_transformer_config` (function): Validates that a FLUX.2 transformer config matches the supported Klein 4B/base-4B slice.
- `_flux2_signature_from_vendored_hf` (function): Builds a FLUX.2 `ModelSignature` from vendored HF metadata for the supported 4B/base-4B slice.
- `_qwen_image_bundle_from_diffusers_metadata` (function): Validates the exact vendored Edit-2511 metadata surface, then rejects metadata-only model selection.
- `_sdxl_expected_signature_from_state_dict` (function): Builds an SDXL/SDXL refiner signature from expected-family checkpoint truth, including core-only detection.
- `_requires_sdxl_checkpoint_keymap` (function): Determines whether SDXL checkpoint keyspace resolution must run for a checkpoint parse call.
- `_maybe_resolve_expected_family_keyspace` (function): Applies family-scoped GGUF/native keyspace interpretation for expected-family loads.
- `_parse_checkpoint` (function): Parses one checkpoint (plus optional addons) into `ParsedCheckpoint` for bundle assembly.
- `_build_diffusion_bundle` (function): Assembles a `DiffusionModelBundle` from a parsed checkpoint and loader options.
- `_load_component_config` (function): Loads a component config dict from a diffusers component directory.
- `_resolve_vae_class` (function): Picks the VAE class/loader path based on model signature and layout (`diffusers` vs legacy layouts).
- `_maybe_convert_sdxl_vae_state_dict` (function): Applies SDXL-specific VAE key conversions and preflights canonical projection keys for explicit lane-shape failures.
- `_detect_sdxl_vae_projection_lane` (function): Detects SDXL VAE canonical projection lane (`linear_2d` or `conv1x1_4d`) from resolved canonical keys.
- `_assert_vae_state_dict_keyspace_for_family` (function): Validates Flow-family VAE keyspace and rejects non-VAE assets with explicit causality.
- `_Conv1x1Projection` (class): Native 1x1-conv projection module compatible with diffusers attention 3D inputs.
- `_apply_sdxl_vae_conv_projection_lane` (function): Replaces SDXL VAE mid-attention projection modules with native 1x1-conv projections.
- `_detect_vae_layout` (function): Detects VAE state dict layout (used to choose conversion/loading strategy).
- `_assert_core_only_vae_path_not_checkpoint` (function): Rejects core-only requests where `vae_path` equals the checkpoint path.
- `_safetensors_primary_dtype_hint` (function): Best-effort safetensors dtype hint reader (header-only, whole-file).
- `_log_weights_dtype_hint` (function): Emits pipeline-debug logs for (role, selected dtype, weights dtype hint).
- `_load_huggingface_component` (function): Loads a diffusers component/pipeline from a local HF-style repo directory (including canonical keymap-owned T5 keyspace resolution).
- `_apply_prediction_type` (function): Applies prediction-type overrides to loaded components/configs when specified.
- `codex_loader` (function): Primary loader entrypoint; coordinates checkpoint parsing, TE override resolution (incl. `tenc_path` shorthand),
  VAE layout handling, dtype selection, and memory-management integration to produce a `DiffusionModelBundle`.
- `_SimpleEstimated` (class): Minimal estimate container used for config detection/compat when only partial metadata is available.
- `resolve_sdxl_diffusers_surface` (function): Resolves SDXL base vs refiner diffusers repos from native JSON/config evidence and expected-family truth.
- `_detect_engine_from_config` (function): Detects engine identifier from a diffusers config dict.
- `load_engine_from_diffusers` (function): Loads a `DiffusionModelBundle` directly from a diffusers repo directory.
- `resolve_diffusion_bundle` (function): Resolves and loads a diffusion bundle from either checkpoint paths or diffusers repos based on inputs.
"""

import importlib
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING
from apps.backend.runtime.logging import get_backend_logger

import torch

if TYPE_CHECKING:  # pragma: no cover
    from diffusers import AutoencoderKL, DiffusionPipeline  # noqa: F401
    from transformers import modeling_utils  # noqa: F401

from apps.backend.huggingface.assets import ensure_repo_minimal_files
from apps.backend.core.contracts.text_encoder_slots import classify_text_encoder_slot
from apps.backend.infra.config.args import args
from apps.backend.infra.config.vae_layout_lane import VaeLayoutLane
from apps.backend.runtime import trace as _trace
from apps.backend.runtime.common.nn.clip import IntegratedCLIP
from apps.backend.runtime.common.nn.t5 import IntegratedT5
from apps.backend.runtime.common.vae_lane_policy import (
    detect_vae_layout,
    resolve_vae_layout_lane,
    validate_vae_key_names,
    uses_ldm_native_lane,
)
from apps.backend.runtime.common.vae_ldm import AutoencoderKL_LDM, sanitize_ldm_vae_config
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.memory.smart_offload import (
    SmartOffloadAction,
    log_smart_offload_action,
    smart_offload_enabled,
)
from apps.backend.runtime.model_parser import parse_state_dict
from apps.backend.runtime.model_parser.quantization import detect_state_dict_dtype
from apps.backend.runtime.model_parser.specs import CodexEstimatedConfig
from apps.backend.runtime.families.ltx2.config import LTX2_REQUIRED_TEXT_ENCODER_SLOT
from apps.backend.runtime.families.ltx2.loader import build_ltx2_bundle_metadata, prepare_ltx2_bundle_inputs
from apps.backend.runtime.families.qwen_image.config import (
    QWEN_IMAGE_EDIT_PIPELINE_CLASS,
    QWEN_IMAGE_EDIT_REPO_ID,
    QWEN_IMAGE_EDIT_VARIANT,
    QWEN_IMAGE_ENGINE_ID,
    QWEN_IMAGE_VARIANT_KEY,
)
from apps.backend.runtime.families.qwen_image.scheduler import qwen_image_scheduler_config_from_mapping
from apps.backend.runtime.families.qwen_image.text_encoder import qwen_image_text_encoder_config_from_mapping
from apps.backend.runtime.families.qwen_image.transformer import qwen_image_transformer_config_from_mapping
from apps.backend.runtime.families.qwen_image.vae import (
    qwen_image_validate_external_vae_path,
    qwen_image_vae_config_from_mapping,
)
from apps.backend.runtime.model_registry.errors import ModelRegistryError
from apps.backend.runtime.model_registry.loader import detect_from_state_dict as registry_detect
from apps.backend.runtime.model_registry.signals import build_bundle, count_blocks
from apps.backend.runtime.model_registry.specs import (
    CodexCoreArchitecture,
    CodexCoreSignature,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationHint,
    LatentFormat,
    QuantizationKind,
    TextEncoderSignature,
    VAESignature,
)
from apps.backend.runtime.models import api as model_api
from apps.backend.runtime.models.text_encoder_overrides import (
    _canonical_override_family,
    TextEncoderOverrideConfig,
    TextEncoderOverrideError,
    resolve_text_encoder_override_paths,
)
from apps.backend.runtime.models.state_dict import load_state_dict, transformers_convert
from apps.backend.runtime.ops import using_codex_operations
from apps.backend.runtime.checkpoint.io import load_torch_file, read_arbitrary_config
from apps.backend.runtime.state_dict.tools import beautiful_print_gguf_state_dict_statics
from apps.backend.runtime.state_dict.key_mapping import KeyMappingError
from apps.backend.runtime.state_dict.keymap_sdxl_clip import ClipLayoutMetadata
from apps.backend.runtime.checkpoint.safetensors_header import (
    detect_safetensors_primary_dtype,
    detect_safetensors_primary_dtype_from_header,
    read_safetensors_header,
)

LOGGER = get_backend_logger(__name__)
CLIP_LOG = get_backend_logger(__name__ + ".clip")
_LOG = get_backend_logger(__name__)
_BACKEND_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_INFERENCE_DTYPES: Dict[ModelFamily, tuple[torch.dtype, ...]] = {
    ModelFamily.FLUX: (torch.bfloat16, torch.float16, torch.float32),
    ModelFamily.FLUX_KONTEXT: (torch.bfloat16, torch.float16, torch.float32),
    ModelFamily.FLUX2: (torch.bfloat16, torch.float16, torch.float32),
    ModelFamily.QWEN_IMAGE: (torch.bfloat16, torch.float16, torch.float32),
    ModelFamily.ZIMAGE_L2P: (torch.bfloat16, torch.float16, torch.float32),
    ModelFamily.CHROMA: (torch.bfloat16, torch.float16, torch.float32),
}
DEFAULT_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_CORE_ARCH_LABELS: Dict[CodexCoreArchitecture, str] = {
    CodexCoreArchitecture.UNET: "UNet",
    CodexCoreArchitecture.DIT: "DiT",
    CodexCoreArchitecture.TRANSFORMER: "Transformer",
    CodexCoreArchitecture.FLOW_TRANSFORMER: "FlowTransformer",
}

PREDICTION_TYPE_MAP = {
    PredictionKind.EPSILON: "epsilon",
    PredictionKind.V_PREDICTION: "v_prediction",
    PredictionKind.EDM: "edm",
    PredictionKind.FLOW: "flow",
}


@dataclass
class ParsedCheckpoint:
    signature: ModelSignature
    config: CodexEstimatedConfig


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    path: str
    format: str
    key_count: int
    shapes: Dict[str, tuple[int, ...]] = field(default_factory=dict)
    header: Dict[str, object] | None = None
    primary_dtype_hint: str | None = None


@dataclass(frozen=True, slots=True)
class DiffusionModelBundle:
    """Fully materialised diffusion checkpoint ready for engine binding."""

    model_ref: str
    family: ModelFamily
    estimated_config: Any
    components: Dict[str, Any]
    signature: Optional[ModelSignature] = None
    source: str = "state_dict"
    metadata: Dict[str, Any] = field(default_factory=dict)


ENGINE_KEY_TO_FAMILY: Dict[str, ModelFamily] = {
    "sdxl": ModelFamily.SDXL,
    "sdxl_refiner": ModelFamily.SDXL_REFINER,
    "flux1": ModelFamily.FLUX,
    "flux1_kontext": ModelFamily.FLUX_KONTEXT,
    "flux2": ModelFamily.FLUX2,
    QWEN_IMAGE_ENGINE_ID: ModelFamily.QWEN_IMAGE,
    "sd35": ModelFamily.SD35,
    "sd3": ModelFamily.SD3,
    "flux1_chroma": ModelFamily.CHROMA,
    "sd20": ModelFamily.SD20,
    "sd15": ModelFamily.SD15,
    "anima": ModelFamily.ANIMA,
    "zimage_l2p": ModelFamily.ZIMAGE_L2P,
    "ltx2": ModelFamily.LTX2,
    "wan22_5b": ModelFamily.WAN22_5B,
    "wan22_14b": ModelFamily.WAN22_14B,
    "wan22_14b_animate": ModelFamily.WAN22_ANIMATE,
}

FAMILY_TO_ENGINE_KEY: Dict[ModelFamily, str] = {
    ModelFamily.SDXL_REFINER: "sdxl_refiner",
    ModelFamily.SDXL: "sdxl",
    ModelFamily.FLUX: "flux1",
    ModelFamily.FLUX_KONTEXT: "flux1_kontext",
    ModelFamily.FLUX2: "flux2",
    ModelFamily.QWEN_IMAGE: QWEN_IMAGE_ENGINE_ID,
    ModelFamily.LTX2: "ltx2",
    ModelFamily.SD35: "sd35",
    ModelFamily.SD3: "sd35",
    ModelFamily.CHROMA: "flux1_chroma",
    ModelFamily.SD20: "sd20",
    ModelFamily.SD15: "sd15",
    ModelFamily.ANIMA: "anima",
    ModelFamily.ZIMAGE_L2P: "zimage_l2p",
    ModelFamily.WAN22_5B: "wan22_5b",
    ModelFamily.WAN22_14B: "wan22_14b",
    ModelFamily.WAN22_ANIMATE: "wan22_14b_animate",
}

_WAN22_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily.WAN22_5B,
    ModelFamily.WAN22_14B,
    ModelFamily.WAN22_ANIMATE,
)

_CLIP_LAYOUT_CACHE_PREFIX = "clip"

_SUPPORTED_FLUX2_REPO_IDS: tuple[str, str] = (
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-base-4B",
)

_SUPPORTED_FLUX2_TRANSFORMER_CONFIG: Dict[str, object] = {
    "attention_head_dim": 128,
    "guidance_embeds": False,
    "in_channels": 128,
    "joint_attention_dim": 7680,
    "num_attention_heads": 24,
    "num_layers": 5,
    "num_single_layers": 20,
    "patch_size": 1,
    "timestep_guidance_channels": 256,
}

_QWEN_IMAGE_PIPELINE_VARIANTS: Dict[str, tuple[str, str]] = {
    QWEN_IMAGE_EDIT_PIPELINE_CLASS.lower(): (QWEN_IMAGE_EDIT_REPO_ID, QWEN_IMAGE_EDIT_VARIANT),
}

_SDXL_VAE_CANONICAL_PROJECTION_KEYS = (
    "encoder.mid_block.attentions.0.to_q.weight",
    "encoder.mid_block.attentions.0.to_k.weight",
    "encoder.mid_block.attentions.0.to_v.weight",
    "encoder.mid_block.attentions.0.to_out.0.weight",
    "decoder.mid_block.attentions.0.to_q.weight",
    "decoder.mid_block.attentions.0.to_k.weight",
    "decoder.mid_block.attentions.0.to_v.weight",
    "decoder.mid_block.attentions.0.to_out.0.weight",
)


class _Conv1x1Projection(torch.nn.Module):
    """Projection module that stores native 1x1-conv weights."""

    def __init__(self, *, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = torch.nn.Parameter(torch.empty(self.out_features, self.in_features, 1, 1))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.scale_weight = None
        self.parameters_manual_cast = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight[:, :, 0, 0], a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(max(self.in_features, 1))
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.parameters_manual_cast:
            from apps.backend.runtime.ops.operations import main_stream_worker, weights_manual_cast

            weight, bias, signal = weights_manual_cast(self, hidden_states)
            with main_stream_worker(weight, bias, signal):
                if hidden_states.ndim == 3:
                    return torch.nn.functional.linear(hidden_states, weight[:, :, 0, 0], bias)
                if hidden_states.ndim == 4:
                    return torch.nn.functional.conv2d(hidden_states, weight, bias)
                raise RuntimeError(
                    "SDXL VAE native 4D projection expected 3D or 4D hidden_states; "
                    f"got_ndim={hidden_states.ndim}."
                )
        if hidden_states.ndim == 3:
            return torch.nn.functional.linear(hidden_states, self.weight[:, :, 0, 0], self.bias)
        if hidden_states.ndim == 4:
            return torch.nn.functional.conv2d(hidden_states, self.weight, self.bias)
        raise RuntimeError(
            "SDXL VAE native 4D projection expected 3D or 4D hidden_states; "
            f"got_ndim={hidden_states.ndim}."
        )


def _supported_inference_dtypes(family: ModelFamily) -> tuple[torch.dtype, ...]:
    return SUPPORTED_INFERENCE_DTYPES.get(family, DEFAULT_SUPPORTED_DTYPES)


def _prediction_type_value(prediction: PredictionKind) -> str:
    return PREDICTION_TYPE_MAP.get(prediction, "epsilon")


def _clip_layout_cache_key(component_name: str) -> str:
    return f"{_CLIP_LAYOUT_CACHE_PREFIX}:{component_name}"


def _clip_layout_metadata_from_cache(payload: Mapping[str, object] | None) -> ClipLayoutMetadata | None:
    if payload is None:
        return None
    qkv_layout = str(payload.get("qkv_layout", "")).strip().lower()
    projection_orientation = str(payload.get("projection_orientation", "")).strip().lower()
    source_style_raw = payload.get("source_style")
    source_style = None if source_style_raw is None else str(source_style_raw).strip().lower() or None
    if qkv_layout not in {"split", "fused"}:
        raise RuntimeError(
            f"Invalid cached qkv_layout={qkv_layout!r} for clip layout metadata (expected split|fused)."
        )
    if projection_orientation not in {"none", "linear", "matmul"}:
        raise RuntimeError(
            "Invalid cached projection_orientation=%r for clip layout metadata "
            "(expected none|linear|matmul)." % (projection_orientation,)
        )
    if source_style is not None and source_style not in {"codex", "hf", "openclip"}:
        raise RuntimeError(
            f"Invalid cached source_style={source_style!r} for clip layout metadata (expected codex|hf|openclip)."
        )
    return ClipLayoutMetadata(
        qkv_layout=qkv_layout,
        projection_orientation=projection_orientation,
        source_style=source_style,
    )


def _clip_layout_metadata_to_cache(layout: ClipLayoutMetadata) -> dict[str, str]:
    payload = {
        "qkv_layout": layout.qkv_layout,
        "projection_orientation": layout.projection_orientation,
    }
    if layout.source_style:
        payload["source_style"] = layout.source_style
    return payload


def _clip_layout_hint_from_cache(
    cached_layout: ClipLayoutMetadata | None,
    *,
    allow_cache: bool,
) -> ClipLayoutMetadata | None:
    return cached_layout if allow_cache else None


def _projection_module_layout(add_projection: bool, layout: ClipLayoutMetadata) -> str:
    if not add_projection:
        return "linear"
    if layout.projection_orientation not in {"linear", "matmul"}:
        return "linear"
    return layout.projection_orientation


def _coerce_safetensors_shape(raw_shape: object) -> tuple[int, ...] | None:
    if not isinstance(raw_shape, (list, tuple)):
        return None
    out: list[int] = []
    for value in raw_shape:
        if not isinstance(value, (int, float)):
            return None
        out.append(int(value))
    return tuple(out)


def _inspect_checkpoint(path: str) -> CheckpointInspection:
    path_str = str(path)
    suffix = Path(path_str).suffix.lower()
    format_name = suffix.lstrip(".") or "unknown"
    _trace.event("checkpoint_inspect_start", path=path_str, format=format_name)

    if suffix in {".safetensor", ".safetensors"}:
        header = read_safetensors_header(Path(path_str))
        header_dict = {str(k): v for k, v in header.items()}
        shapes: Dict[str, tuple[int, ...]] = {}
        key_count = 0
        for raw_key, meta in header_dict.items():
            if raw_key == "__metadata__":
                continue
            key_count += 1
            if not isinstance(meta, Mapping):
                continue
            shape = _coerce_safetensors_shape(meta.get("shape"))
            if shape is not None:
                shapes[raw_key] = shape
        primary_dtype_hint = detect_safetensors_primary_dtype_from_header(header_dict)
        inspection = CheckpointInspection(
            path=path_str,
            format="safetensors",
            key_count=key_count,
            shapes=shapes,
            header=header_dict,
            primary_dtype_hint=primary_dtype_hint,
        )
        _trace.event(
            "checkpoint_inspect_done",
            path=path_str,
            format="safetensors",
            keys=inspection.key_count,
            shapes=len(inspection.shapes),
            primary_dtype=str(inspection.primary_dtype_hint or "unknown"),
        )
        return inspection

    inspection = CheckpointInspection(path=path_str, format=format_name, key_count=-1)
    _trace.event(
        "checkpoint_inspect_done",
        path=path_str,
        format=format_name,
        keys=-1,
        shapes=0,
        primary_dtype="unknown",
    )
    return inspection


def _attach_checkpoint_inspection(state_dict: Mapping[str, Any], inspection: CheckpointInspection) -> None:
    if inspection.format != "safetensors":
        return
    if not hasattr(state_dict, "__dict__"):
        return
    try:
        setattr(state_dict, "source_format", "safetensors")
        setattr(state_dict, "source_path", inspection.path)
        setattr(state_dict, "header_shapes", dict(inspection.shapes))
        setattr(state_dict, "safetensors_header", dict(inspection.header or {}))
        setattr(state_dict, "primary_dtype_hint", inspection.primary_dtype_hint)
    except Exception:
        # Best-effort metadata wiring; never weaken checkpoint load strictness.
        pass


def _load_state_dict(path: str, *, inspection: CheckpointInspection | None = None) -> Mapping[str, Any]:
    inspection = inspection or _inspect_checkpoint(path)
    _trace.event("load_torch_file_start", path=str(path))
    # Resolve the initial load device explicitly (no 'auto' fallback)
    initial_device = memory_management.manager.get_offload_device(DeviceRole.CORE)
    if inspection.format != "safetensors":
        _trace.event("checkpoint_materialize_start", path=str(path), format=inspection.format)
    sd = load_torch_file(path, device=initial_device)
    if inspection.format == "safetensors":
        _attach_checkpoint_inspection(sd, inspection)
    try:
        tensor_count = len(sd.keys())  # type: ignore[attr-defined]
    except Exception:
        tensor_count = -1
    if inspection.format != "safetensors":
        _trace.event(
            "checkpoint_materialize_done",
            path=str(path),
            format=inspection.format,
            tensors=tensor_count,
        )
    _trace.event("load_torch_file_done", path=str(path), type=type(sd).__name__, tensors=tensor_count)
    return sd

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required metadata file missing: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON metadata file: {path}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(data).__name__}")
    return data


def _load_diffusers_model_index(repo_dir: str) -> Dict[str, Any]:
    return _read_json(Path(repo_dir) / "model_index.json")


def _zimage_signature_from_vendored_hf(*, model_path: str) -> ModelSignature:
    """Build a Z-Image `ModelSignature` from vendored HF metadata (no state-dict detection)."""

    vendor_root = _BACKEND_ROOT / "huggingface" / "Tongyi-MAI" / "Z-Image-Turbo"
    transformer_cfg = _read_json(vendor_root / "transformer" / "config.json")
    text_encoder_cfg = _read_json(vendor_root / "text_encoder" / "config.json")

    patch_size = transformer_cfg.get("all_patch_size")
    if isinstance(patch_size, list) and patch_size:
        patch_size = patch_size[0]
    patch_size = int(patch_size) if isinstance(patch_size, (int, float, str)) and str(patch_size).strip() else 2
    patch_area = int(max(1, patch_size) * max(1, patch_size))

    latent_channels_raw = transformer_cfg.get("in_channels", 16)
    latent_channels = int(latent_channels_raw) if isinstance(latent_channels_raw, (int, float, str)) else 16

    hidden_dim_raw = transformer_cfg.get("dim", 3840)
    hidden_dim = int(hidden_dim_raw) if isinstance(hidden_dim_raw, (int, float, str)) else 3840

    context_dim_raw = transformer_cfg.get("cap_feat_dim")
    if context_dim_raw is None:
        context_dim_raw = text_encoder_cfg.get("hidden_size", 2560)
    context_dim = int(context_dim_raw) if isinstance(context_dim_raw, (int, float, str)) else 2560

    num_layers_raw = transformer_cfg.get("n_layers", 30)
    num_layers = int(num_layers_raw) if isinstance(num_layers_raw, (int, float, str)) else 30

    num_refiner_layers_raw = transformer_cfg.get("n_refiner_layers", 2)
    num_refiner_layers = int(num_refiner_layers_raw) if isinstance(num_refiner_layers_raw, (int, float, str)) else 2

    num_heads_raw = transformer_cfg.get("n_heads", 30)
    num_heads = int(num_heads_raw) if isinstance(num_heads_raw, (int, float, str)) else 30

    suffix = Path(model_path).suffix.lower()
    quantization = QuantizationHint()
    gguf_core_only = False
    if suffix == ".gguf":
        quantization = QuantizationHint(kind=QuantizationKind.GGUF, detail="file_extension")
        gguf_core_only = True

    channels_out = latent_channels
    channels_in = latent_channels * patch_area

    extras = {
        "hidden_dim": hidden_dim,
        "context_dim": context_dim,
        "num_layers": num_layers,
        "num_refiner_layers": num_refiner_layers,
        "num_heads": num_heads,
        "latent_channels": latent_channels,
        "guidance_embeds": False,
        "gguf_core_only": gguf_core_only,
        "signature_source": "vendored_hf",
    }

    text_encoders = [
        TextEncoderSignature(
            name="qwen3_4b",
            key_prefix="text_encoder.",
            expected_dim=context_dim,
            tokenizer_hint="Qwen/Qwen3-4B",
        )
    ]

    # Z-Image supports embedded VAE/text-encoder in non-GGUF exports, but the engine
    # decides core-only vs full based on parsed components. Keep VAE signature
    # optional here so metadata drives the pipeline without constraining assets.
    vae: VAESignature | None = None if gguf_core_only else VAESignature(key_prefix="vae.", latent_channels=latent_channels)

    return ModelSignature(
        family=ModelFamily.ZIMAGE,
        repo_hint="Tongyi-MAI/Z-Image-Turbo",
        prediction=PredictionKind.FLOW,
        latent_format=LatentFormat.ZIMAGE,
        quantization=quantization,
        core=CodexCoreSignature(
            architecture=CodexCoreArchitecture.DIT,
            channels_in=channels_in,
            channels_out=channels_out,
            context_dim=context_dim,
            temporal=False,
            depth=num_layers,
            key_prefixes=["layers."],
        ),
        text_encoders=text_encoders,
        vae=vae,
        extras=extras,
    )


def _flux_signature_from_vendored_hf(
    *,
    model_path: str,
    expected_family: ModelFamily,
    has_guidance: bool | None = None,
) -> ModelSignature:
    """Build a Flux `ModelSignature` from vendored HF metadata (no state-dict detection)."""

    vendor_root = _BACKEND_ROOT / "huggingface" / "black-forest-labs"
    if expected_family is ModelFamily.FLUX_KONTEXT:
        repo_hint = "black-forest-labs/FLUX.1-Kontext-dev"
        transformer_cfg = _read_json(vendor_root / "FLUX.1-Kontext-dev" / "transformer" / "config.json")
    else:
        # Default Flux core signature (dev/schnell share the same architecture, but guidance differs).
        if has_guidance is False:
            repo_hint = "black-forest-labs/FLUX.1-schnell"
            transformer_cfg = _read_json(vendor_root / "FLUX.1-schnell" / "transformer" / "config.json")
        else:
            repo_hint = "black-forest-labs/FLUX.1-dev"
            transformer_cfg = _read_json(vendor_root / "FLUX.1-dev" / "transformer" / "config.json")

    latent_channels_raw = transformer_cfg.get("in_channels", 64)
    latent_channels = int(latent_channels_raw) if isinstance(latent_channels_raw, (int, float, str)) else 64

    context_dim_raw = transformer_cfg.get("joint_attention_dim", 4096)
    context_dim = int(context_dim_raw) if isinstance(context_dim_raw, (int, float, str)) else 4096

    double_layers_raw = transformer_cfg.get("num_layers", 19)
    double_layers = int(double_layers_raw) if isinstance(double_layers_raw, (int, float, str)) else 19

    single_layers_raw = transformer_cfg.get("num_single_layers", 38)
    single_layers = int(single_layers_raw) if isinstance(single_layers_raw, (int, float, str)) else 38

    guidance_embed_raw = transformer_cfg.get("guidance_embeds", False)
    if not isinstance(guidance_embed_raw, bool):
        raise RuntimeError(
            "Flux signature metadata error: 'guidance_embeds' must be a boolean in vendored transformer config "
            f"(got {type(guidance_embed_raw).__name__})."
        )
    guidance_embed = guidance_embed_raw

    suffix = Path(model_path).suffix.lower()
    quantization = QuantizationHint()
    gguf_core_only = False
    if suffix == ".gguf":
        quantization = QuantizationHint(kind=QuantizationKind.GGUF, detail="file_extension")
        gguf_core_only = True

    extras = {
        "flow_double_layers": double_layers,
        "flow_single_layers": single_layers,
        "guidance_embed": guidance_embed,
        "gguf_core_only": gguf_core_only,
        "signature_source": "vendored_hf",
    }

    text_encoders = [
        TextEncoderSignature(
            name="clip_l",
            key_prefix="text_encoders.clip_l.",
            expected_dim=768,
            tokenizer_hint=f"{repo_hint}/tokenizer",
        ),
        TextEncoderSignature(
            name="t5xxl",
            key_prefix="text_encoders.t5xxl.",
            expected_dim=4096,
            tokenizer_hint=f"{repo_hint}/tokenizer_2",
        ),
    ]

    # Flux core-only checkpoints require external VAE in the engine loader path.
    vae: VAESignature | None = None if gguf_core_only else VAESignature(key_prefix="vae.", latent_channels=16)

    return ModelSignature(
        family=expected_family,
        repo_hint=repo_hint,
        prediction=PredictionKind.FLOW,
        latent_format=LatentFormat.FLOW16,
        quantization=quantization,
        core=CodexCoreSignature(
            architecture=CodexCoreArchitecture.FLOW_TRANSFORMER,
            channels_in=latent_channels,
            channels_out=latent_channels,
            context_dim=context_dim,
            temporal=False,
            depth=double_layers + single_layers,
            key_prefixes=["transformer.", "model.diffusion_model."],
        ),
        text_encoders=text_encoders,
        vae=vae,
        extras=extras,
    )


def _resolve_flux2_repo_id_from_path(model_path: str) -> str:
    lower = str(model_path or "").strip().lower().replace("\\", "/")
    if any(
        marker in lower
        for marker in (
            "flux.2-klein-base-9b",
            "flux2-klein-base-9b",
            "flux.2-klein-9b",
            "flux2-klein-9b",
            "base-9b",
            "/9b/",
            "-9b",
        )
    ):
        raise RuntimeError(
            "Unsupported FLUX.2 checkpoint variant in %r. Only Klein 4B/base-4B is supported."
            % (model_path,)
        )
    if any(
        marker in lower
        for marker in (
            "flux.2-klein-base-4b",
            "flux2-klein-base-4b",
            "base-4b",
            "base_4b",
            "/base/",
        )
    ):
        return _SUPPORTED_FLUX2_REPO_IDS[1]
    return _SUPPORTED_FLUX2_REPO_IDS[0]


def _validate_supported_flux2_transformer_config(
    transformer_cfg: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for field, expected in _SUPPORTED_FLUX2_TRANSFORMER_CONFIG.items():
        actual = transformer_cfg.get(field)
        if actual != expected:
            raise RuntimeError(
                "Unsupported FLUX.2 transformer config for %s. Only Klein 4B/base-4B is supported. "
                "Field %r expected %r, got %r."
                % (context, field, expected, actual)
            )


def _component_class_name(model_index: Mapping[str, Any], component_name: str) -> str:
    spec = model_index.get(component_name)
    if not isinstance(spec, list) or len(spec) != 2:
        return ""
    class_name = spec[1]
    return class_name.strip() if isinstance(class_name, str) else ""


def _require_component_class(
    model_index: Mapping[str, Any],
    *,
    component_name: str,
    expected_class_name: str,
    context: str,
) -> None:
    actual = _component_class_name(model_index, component_name)
    if actual != expected_class_name:
        raise RuntimeError(
            "Unsupported Qwen Image metadata for %s. Component %r expected class %r, got %r."
            % (context, component_name, expected_class_name, actual or "<missing>")
        )


def _is_qwen_image_model_index(model_index: Mapping[str, Any]) -> bool:
    pipeline_cls = str(model_index.get("_class_name") or "").strip().lower()
    if pipeline_cls in _QWEN_IMAGE_PIPELINE_VARIANTS:
        return True
    return _component_class_name(model_index, "transformer") == "QwenImageTransformer2DModel"


def _qwen_image_variant_from_metadata(
    *,
    model_index: Mapping[str, Any],
    transformer_config: Mapping[str, Any],
    repo_dir: str,
) -> tuple[str, str]:
    pipeline_cls = str(model_index.get("_class_name") or "").strip().lower()
    if pipeline_cls != QWEN_IMAGE_EDIT_PIPELINE_CLASS.lower():
        raise RuntimeError(
            "Unsupported Qwen Image metadata for %s: only pipeline class %r is executable; got %r."
            % (repo_dir, QWEN_IMAGE_EDIT_PIPELINE_CLASS, pipeline_cls or "<missing>")
        )
    if transformer_config.get("zero_cond_t") is not True:
        raise RuntimeError(
            "Unsupported Qwen Image transformer config for %s: Edit-2511 requires zero_cond_t=true."
            % repo_dir
        )
    return _QWEN_IMAGE_PIPELINE_VARIANTS[pipeline_cls]


def _validate_supported_qwen_image_transformer_config(
    transformer_cfg: Mapping[str, Any],
    *,
    variant: str,
    context: str,
) -> None:
    qwen_image_transformer_config_from_mapping(transformer_cfg, variant=variant, context=context)


def _validate_qwen_image_text_encoder_config(text_encoder_cfg: Mapping[str, Any], *, context: str) -> None:
    qwen_image_text_encoder_config_from_mapping(text_encoder_cfg, context=context)


def _validate_qwen_image_vae_config(vae_cfg: Mapping[str, Any], *, context: str) -> None:
    qwen_image_vae_config_from_mapping(vae_cfg, context=context)


def _validate_qwen_image_scheduler_config(scheduler_cfg: Mapping[str, Any], *, context: str) -> None:
    qwen_image_scheduler_config_from_mapping(scheduler_cfg, context=context)


def _qwen_image_bundle_from_diffusers_metadata(
    *,
    repo_dir: str,
    model_index: Mapping[str, Any],
    expected_family: ModelFamily | None,
) -> DiffusionModelBundle:
    if expected_family is not None and expected_family is not ModelFamily.QWEN_IMAGE:
        raise RuntimeError(
            "Diffusers repo %r is Qwen Image metadata, but expected_family=%r."
            % (repo_dir, getattr(expected_family, "value", expected_family))
        )

    context = str(repo_dir)
    _require_component_class(
        model_index,
        component_name="transformer",
        expected_class_name="QwenImageTransformer2DModel",
        context=context,
    )
    _require_component_class(
        model_index,
        component_name="text_encoder",
        expected_class_name="Qwen2_5_VLForConditionalGeneration",
        context=context,
    )
    _require_component_class(
        model_index,
        component_name="vae",
        expected_class_name="AutoencoderKLQwenImage",
        context=context,
    )
    _require_component_class(
        model_index,
        component_name="scheduler",
        expected_class_name="FlowMatchEulerDiscreteScheduler",
        context=context,
    )
    _require_component_class(
        model_index,
        component_name="tokenizer",
        expected_class_name="Qwen2Tokenizer",
        context=context,
    )

    repo_path = Path(repo_dir)
    transformer_cfg = _read_json(repo_path / "transformer" / "config.json")
    text_encoder_cfg = _read_json(repo_path / "text_encoder" / "config.json")
    vae_cfg = _read_json(repo_path / "vae" / "config.json")
    scheduler_cfg = _read_json(repo_path / "scheduler" / "scheduler_config.json")

    _, variant = _qwen_image_variant_from_metadata(
        model_index=model_index,
        transformer_config=transformer_cfg,
        repo_dir=repo_dir,
    )
    _validate_supported_qwen_image_transformer_config(transformer_cfg, variant=variant, context=context)
    _validate_qwen_image_text_encoder_config(text_encoder_cfg, context=context)
    _validate_qwen_image_vae_config(vae_cfg, context=context)
    _validate_qwen_image_scheduler_config(scheduler_cfg, context=context)

    _require_component_class(
        model_index,
        component_name="processor",
        expected_class_name="Qwen2VLProcessor",
        context=context,
    )

    raise RuntimeError(
        "Qwen Image Edit-2511 metadata directory %r is not an executable model. "
        "Select the exact core transformer GGUF and provide the external Qwen2.5-VL GGUF and VAE SafeTensors assets."
        % repo_dir
    )


def _flux2_signature_from_vendored_hf(*, model_path: str) -> ModelSignature:
    suffix = Path(model_path).suffix.lower()
    if suffix not in {".safetensors", ".safetensor", ".gguf"}:
        raise RuntimeError(
            "Unsupported FLUX.2 checkpoint format %r for %s. Only core-only SafeTensors or GGUF checkpoints are supported."
            % (suffix or "<none>", model_path)
        )

    repo_hint = _resolve_flux2_repo_id_from_path(model_path)
    variant = "base" if repo_hint.endswith("base-4B") else "klein"
    is_distilled = variant != "base"
    vendor_root = _BACKEND_ROOT / "huggingface" / "black-forest-labs"
    repo_name = repo_hint.split("/", 1)[1]
    transformer_cfg = _read_json(vendor_root / repo_name / "transformer" / "config.json")
    text_encoder_cfg = _read_json(vendor_root / repo_name / "text_encoder" / "config.json")

    _validate_supported_flux2_transformer_config(transformer_cfg, context=repo_hint)

    text_hidden = int(text_encoder_cfg.get("hidden_size", 0))
    if text_hidden != 2560:
        raise RuntimeError(
            "Unsupported FLUX.2 text encoder config for %s. Expected Qwen3-4B hidden_size=2560, got %r."
            % (repo_hint, text_encoder_cfg.get("hidden_size"))
        )

    latent_channels = int(transformer_cfg["in_channels"])
    context_dim = int(transformer_cfg["joint_attention_dim"])
    double_layers = int(transformer_cfg["num_layers"])
    single_layers = int(transformer_cfg["num_single_layers"])

    quantization = QuantizationHint()
    if suffix == ".gguf":
        quantization = QuantizationHint(kind=QuantizationKind.GGUF, detail="file_extension")

    return ModelSignature(
        family=ModelFamily.FLUX2,
        repo_hint=repo_hint,
        prediction=PredictionKind.FLOW,
        latent_format=LatentFormat.FLUX2,
        quantization=quantization,
        core=CodexCoreSignature(
            architecture=CodexCoreArchitecture.FLOW_TRANSFORMER,
            channels_in=latent_channels,
            channels_out=latent_channels,
            context_dim=context_dim,
            temporal=False,
            depth=double_layers + single_layers,
            key_prefixes=["double_blocks.", "single_blocks."],
        ),
        text_encoders=[
            TextEncoderSignature(
                name="qwen3_4b",
                key_prefix="text_encoder.",
                expected_dim=text_hidden,
                tokenizer_hint=f"{repo_hint}/tokenizer",
            )
        ],
        vae=None,
        extras={
            "core_only": True,
            "flux2_variant": variant,
            "flow_double_layers": double_layers,
            "flow_single_layers": single_layers,
            "guidance_embed": bool(transformer_cfg.get("guidance_embeds", False)),
            "is_distilled": is_distilled,
            "signature_source": "vendored_hf",
            "supported_repo_variants": list(_SUPPORTED_FLUX2_REPO_IDS),
        },
    )


def _ltx2_signature_from_vendored_hf(*, model_path: str) -> ModelSignature:
    suffix = Path(model_path).suffix.lower()
    if suffix not in {".safetensors", ".safetensor", ".gguf"}:
        raise RuntimeError(
            "Unsupported LTX2 checkpoint format %r for %s. Only monolithic SafeTensors or core-only GGUF checkpoints are supported."
            % (suffix or "<none>", model_path)
        )

    vendor_root = _BACKEND_ROOT / "huggingface" / "Lightricks" / "LTX-2"
    transformer_cfg = _read_json(vendor_root / "transformer" / "config.json")

    latent_channels = int(transformer_cfg.get("in_channels", 128))
    context_dim = int(transformer_cfg.get("cross_attention_dim", 4096))
    num_layers = int(transformer_cfg.get("num_layers", 48))

    quantization = QuantizationHint()
    core_only = False
    if suffix == ".gguf":
        quantization = QuantizationHint(kind=QuantizationKind.GGUF, detail="file_extension")
        core_only = True

    return ModelSignature(
        family=ModelFamily.LTX2,
        repo_hint="Lightricks/LTX-2",
        prediction=PredictionKind.FLOW,
        latent_format=LatentFormat.LTX2,
        quantization=quantization,
        core=CodexCoreSignature(
            architecture=CodexCoreArchitecture.DIT,
            channels_in=latent_channels,
            channels_out=latent_channels,
            context_dim=context_dim,
            temporal=True,
            depth=num_layers,
            key_prefixes=["transformer_blocks."],
        ),
        text_encoders=[
            TextEncoderSignature(
                name="gemma3_12b",
                key_prefix="text_encoder.",
                expected_dim=context_dim,
                tokenizer_hint="Lightricks/LTX-2/tokenizer",
            )
        ],
        vae=None if core_only else VAESignature(key_prefix="vae.", latent_channels=latent_channels),
        extras={
            "core_only": core_only,
            "signature_source": "vendored_hf",
        },
    )


def _shape_from_bundle(
    bundle,
    key: str,
    dim: int,
    *,
    default: int | None = None,
) -> int | None:
    shape = bundle.shape(key)
    if not shape:
        return default
    try:
        return int(shape[dim])
    except Exception:
        return default


def _sdxl_expected_signature_from_state_dict(
    *,
    state_dict: Mapping[str, Any],
    expected_family: ModelFamily,
) -> ModelSignature:
    bundle = build_bundle(state_dict)
    channels_in = _shape_from_bundle(bundle, "model.diffusion_model.input_blocks.0.0.weight", 1, default=4) or 4
    channels_out = _shape_from_bundle(bundle, "model.diffusion_model.out.2.weight", 0, default=4) or 4
    depth = count_blocks(bundle.keys, "model.diffusion_model.output_blocks.{}.")
    has_vae = bundle.has_prefix("first_stage_model.") or bundle.has_prefix("vae.")

    if expected_family is ModelFamily.SDXL_REFINER:
        has_text_encoder = bundle.has_prefix("conditioner.embedders.0.")
        core_only = (not has_vae) and (not has_text_encoder)
        extras: Dict[str, object] = {"sdxl_variant": "refiner"}
        if core_only:
            extras["core_only"] = True
        return ModelSignature(
            family=ModelFamily.SDXL_REFINER,
            repo_hint="stabilityai/stable-diffusion-xl-refiner-1.0",
            prediction=PredictionKind.EPSILON,
            latent_format=LatentFormat.SD_XL,
            quantization=QuantizationHint(),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.UNET,
                channels_in=channels_in,
                channels_out=channels_out,
                context_dim=1280,
                temporal=False,
                depth=depth,
                key_prefixes=["model.diffusion_model."],
            ),
            text_encoders=[
                TextEncoderSignature(
                    name="clip_g",
                    key_prefix="conditioner.embedders.0.model.",
                    expected_dim=1280,
                    tokenizer_hint="openclip/ViT-bigG-14",
                ),
            ],
            vae=None if core_only else VAESignature(key_prefix="first_stage_model.", latent_channels=channels_out),
            extras=extras,
        )

    has_clip_l = bundle.has_prefix("conditioner.embedders.0.")
    has_clip_g = bundle.has_prefix("conditioner.embedders.1.")
    core_only = (not has_vae) and (not has_clip_l) and (not has_clip_g)
    extras = {"sdxl_variant": "base"}
    if core_only:
        extras["core_only"] = True
    return ModelSignature(
        family=ModelFamily.SDXL,
        repo_hint="stabilityai/stable-diffusion-xl-base-1.0",
        prediction=PredictionKind.EPSILON,
        latent_format=LatentFormat.SD_XL,
        quantization=QuantizationHint(),
        core=CodexCoreSignature(
            architecture=CodexCoreArchitecture.UNET,
            channels_in=channels_in,
            channels_out=channels_out,
            context_dim=2048,
            temporal=False,
            depth=depth,
            key_prefixes=["model.diffusion_model."],
        ),
        text_encoders=[
            TextEncoderSignature(
                name="clip_l",
                key_prefix="conditioner.embedders.0.transformer.",
                expected_dim=768,
                tokenizer_hint="openai/clip-vit-large-patch14",
            ),
            TextEncoderSignature(
                name="clip_g",
                key_prefix="conditioner.embedders.1.model.",
                expected_dim=1280,
                tokenizer_hint="openclip/ViT-bigG-14",
            ),
        ],
        vae=None if core_only else VAESignature(key_prefix="first_stage_model.", latent_channels=channels_out),
        extras=extras,
    )


def _requires_sdxl_checkpoint_keymap(
    state_dict: Mapping[str, Any],
    *,
    expected_family: ModelFamily | None,
) -> bool:
    if expected_family in {ModelFamily.SDXL, ModelFamily.SDXL_REFINER}:
        return True
    if expected_family is not None:
        return False
    for raw_key in state_dict.keys():
        key = str(raw_key)
        if key.startswith("module."):
            module_inner_key = key[len("module.") :]
            if not module_inner_key.startswith("module."):
                key = module_inner_key
        if key.startswith(
            (
                "model.diffusion_model.",
                "model.diffusion_model.label_emb.0.",
                "model.model.diffusion_model.label_emb.0.",
                "diffusion_model.label_emb.0.",
                "diffusion_model.",
                "model.model.diffusion_model.",
                "conditioner.",
                "model.conditioner.",
                "model.model.conditioner.",
                "first_stage_model.",
                "model.first_stage_model.",
                "model.model.first_stage_model.",
                "model.vae.",
                "model.model.vae.",
                "vae.",
            )
        ):
            return True
    return False


def _maybe_resolve_expected_family_keyspace(
    state_dict: Mapping[str, Any],
    *,
    inspection: CheckpointInspection,
    expected_family: ModelFamily | None,
) -> Mapping[str, Any]:
    if expected_family is None:
        return state_dict

    if inspection.format == "gguf" and expected_family in {ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT}:
        from apps.backend.runtime.state_dict.keymap_flux_transformer import resolve_flux_transformer_keyspace

        return resolve_flux_transformer_keyspace(state_dict).view

    if expected_family is ModelFamily.FLUX2:
        from apps.backend.runtime.state_dict.keymap_flux2_transformer import resolve_flux2_transformer_keyspace

        return resolve_flux2_transformer_keyspace(state_dict).view

    if expected_family is ModelFamily.QWEN_IMAGE:
        if inspection.format != "gguf":
            raise RuntimeError(
                "Qwen Image Edit-2511 requires the exact core transformer GGUF; "
                f"got format={inspection.format!r} for {inspection.path!r}."
            )
        from apps.backend.runtime.state_dict.keymap_qwen_image_transformer import (
            resolve_qwen_image_edit_transformer_keyspace,
        )

        return resolve_qwen_image_edit_transformer_keyspace(state_dict).view

    if inspection.format == "gguf" and expected_family is ModelFamily.ZIMAGE:
        from apps.backend.runtime.state_dict.keymap_zimage_transformer import resolve_zimage_transformer_keyspace

        return resolve_zimage_transformer_keyspace(state_dict).view

    return state_dict


def _parse_checkpoint(
    primary_path: str,
    additional_paths: list[str] | None,
    *,
    expected_family: ModelFamily | None = None,
) -> ParsedCheckpoint:
    primary_inspection = _inspect_checkpoint(primary_path)
    base_state = _load_state_dict(primary_path, inspection=primary_inspection)
    base_state = _maybe_resolve_expected_family_keyspace(
        base_state,
        inspection=primary_inspection,
        expected_family=expected_family,
    )
    if _requires_sdxl_checkpoint_keymap(base_state, expected_family=expected_family):
        from apps.backend.runtime.state_dict.keymap_sdxl_checkpoint import resolve_sdxl_checkpoint_keyspace

        base_state = resolve_sdxl_checkpoint_keyspace(base_state).view
    if expected_family is ModelFamily.ZIMAGE:
        signature = _zimage_signature_from_vendored_hf(model_path=primary_path)
    elif expected_family is ModelFamily.QWEN_IMAGE:
        signature = registry_detect(base_state)
        if signature.family is not ModelFamily.QWEN_IMAGE:
            raise ValueError(
                "Expected Qwen Image Edit-2511 checkpoint, but registry detected "
                f"{signature.family.value!r}."
            )
    elif expected_family is ModelFamily.ZIMAGE_L2P:
        signature = registry_detect(base_state)
        if signature.family is not ModelFamily.ZIMAGE_L2P:
            raise ValueError(
                "Expected Z-Image L2P checkpoint, but registry detected "
                f"{signature.family.value!r}."
            )
    elif expected_family is ModelFamily.FLUX2:
        signature = _flux2_signature_from_vendored_hf(model_path=primary_path)
    elif expected_family is ModelFamily.LTX2:
        signature = _ltx2_signature_from_vendored_hf(model_path=primary_path)
    elif expected_family in {ModelFamily.SDXL, ModelFamily.SDXL_REFINER}:
        signature = _sdxl_expected_signature_from_state_dict(
            state_dict=base_state,
            expected_family=expected_family,
        )
    elif expected_family in {ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT}:
        guidance_key = "guidance_in.in_layer.weight"
        has_guidance = any(
            k in base_state for k in (guidance_key, f"transformer.{guidance_key}", f"model.diffusion_model.{guidance_key}")
        )
        signature = _flux_signature_from_vendored_hf(
            model_path=primary_path,
            expected_family=expected_family,
            has_guidance=has_guidance,
        )
    else:
        signature = registry_detect(base_state)
    config = parse_state_dict(base_state, signature)

    if additional_paths and expected_family is ModelFamily.QWEN_IMAGE:
        raise RuntimeError(
            "Qwen Image Edit-2511 accepts exactly one core transformer GGUF; additional checkpoint files are unsupported."
        )

    if additional_paths:
        replacements: Dict[str, Mapping[str, Any]] = {}
        for extra in additional_paths:
            extra_inspection = _inspect_checkpoint(extra)
            extra_state = _load_state_dict(extra, inspection=extra_inspection)
            extra_state = _maybe_resolve_expected_family_keyspace(
                extra_state,
                inspection=extra_inspection,
                expected_family=expected_family,
            )
            if _requires_sdxl_checkpoint_keymap(extra_state, expected_family=expected_family):
                from apps.backend.runtime.state_dict.keymap_sdxl_checkpoint import resolve_sdxl_checkpoint_keyspace

                extra_state = resolve_sdxl_checkpoint_keyspace(extra_state).view
            if expected_family is ModelFamily.ZIMAGE:
                extra_signature = _zimage_signature_from_vendored_hf(model_path=extra)
            elif expected_family is ModelFamily.QWEN_IMAGE:
                raise RuntimeError(
                    "Qwen Image Edit-2511 accepts exactly one core transformer GGUF; "
                    f"additional checkpoint is unsupported: {extra!r}."
                )
            elif expected_family is ModelFamily.ZIMAGE_L2P:
                extra_signature = registry_detect(extra_state)
                if extra_signature.family is not ModelFamily.ZIMAGE_L2P:
                    raise ValueError(
                        "Expected Z-Image L2P additional checkpoint, but registry detected "
                        f"{extra_signature.family.value!r}."
                    )
            elif expected_family is ModelFamily.FLUX2:
                extra_signature = _flux2_signature_from_vendored_hf(model_path=extra)
            elif expected_family is ModelFamily.LTX2:
                extra_signature = _ltx2_signature_from_vendored_hf(model_path=extra)
            elif expected_family in {ModelFamily.SDXL, ModelFamily.SDXL_REFINER}:
                extra_signature = _sdxl_expected_signature_from_state_dict(
                    state_dict=extra_state,
                    expected_family=expected_family,
                )
            elif expected_family in {ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT}:
                guidance_key = "guidance_in.in_layer.weight"
                has_guidance = any(
                    k in extra_state for k in (guidance_key, f"transformer.{guidance_key}", f"model.diffusion_model.{guidance_key}")
                )
                extra_signature = _flux_signature_from_vendored_hf(
                    model_path=extra,
                    expected_family=expected_family,
                    has_guidance=has_guidance,
                )
            else:
                extra_signature = registry_detect(extra_state)
            extra_config = parse_state_dict(extra_state, extra_signature)
            for name, component in extra_config.components.items():
                replacements[name] = component.state_dict
        if replacements:
            config = config.replace_components(replacements)

    return ParsedCheckpoint(signature=signature, config=config)


def _build_diffusion_bundle(
    *,
    model_ref: str,
    family: ModelFamily,
    estimated_config: Any,
    components: Dict[str, Any],
    signature: Optional[ModelSignature] = None,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> DiffusionModelBundle:
    return DiffusionModelBundle(
        model_ref=model_ref,
        family=family,
        estimated_config=estimated_config,
        components=dict(components),
        signature=signature,
        source=source,
        metadata=dict(metadata or {}),
    )


def _load_component_config(component_path: str) -> Dict[str, Any]:
    config_file = os.path.join(component_path, "config.json")
    if os.path.isfile(config_file):
        with open(config_file, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _resolve_vae_class(
    signature: ModelSignature | None,
    *,
    layout: str = "diffusers",
    lane: VaeLayoutLane | None = None,
):
    """Select the appropriate VAE class.

    - WAN22 families use the shared native LDM `AutoencoderKL_LDM` lane.
    - Other families choose the runtime lane class:
      - LDM lane: `AutoencoderKL_LDM`
      - Diffusers lane: `diffusers.AutoencoderKL`
    """

    family = getattr(signature, "family", None)
    if family in _WAN22_FAMILIES:
        if lane is not None and lane is not VaeLayoutLane.LDM_NATIVE:
            raise RuntimeError(
                "WAN22 variants support only native LDM VAE lane; "
                f"resolved lane={getattr(lane, 'value', lane)!r}."
            )
        if layout != "ldm":
            raise RuntimeError(
                "WAN22 native VAE lane requires LDM keyspace layout; "
                f"detected layout={layout!r}."
            )
        return AutoencoderKL_LDM
    if lane is not None and uses_ldm_native_lane(lane):
        if layout != "ldm":
            raise RuntimeError(
                "Native LDM VAE lane requires LDM keyspace layout; "
                f"detected layout={layout!r} for family={getattr(family, 'value', family)!r}."
            )
        return AutoencoderKL_LDM
    if family is ModelFamily.FLUX2:
        from diffusers import AutoencoderKLFlux2

        return AutoencoderKLFlux2
    from diffusers import AutoencoderKL

    return AutoencoderKL


def _detect_sdxl_vae_projection_lane(
    state_dict: Mapping[str, Any],
    signature: Optional[ModelSignature],
) -> str:
    family = getattr(signature, "family", None) if signature is not None else None
    if family not in (ModelFamily.SDXL, ModelFamily.SDXL_REFINER):
        return "linear_2d"

    observed: set[str] = set()
    shape_getter = getattr(state_dict, "shape_of", None)
    for key in _SDXL_VAE_CANONICAL_PROJECTION_KEYS:
        if key not in state_dict:
            continue
        shape = None
        ndim = None
        if callable(shape_getter):
            try:
                raw_shape = shape_getter(key)
            except Exception:
                raw_shape = None
            if raw_shape is not None:
                shape = tuple(int(value) for value in raw_shape)
                ndim = len(shape)
        if shape is None:
            tensor = state_dict[key]
            ndim = getattr(tensor, "ndim", None)
            shape = getattr(tensor, "shape", None)
        if ndim == 2 and shape and len(shape) == 2:
            observed.add("linear_2d")
            continue
        if ndim == 4 and shape and len(shape) == 4 and tuple(shape[-2:]) == (1, 1):
            observed.add("conv1x1_4d")
            continue
        raise KeyMappingError(
            "SDXL VAE projection lane mismatch after keyspace resolution. "
            f"key={key!r} ndim={ndim} shape={tuple(shape) if shape is not None else None}"
        )

    if not observed:
        return "linear_2d"
    if len(observed) != 1:
        raise KeyMappingError(
            "SDXL VAE projection lane is mixed across canonical keys. "
            f"lanes={sorted(observed)}"
        )
    return next(iter(observed))


def _apply_sdxl_vae_conv_projection_lane(model: Any) -> None:
    def _replace(module: Any, *, key: str) -> _Conv1x1Projection:
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        weight = getattr(module, "weight", None)
        if not isinstance(in_features, int) or not isinstance(out_features, int) or weight is None or getattr(weight, "ndim", None) != 2:
            raise RuntimeError(
                "SDXL VAE native 4D lane expects linear-like projection modules before replacement. "
                f"key={key} got={type(module).__name__}"
            )
        replacement = _Conv1x1Projection(
            in_features=in_features,
            out_features=out_features,
            bias=getattr(module, "bias", None) is not None,
        )
        replacement = replacement.to(device=weight.device, dtype=weight.dtype)
        replacement.parameters_manual_cast = bool(getattr(module, "parameters_manual_cast", False))
        return replacement

    for branch in ("encoder", "decoder"):
        side = getattr(model, branch, None)
        mid_block = getattr(side, "mid_block", None) if side is not None else None
        attentions = getattr(mid_block, "attentions", None) if mid_block is not None else None
        if attentions is None or len(attentions) == 0:
            raise RuntimeError(f"SDXL VAE native 4D lane missing {branch}.mid_block.attentions[0].")
        attn = attentions[0]

        attn.to_q = _replace(attn.to_q, key=f"{branch}.mid_block.attentions.0.to_q")
        attn.to_k = _replace(attn.to_k, key=f"{branch}.mid_block.attentions.0.to_k")
        attn.to_v = _replace(attn.to_v, key=f"{branch}.mid_block.attentions.0.to_v")

        to_out = getattr(attn, "to_out", None)
        if to_out is None or len(to_out) == 0:
            raise RuntimeError(f"SDXL VAE native 4D lane missing {branch}.mid_block.attentions.0.to_out[0].")
        to_out[0] = _replace(to_out[0], key=f"{branch}.mid_block.attentions.0.to_out.0")


def _maybe_convert_sdxl_vae_state_dict(
    state_dict: Mapping[str, Any],
    signature: Optional[ModelSignature],
) -> Mapping[str, Any]:
    """Normalize LDM-style Flow/SDXL VAE weights into diffusers `AutoencoderKL` keyspace.

    This is a thin wrapper around the strict, string-only keymap in
    `apps/backend/runtime/state_dict/keymap_sdxl_vae.py`. It:
    - Rejects wrapper-prefix rewrite attempts (`first_stage_model.`, `vae.`, `model.`, `module.`) instead of normalizing them away.
    - Detects diffusers vs. LDM-style SDXL layouts and resolves LDM keys into diffusers lookup space.
    - Validates canonical mid-attention projection lanes (`linear_2d` or `conv1x1_4d`) using lazy shape metadata first.
    - Drops known non-weight training metadata (`model_ema.decay`, `model_ema.num_updates`).

    Unknown/ambiguous layouts raise (fail loud). Families outside the Flow/SDXL lane are returned
    unchanged.
    """
    family = getattr(signature, "family", None) if signature is not None else None
    if family not in (
        ModelFamily.SDXL,
        ModelFamily.SDXL_REFINER,
        ModelFamily.FLUX,
        ModelFamily.FLUX_KONTEXT,
        ModelFamily.FLUX2,
        ModelFamily.ZIMAGE,
    ):
        return state_dict

    from apps.backend.runtime.state_dict.keymap_sdxl_vae import resolve_sdxl_vae_keyspace

    resolved_state_dict = resolve_sdxl_vae_keyspace(state_dict).view  # fail loud on unknown/ambiguous layouts

    # Preflight the canonical mid-attention projections so lane/shape contract errors
    # surface explicitly before safe-load accounting can collapse them into generic
    # missing-key noise.
    for key in _SDXL_VAE_CANONICAL_PROJECTION_KEYS:
        if key not in resolved_state_dict:
            continue
        _ = resolved_state_dict[key]
    return resolved_state_dict


def _detect_vae_layout(sd: Mapping[str, Any]) -> str:
    """Detect VAE keyspace layout using shared fail-loud sentinels."""
    return detect_vae_layout(sd)


def _assert_vae_state_dict_keyspace_for_family(
    state_dict: Mapping[str, Any],
    *,
    weights_path: str | None,
    family: ModelFamily | None,
) -> None:
    """Fail loud when a Flow-family VAE load receives a non-VAE state_dict."""
    key_prefixes = ("encoder.", "decoder.", "module.encoder.", "module.decoder.")
    sample_keys: list[str] = []
    for raw_key in state_dict.keys():
        if not isinstance(raw_key, str):
            continue
        if len(sample_keys) < 8:
            sample_keys.append(raw_key)
        if raw_key.startswith(key_prefixes):
            return

    origin = str(weights_path).strip() if isinstance(weights_path, str) and weights_path.strip() else "<unknown>"
    family_label = getattr(family, "value", None) or "flow-family"
    raise RuntimeError(
        f"{family_label} VAE rejected non-VAE asset keyspace at "
        f"{origin}. Expected AutoencoderKL keys under 'encoder.'/'decoder.'; "
        "this usually means extras.vae_sha resolved to a non-VAE asset (for example the core checkpoint). "
        f"Sample keys: {sample_keys}"
    )


def _assert_core_only_vae_path_not_checkpoint(*, model_ref: str, vae_path: str, family_label: str) -> None:
    """Reject configurations where a core-only checkpoint and VAE path are the same file."""
    model_realpath = os.path.realpath(os.path.expanduser(model_ref.strip()))
    vae_realpath = os.path.realpath(os.path.expanduser(vae_path.strip()))
    if model_realpath == vae_realpath:
        raise RuntimeError(
            f"{family_label} core-only VAE path resolves to the same file as the core checkpoint. "
            "Select a SHA from inventory.vaes (do not reuse the model checkpoint SHA in extras.vae_sha). "
            f"path={vae_realpath}"
        )


def _safetensors_primary_dtype_hint(weights_path: str | None) -> str | None:
    if not weights_path:
        return None
    try:
        return detect_safetensors_primary_dtype(Path(weights_path))
    except Exception:
        return None


def _log_weights_dtype_hint(*, role: DeviceRole, selected: torch.dtype, hint: str | None) -> None:
    try:
        from apps.backend.runtime.diagnostics import pipeline_debug as _pipeline_debug
    except Exception:
        return
    if not hint:
        return

    hint_map: dict[str, object] = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }
    # Float8 dtypes may not exist on all torch builds.
    fp8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    fp8_e5m2 = getattr(torch, "float8_e5m2", None)
    if fp8_e4m3fn is not None:
        hint_map["fp8_e4m3fn"] = fp8_e4m3fn
    if fp8_e5m2 is not None:
        hint_map["fp8_e5m2"] = fp8_e5m2

    expected = hint_map.get(hint)
    if expected is not None and expected == selected:
        return

    _pipeline_debug.log(f"[dtype] role={role.value} selected={selected} weights_primary={hint}")


def _torch_dtype_from_weights_primary_hint(hint: str | None) -> torch.dtype | None:
    if not hint:
        return None
    hint_map: dict[str, torch.dtype] = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }
    # Float8 dtypes may not exist on all torch builds.
    fp8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    fp8_e5m2 = getattr(torch, "float8_e5m2", None)
    if isinstance(fp8_e4m3fn, torch.dtype):
        hint_map["fp8_e4m3fn"] = fp8_e4m3fn
    if isinstance(fp8_e5m2, torch.dtype):
        hint_map["fp8_e5m2"] = fp8_e5m2
    return hint_map.get(hint)


def _native_weights_storage_dtype(weights_path: str | None, state_dict: Mapping[str, object] | None) -> torch.dtype | None:
    """Resolve the native (dominant) dtype for the weights source.

    - For SafeTensors: uses the header-only primary dtype hint (majority bytes).
    - For other formats: best-effort scan of state_dict values (first floating tensor dtype).
    """

    native = _torch_dtype_from_weights_primary_hint(_safetensors_primary_dtype_hint(weights_path))
    if native is not None:
        return native
    if not state_dict:
        return None

    materialize = getattr(state_dict, "materialize", None)
    if callable(materialize):
        for value in state_dict.values():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                return value.dtype
        return None

    for idx, value in enumerate(state_dict.values()):
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            return value.dtype
        if idx >= 4096:
            break
    return None


def _load_huggingface_component(
    parsed: ParsedCheckpoint,
    component_name: str,
    lib_name: str,
    cls_name: str,
    repo_path: str,
    state_dict: Mapping[str, Any] | None,
    *,
    weights_path: str | None = None,
):
    family = parsed.signature.family
    config = parsed.config
    component_path = os.path.join(repo_path, component_name)

    if component_name in {"feature_extractor", "safety_checker"}:
        return None

    if lib_name in {"transformers", "diffusers"} and component_name == "scheduler":
        cls = getattr(importlib.import_module(lib_name), cls_name)
        _trace.event("component_from_pretrained", name=component_name, lib=lib_name, cls=cls_name)
        return cls.from_pretrained(os.path.join(repo_path, component_name))

    if lib_name in {"transformers", "diffusers"} and component_name.startswith("tokenizer"):
        cls = getattr(importlib.import_module(lib_name), cls_name)
        _trace.event("component_from_pretrained", name=component_name, lib=lib_name, cls=cls_name)
        tokenizer = cls.from_pretrained(os.path.join(repo_path, component_name))
        if hasattr(tokenizer, "_eventual_warn_about_too_long_sequence"):
            tokenizer._eventual_warn_about_too_long_sequence = lambda *_, **__: None
        return tokenizer

    if cls_name in {"AutoencoderKL", "AutoencoderKLFlux2"}:
        if state_dict is None:
            # For SDXL (and refiner) a VAE is mandatory; fail fast instead of
            # attempting to proceed without it.
            if family in (ModelFamily.SDXL, ModelFamily.SDXL_REFINER):
                raise RuntimeError(
                    "No VAE detected in checkpoint for SDXL. Provide a VAE override (vae_path) "
                    "or use a checkpoint with an embedded SDXL VAE."
                )
            # Flux GGUF core-only checkpoints carry only the rectified-flow backbone;
            # they must be composed with an explicit external VAE (sha-selected).
            signature = getattr(parsed, "signature", None)
            quant = getattr(signature, "quantization", None)
            extras = getattr(signature, "extras", {}) or {}
            is_flux_core_gguf = (
                isinstance(signature, ModelSignature)
                and signature.family in (ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT)
                and getattr(quant, "kind", None) is QuantizationKind.GGUF
                and bool(extras.get("gguf_core_only"))
            )
            is_flux2_core_only = (
                isinstance(signature, ModelSignature)
                and signature.family is ModelFamily.FLUX2
                and bool(extras.get("core_only"))
                and signature.vae is None
            )
            if is_flux_core_gguf:
                raise RuntimeError(
                    "Flux GGUF core-only checkpoint is missing a VAE. "
                    "Provide one explicitly (request extras.vae_sha), so the API passes a valid vae_path to the loader."
                )
            if is_flux2_core_only:
                raise RuntimeError(
                    "FLUX.2 core-only checkpoint is missing a VAE. "
                    "Provide one explicitly (request extras.vae_sha), so the API passes a valid vae_path to the loader."
                )
            return None

        # Unwrap common packing shapes (e.g., {'state_dict': {...}})
        if isinstance(state_dict, Mapping) and len(state_dict) == 1 and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if not isinstance(state_dict, Mapping):
            raise RuntimeError(
                f"VAE state_dict must be a mapping; got {type(state_dict).__name__}. "
                "Checkpoint may be malformed or require manual VAE extraction."
            )

        LOGGER.debug(
            "VAE state_dict type=%s len=%d sample_keys=%s",
            type(state_dict).__name__,
            len(state_dict.keys()) if hasattr(state_dict, "keys") else -1,
            list(state_dict.keys())[:5] if hasattr(state_dict, "keys") else None,
        )

        state_dict = validate_vae_key_names(state_dict)
        vae_layout = _detect_vae_layout(state_dict)
        signature = getattr(parsed, "signature", None)
        vae_lane = resolve_vae_layout_lane(family=getattr(signature, "family", None), layout=vae_layout)
        using_ldm_native = uses_ldm_native_lane(vae_lane)

        # Diffusers lane expects canonical diffusers keyspace.
        if not using_ldm_native:
            state_dict = _maybe_convert_sdxl_vae_state_dict(state_dict, signature)

        if family in (ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT, ModelFamily.FLUX2):
            _assert_vae_state_dict_keyspace_for_family(state_dict, weights_path=weights_path, family=family)

        if using_ldm_native:
            vae_projection_lane = "linear_2d"
        else:
            vae_projection_lane = _detect_sdxl_vae_projection_lane(state_dict, signature)
        LOGGER.debug(
            "VAE layout detected=%s lane=%s projection_lane=%s",
            vae_layout,
            vae_lane.value,
            vae_projection_lane,
        )

        vae_cls = _resolve_vae_class(signature, layout=vae_layout, lane=vae_lane)
        try:
            config_json = vae_cls.load_config(component_path)
        except Exception:
            config_json = _load_component_config(component_path)
        if vae_cls is AutoencoderKL_LDM:
            config_json = sanitize_ldm_vae_config(config_json)
        vae_device = memory_management.manager.get_device(DeviceRole.VAE)
        vae_dtype = memory_management.manager.dtype_for_role(
            DeviceRole.VAE,
            native_dtype=_native_weights_storage_dtype(weights_path, state_dict),
        )
        _log_weights_dtype_hint(
            role=DeviceRole.VAE,
            selected=vae_dtype,
            hint=_safetensors_primary_dtype_hint(weights_path),
        )
        _trace.event("vae_construct", device=str(vae_device), dtype=str(vae_dtype), cls=vae_cls.__name__)

        with using_codex_operations(device=vae_device, dtype=vae_dtype, manual_cast_enabled=True):
            model = vae_cls.from_config(config_json)
        if vae_projection_lane == "conv1x1_4d":
            _apply_sdxl_vae_conv_projection_lane(model)

        _trace.event("load_state_dict", module="vae", tensors=len(state_dict))
        from .state_dict import safe_load_state_dict as _safe_load
        expected_total = len(model.state_dict())
        family_name = getattr(getattr(parsed, "signature", None), "family", "unknown")

        try:
            missing, unexpected = _safe_load(model, state_dict, log_name="VAE")
        except Exception as exc:  # no silent fallbacks for VAE
            raise RuntimeError(
                f"Failed to load VAE weights for family {family_name}: {exc!s}"
            ) from exc

        if missing:
            sample = missing[:10]
            LOGGER.error(
                "VAE load failed: missing %d/%d keys for family=%s sample=%s",
                len(missing),
                expected_total,
                family_name,
                sample,
            )
            if family in (ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT):
                guidance = (
                    "Provided VAE weights are incompatible with Flux AutoencoderKL. "
                    "Verify extras.vae_sha resolves to inventory.vaes and does not point to the core GGUF checkpoint."
                )
            elif family is ModelFamily.FLUX2:
                guidance = (
                    "Provided VAE weights are incompatible with FLUX.2 AutoencoderKLFlux2. "
                    "Verify extras.vae_sha resolves to inventory.vaes and points to a FLUX.2 VAE weights file."
                )
            else:
                guidance = (
                    "Provided VAE weights are incompatible with the expected keyspace for this model family. "
                    "The required VAE keys do not match after normalization."
                )
            raise RuntimeError(
                "VAE state_dict missing %d/%d keys for family %s. "
                "%s "
                "Sample missing keys: %s"
                % (len(missing), expected_total, family_name, guidance, sample)
            )

        if unexpected:
            sample = unexpected[:10]
            if family in (ModelFamily.SDXL, ModelFamily.SDXL_REFINER):
                raise RuntimeError(
                    "VAE state_dict has unexpected keys for SDXL. "
                    "This indicates a keymap/conversion mismatch; refusing to continue. "
                    f"unexpected_count={len(unexpected)} sample={sample}"
                )
            LOGGER.warning("VAE load: unexpected %d keys (sample=%s)", len(unexpected), sample)
        return model

    if cls_name == "Qwen3ForCausalLM":
        te_dtype = memory_management.manager.dtype_for_role(
            DeviceRole.TEXT_ENCODER,
            native_dtype=_native_weights_storage_dtype(weights_path, state_dict),
        )
        _log_weights_dtype_hint(
            role=DeviceRole.TEXT_ENCODER,
            selected=te_dtype,
            hint=_safetensors_primary_dtype_hint(weights_path),
        )

        tokenizer_path = os.path.join(repo_path, "tokenizer")
        tokenizer_hint = tokenizer_path if os.path.isdir(tokenizer_path) else None

        if family is ModelFamily.FLUX2:
            from apps.backend.runtime.families.flux2 import Flux2TextEncoder
            from transformers import Qwen3ForCausalLM as HfQwen3ForCausalLM

            if state_dict is None:
                _trace.event("component_from_pretrained", name=component_name, lib=lib_name, cls=cls_name)
                model = HfQwen3ForCausalLM.from_pretrained(
                    component_path,
                    torch_dtype=te_dtype,
                    local_files_only=True,
                )
                wrapped = Flux2TextEncoder.from_pretrained_model(model)
                if tokenizer_hint is not None:
                    wrapped.set_tokenizer_path_hint(tokenizer_hint)
                return wrapped

            resolved_weights_path = str(weights_path).strip() if isinstance(weights_path, str) and weights_path.strip() else None
            if resolved_weights_path and resolved_weights_path.lower().endswith(".gguf"):
                wrapped = Flux2TextEncoder.from_gguf(resolved_weights_path, torch_dtype=te_dtype)
            else:
                if not isinstance(state_dict, Mapping):
                    raise RuntimeError(
                        f"Qwen3 text encoder state_dict must be a mapping; got {type(state_dict).__name__}."
                    )
                wrapped = Flux2TextEncoder.from_state_dict(state_dict, torch_dtype=te_dtype)
            if tokenizer_hint is not None:
                wrapped.set_tokenizer_path_hint(tokenizer_hint)
            return wrapped

        if state_dict is None:
            return None
        from apps.backend.runtime.families.zimage.text_encoder import ZImageTextEncoder

        resolved_weights_path = str(weights_path).strip() if isinstance(weights_path, str) and weights_path.strip() else None
        if resolved_weights_path and resolved_weights_path.lower().endswith(".gguf"):
            model = ZImageTextEncoder.from_gguf(resolved_weights_path, torch_dtype=te_dtype)
        else:
            if not isinstance(state_dict, Mapping):
                raise RuntimeError(
                    f"Qwen3 text encoder state_dict must be a mapping; got {type(state_dict).__name__}."
                )
            model = ZImageTextEncoder.from_state_dict(state_dict, torch_dtype=te_dtype)

        if tokenizer_hint is not None:
            model.set_tokenizer_path_hint(tokenizer_hint)
        return model

    if cls_name in {"CLIPTextModel", "CLIPTextModelWithProjection"}:
        if state_dict is None:
            return None
        
        # Detect T5 state dict keys - if found, load as T5 instead of CLIP
        # This handles GGUF models that may have T5 bundled as text_encoder
        _T5_KEY_PATTERNS = ("encoder.block.", "encoder.final_layer_norm.", "shared.weight")
        _is_actually_t5 = any(
            any(pattern in k for pattern in _T5_KEY_PATTERNS)
            for k in list(state_dict.keys())[:100]
        )
        if _is_actually_t5:
            if family in {ModelFamily.SDXL, ModelFamily.SDXL_REFINER}:
                raise ValueError(
                    f"SDXL slot '{component_name}' requires CLIP weights; received T5-style state dict instead."
                )
            # Load as T5 using T5 handler but keep in same slot
            # The spec.py will detect the correct type later
            _LOG.info(
                "Detected T5 state dict in %s (expected CLIP); loading as T5 model in same slot",
                component_name
            )
            # Use T5EncoderModel handler but keep this component_name (not redirecting)
            return _load_huggingface_component(
                parsed, component_name, lib_name, "T5EncoderModel", repo_path, state_dict, weights_path=weights_path
            )
        # Build native Codex CLIP instead of HF; normalise state dict beforehand
        strict_sdxl = family in (ModelFamily.SDXL, ModelFamily.SDXL_REFINER)
        te_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
        te_dtype = memory_management.manager.dtype_for_role(
            DeviceRole.TEXT_ENCODER,
            native_dtype=_native_weights_storage_dtype(weights_path, state_dict),
        )
        _log_weights_dtype_hint(
            role=DeviceRole.TEXT_ENCODER,
            selected=te_dtype,
            hint=_safetensors_primary_dtype_hint(weights_path),
        )
        to_args = dict(device=te_device, dtype=te_dtype)
        add_proj = component_name in {"text_encoder_2", "text_encoder_3"}
        from .state_dict import safe_load_state_dict
        from apps.backend.infra.config.sdxl_te_qkv_impl import SdxlTeQkvImpl, read_sdxl_te_qkv_impl
        from apps.backend.runtime.common.nn.clip_text_cx import (
            CodexCLIPTextConfig,
            CodexCLIPTextModel,
            CodexCLIPTextModelFusedQKV,
        )
        from apps.backend.runtime.models.clip_key_normalization import normalize_codex_clip_state_dict_with_layout

        # CLIP-ViT-Large-patch14 default config (used by Flux CLIP-L and SD1.x/2.x)
        _CLIP_L_DEFAULT_CONFIG = {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "hidden_act": "quick_gelu",
            "max_position_embeddings": 77,
            "layer_norm_eps": 1e-05,
            "vocab_size": 49408,
            "projection_dim": 768,
        }

        # OpenCLIP-ViT-bigG config (used by SDXL text_encoder_2)
        _CLIP_G_DEFAULT_CONFIG = {
            "hidden_size": 1280,
            "intermediate_size": 5120,
            "num_hidden_layers": 32,
            "num_attention_heads": 20,
            "hidden_act": "gelu",
            "max_position_embeddings": 77,
            "layer_norm_eps": 1e-05,
            "vocab_size": 49408,
            "projection_dim": 1280,
        }

        # Try to infer the correct config from state_dict shapes
        def _infer_clip_config_from_state(sd: Mapping[str, Any]) -> dict | None:
            """Infer CLIP config variant from state_dict tensor shapes."""
            # Check embedding weight shape to determine hidden_size
            embedding_keys = [
                "transformer.text_model.embeddings.token_embedding.weight",
                "text_model.embeddings.token_embedding.weight",
            ]
            for key in embedding_keys:
                if key in sd:
                    t = sd[key]
                    if hasattr(t, "shape"):
                        hidden_size = t.shape[1] if len(t.shape) > 1 else None
                        if hidden_size == 1280:
                            _LOG.info("Detected OpenCLIP-G variant (hidden_size=1280)")
                            return _CLIP_G_DEFAULT_CONFIG
                        elif hidden_size == 768:
                            _LOG.info("Detected CLIP-L variant (hidden_size=768)")
                            return _CLIP_L_DEFAULT_CONFIG
            return None

        if strict_sdxl:
            if family is ModelFamily.SDXL:
                if component_name == "text_encoder":
                    config_json = _CLIP_L_DEFAULT_CONFIG
                elif component_name == "text_encoder_2":
                    config_json = _CLIP_G_DEFAULT_CONFIG
                else:
                    raise RuntimeError(f"SDXL: unexpected CLIP component name: {component_name!r}")
            elif family is ModelFamily.SDXL_REFINER:
                if component_name == "text_encoder":
                    add_proj = True
                    config_json = _CLIP_G_DEFAULT_CONFIG
                else:
                    raise RuntimeError(f"SDXL refiner: unexpected CLIP component name: {component_name!r}")
            else:
                raise RuntimeError(f"Unexpected strict_sdxl family: {family!r}")
        else:
            # Determine if we should use default CLIP config
            use_default_config = False
            inferred_config = None

            # Heuristic: component_name text_encoder_2 is commonly T5 for Flux; if CLIP ends up here,
            # infer config from the state dict so we don't guess wrong.
            if component_name == "text_encoder_2":
                inferred_config = _infer_clip_config_from_state(state_dict)
                if inferred_config:
                    _LOG.info("CLIP config inferred for %s", component_name)
                    use_default_config = True

            if use_default_config:
                config_json = inferred_config or _CLIP_L_DEFAULT_CONFIG
            else:
                try:
                    config_json = read_arbitrary_config(component_path)
                    # Validate it's actually a CLIP config
                    if "hidden_size" not in config_json:
                        _LOG.warning(
                            "Config at %s missing 'hidden_size' (got %s); using CLIP-L defaults",
                            component_path, list(config_json.keys())[:5]
                        )
                        config_json = _CLIP_L_DEFAULT_CONFIG
                except FileNotFoundError:
                    _LOG.info(
                        "No config.json found at %s; using default CLIP-L configuration",
                        component_path
                    )
                    config_json = _CLIP_L_DEFAULT_CONFIG

        cfg = CodexCLIPTextConfig.from_dict(config_json)

        layout_sha: str | None = None
        cached_layout: ClipLayoutMetadata | None = None
        layout_key = _clip_layout_cache_key(component_name)
        if isinstance(weights_path, str) and weights_path.strip():
            layout_sha, _ = model_api.hash_for_file(weights_path)
            if layout_sha:
                cached_layout = _clip_layout_metadata_from_cache(
                    model_api.get_layout_metadata(layout_sha, layout_key)
                )

        clip_model_cls = CodexCLIPTextModel
        clip_layout = ClipLayoutMetadata(qkv_layout="split", projection_orientation="none", source_style=None)
        persist_layout_metadata = True
        if family in (ModelFamily.SDXL, ModelFamily.SDXL_REFINER) and component_name in {"text_encoder", "text_encoder_2"}:
            from apps.backend.runtime.state_dict.keymap_sdxl_clip import (
                clip_layout_metadata_from_resolved,
                resolve_sdxl_clip_g_keyspace_with_layout,
                resolve_sdxl_clip_l_keyspace_with_layout,
            )

            requested_qkv = read_sdxl_te_qkv_impl()
            requested_impl = requested_qkv.value
            persist_layout_metadata = requested_qkv is SdxlTeQkvImpl.AUTO
            layout_hint = _clip_layout_hint_from_cache(
                cached_layout,
                allow_cache=persist_layout_metadata,
            )
            if family is ModelFamily.SDXL and component_name == "text_encoder":
                resolved_clip = resolve_sdxl_clip_l_keyspace_with_layout(
                    state_dict,
                    qkv_impl=requested_impl,
                    layout_metadata=layout_hint,
                )
            else:
                resolved_clip = resolve_sdxl_clip_g_keyspace_with_layout(
                    state_dict,
                    qkv_impl=requested_impl,
                    projection_orientation="auto",
                    layout_metadata=layout_hint,
                )
            clip_layout = clip_layout_metadata_from_resolved(resolved_clip)
            state_dict = resolved_clip.view
        else:
            state_dict, clip_layout = normalize_codex_clip_state_dict_with_layout(
                state_dict,
                num_layers=cfg.num_hidden_layers,
                keep_projection=add_proj,
                qkv_impl="auto",
                projection_orientation="auto",
                layout_metadata=cached_layout,
                require_projection=False,
            )

        clip_model_cls = CodexCLIPTextModelFusedQKV if clip_layout.qkv_layout == "fused" else CodexCLIPTextModel
        if layout_sha and persist_layout_metadata:
            model_api.set_layout_metadata(
                layout_sha,
                layout_key,
                _clip_layout_metadata_to_cache(clip_layout),
            )

        projection_layout = _projection_module_layout(add_proj, clip_layout)
        with using_codex_operations(**to_args, manual_cast_enabled=True):
            model = IntegratedCLIP(
                clip_model_cls,
                cfg,
                add_text_projection=add_proj,
                text_projection_layout=projection_layout,
            ).to(**to_args)

        # Compute dtype is distinct from storage dtype. Keep weights in `te_dtype`,
        # but allow activations to run in higher precision when configured.
        model.transformer.compute_dtype = memory_management.manager.compute_dtype_for_role(
            DeviceRole.TEXT_ENCODER,
            storage_dtype=te_dtype,
        )

        missing, unexpected = safe_load_state_dict(model, state_dict, log_name=cls_name)
        if missing or unexpected:
            if strict_sdxl:
                raise RuntimeError(
                    "SDXL CLIP load failed (strict): missing/unexpected keys detected. "
                    f"component={component_name} class={cls_name} missing={len(missing)} unexpected={len(unexpected)} "
                    f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
                )
            if missing:
                CLIP_LOG.warning("CLIP missing (%s): %s", component_name, missing[:10])
            if unexpected:
                CLIP_LOG.debug("CLIP unexpected (%s): %s", component_name, unexpected[:10])
        return model

    if cls_name == "T5EncoderModel":
        if state_dict is None:
            return None
        
        # Detect CLIP state dict keys - if found, load as CLIP instead of T5
        # This handles checkpoints that may have CLIP bundled as text_encoder_2
        _CLIP_KEY_PATTERNS = ("text_model.embeddings.", "text_model.encoder.layers.", "logit_scale")
        _is_actually_clip = any(
            any(pattern in k for pattern in _CLIP_KEY_PATTERNS)
            for k in list(state_dict.keys())[:100]
        )
        if _is_actually_clip:
            if family in {ModelFamily.SDXL, ModelFamily.SDXL_REFINER}:
                raise ValueError(
                    f"SDXL slot '{component_name}' must stay CLIP-only; received CLIP weights in a non-CLIP slot."
                )
            # Load as CLIP using CLIP handler but keep in same slot
            # The spec.py will detect the correct type later
            _LOG.info(
                "Detected CLIP state dict in %s (expected T5); loading as CLIP model in same slot",
                component_name
            )
            # Use CLIPTextModel handler but keep this component_name (not redirecting)
            return _load_huggingface_component(
                parsed, component_name, lib_name, "CLIPTextModel", repo_path, state_dict, weights_path=weights_path
            )
        # T5-XXL config (google/t5-v1_1-xxl) - used by Flux
        _T5_XXL_DEFAULT_CONFIG = {
            "d_ff": 10240,
            "d_kv": 64,
            "d_model": 4096,
            "dense_act_fn": "gelu_new",
            "is_gated_act": True,
            "model_type": "t5",
            "num_heads": 64,
            "num_layers": 24,
            "vocab_size": 32128,
        }
        
        # Determine if we should use default T5 config:
        # 1. T5 loaded in CLIP slot via redirect (component_path has CLIP config)
        # 2. No config.json exists
        # 3. Config exists but is wrong type (e.g., CLIP config with num_hidden_layers)
        use_default_config = False
        
        # Case 1: T5 in CLIP slot (text_encoder is normally CLIP for Flux)
        if component_name == "text_encoder":
            _LOG.info("T5 loading in CLIP slot (%s); using T5-XXL default config", component_name)
            use_default_config = True
        
        if use_default_config:
            t5_config = _T5_XXL_DEFAULT_CONFIG
        else:
            try:
                t5_config = read_arbitrary_config(component_path)
                # Validate it's actually a T5 config
                if "num_layers" not in t5_config:
                    _LOG.warning(
                        "Config at %s missing 'num_layers' (got %s); using T5-XXL defaults",
                        component_path, list(t5_config.keys())[:5]
                    )
                    t5_config = _T5_XXL_DEFAULT_CONFIG
            except FileNotFoundError:
                _LOG.info(
                    "No config.json found at %s; using default T5-XXL configuration",
                    component_path
                )
                t5_config = _T5_XXL_DEFAULT_CONFIG
        te_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
        storage_dtype = memory_management.manager.dtype_for_role(
            DeviceRole.TEXT_ENCODER,
            native_dtype=_native_weights_storage_dtype(weights_path, state_dict),
        )
        _log_weights_dtype_hint(
            role=DeviceRole.TEXT_ENCODER,
            selected=storage_dtype,
            hint=_safetensors_primary_dtype_hint(weights_path),
        )
        state_dict_dtype = detect_state_dict_dtype(state_dict)
        if state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "gguf"]:
            LOGGER.info("Using Detected T5 Data Type: %s", state_dict_dtype)
            storage_dtype = state_dict_dtype
            if state_dict_dtype == "gguf":
                LOGGER.info("Using pre-quant state dict!")
                beautiful_print_gguf_state_dict_statics(state_dict)
        else:
            LOGGER.info("Using Default T5 Data Type: %s", storage_dtype)

        te_load_device = te_device

        from transformers import modeling_utils

        if storage_dtype == "gguf":
            with modeling_utils.no_init_weights():
                with using_codex_operations(
                    device=te_load_device,
                    dtype=memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER),
                    manual_cast_enabled=False,
                    weight_format="gguf",
                ):
                    model = IntegratedT5(t5_config)
        else:
            with modeling_utils.no_init_weights():
                with using_codex_operations(device=te_load_device, dtype=storage_dtype, manual_cast_enabled=True):
                    model = IntegratedT5(t5_config)
        model.transformer.compute_dtype = memory_management.manager.compute_dtype_for_role(
            DeviceRole.TEXT_ENCODER,
            storage_dtype=storage_dtype if isinstance(storage_dtype, torch.dtype) else None,
        )

        if hasattr(state_dict, "keys"):
            from apps.backend.runtime.state_dict.keymap_t5_text_encoder import resolve_t5_text_encoder_keyspace

            state_dict = resolve_t5_text_encoder_keyspace(state_dict).view

        load_state_dict(
            model,
            state_dict,
            log_name=cls_name,
            ignore_errors=["transformer.encoder.embed_tokens.weight", "logit_scale"],
        )
        return model

    if cls_name in {
        "UNet2DConditionModel",
        "FluxTransformer2DModel",
        "Flux2Transformer2DModel",
        "SD3Transformer2DModel",
        "ChromaTransformer2DModel",
        "ZImageTransformer2DModel",
    }:
        if state_dict is None:
            return None
        # Choose configuration source per family/model class
        if cls_name == "UNet2DConditionModel" and family in (ModelFamily.SD15, ModelFamily.SD20, ModelFamily.SDXL, ModelFamily.SDXL_REFINER):
            # Start from parser-provided core_config (LDM-style) and drop unknown keys for legacy UNet
            raw_cfg = dict(config.core_config or {})
            allowed = {
                "in_channels",
                "model_channels",
                "out_channels",
                "num_res_blocks",
                "dropout",
                "channel_mult",
                "conv_resample",
                "dims",
                "num_classes",
                "use_checkpoint",
                "num_heads",
                "num_head_channels",
                "use_scale_shift_norm",
                "resblock_updown",
                "use_spatial_transformer",
                "transformer_depth",
                "context_dim",
                "disable_self_attentions",
                "num_attention_blocks",
                "disable_middle_self_attn",
                "use_linear_in_transformer",
                "adm_in_channels",
                "transformer_depth_middle",
                "transformer_depth_output",
            }
            config_json = {k: v for k, v in raw_cfg.items() if k in allowed}
        else:
            config_json = _load_component_config(component_path)
        core_arch = config.signature.core.architecture
        core_label = _CORE_ARCH_LABELS.get(core_arch, "Core")
        architecture_value = core_arch.value
        module_name = component_name or ("unet" if core_arch is CodexCoreArchitecture.UNET else "transformer")

        if cls_name == "UNet2DConditionModel":
            # For SD15/SD20/SDXL families use Codex legacy UNet with LDM-style config
            from apps.backend.runtime.common.nn.unet.model import UNet2DConditionModel as _CodexUNet

            def model_ctor(cfg: Mapping[str, Any]) -> Any:
                return _CodexUNet.from_config(cfg)

        elif cls_name == "FluxTransformer2DModel":
            from apps.backend.runtime.families.flux.flux import FluxTransformer2DModel

            def model_ctor(cfg: Mapping[str, Any]) -> Any:
                return FluxTransformer2DModel(**dict(cfg))

        elif cls_name == "Flux2Transformer2DModel":
            from diffusers import Flux2Transformer2DModel

            def model_ctor(cfg: Mapping[str, Any]) -> Any:
                filtered = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
                return Flux2Transformer2DModel(**filtered)

        elif cls_name == "ChromaTransformer2DModel":
            from apps.backend.runtime.families.chroma.chroma import ChromaTransformer2DModel

            def model_ctor(cfg: Mapping[str, Any]) -> Any:
                return ChromaTransformer2DModel(**dict(cfg))

        elif cls_name == "ZImageTransformer2DModel":
            from apps.backend.runtime.families.zimage.model import ZImageTransformer2DModel
            # Filter out HuggingFace metadata keys (starting with _)

            def model_ctor(cfg: Mapping[str, Any]) -> Any:
                filtered = {k: v for k, v in cfg.items() if not k.startswith("_")}
                return ZImageTransformer2DModel(**filtered)

        else:
            from apps.backend.runtime.families.sd.mmditx import SD3Transformer2DModel

            def model_ctor(cfg: Mapping[str, Any]) -> Any:
                return SD3Transformer2DModel(**dict(cfg))

        supported_dtypes = _supported_inference_dtypes(family)
        quant_kind = config.quantization.kind
        storage_dtype = memory_management.manager.dtype_for_role(
            DeviceRole.CORE,
            supported=supported_dtypes,
            native_dtype=_native_weights_storage_dtype(weights_path, state_dict),
        )
        _log_weights_dtype_hint(
            role=DeviceRole.CORE,
            selected=storage_dtype,
            hint=_safetensors_primary_dtype_hint(weights_path),
        )
        if quant_kind in (QuantizationKind.NF4, QuantizationKind.FP4):
            raise NotImplementedError(
                "NF4/FP4 is not supported. "
                "Convert the model to GGUF or use a safetensors fp16/bf16/fp32 checkpoint."
            )
        if quant_kind == QuantizationKind.GGUF:
            storage_dtype = "gguf"

        load_device = memory_management.manager.get_device(DeviceRole.CORE)
        offload_device = memory_management.manager.get_offload_device(DeviceRole.CORE)

        mem_config = memory_management.manager.config

        if storage_dtype == "gguf":
            computation_dtype = memory_management.manager.compute_dtype_for_role(
                DeviceRole.CORE,
                supported=supported_dtypes,
            )
            cpu_device = memory_management.manager.cpu_device
            initial_device = memory_management.manager.get_offload_device(DeviceRole.CORE)
            # Smart offload: load transformers to CPU to prevent OOM
            # The engine will move it to GPU on demand via streaming or memory management
            _SMART_OFFLOAD_TRANSFORMERS = {
                "FluxTransformer2DModel",
                "Flux2Transformer2DModel",
                "QwenImageTransformer2DModel",
                "ZImageTransformer2DModel",
                "ChromaTransformer2DModel",
                "SD3Transformer2DModel",
            }
            if (
                cls_name in _SMART_OFFLOAD_TRANSFORMERS
                and smart_offload_enabled()
                and load_device.type != cpu_device.type
            ):
                initial_device = offload_device
                LOGGER.info(
                    "[loader] Smart offload: loading %s to offload device %s (will stream to load device %s)",
                    cls_name,
                    initial_device,
                    load_device,
                )
                log_smart_offload_action(
                    SmartOffloadAction.STAGE_LOAD,
                    source="runtime.models.loader",
                    component=cls_name,
                    from_device=str(load_device),
                    to_device=str(initial_device),
                )

            # For GGUF on CPU, use bfloat16 for storage to avoid unnecessary upcasting
            # The model will automatically cast to appropriate dtype during forward pass on GPU
            construct_dtype = torch.bfloat16 if initial_device.type == cpu_device.type else computation_dtype

            with using_codex_operations(device=initial_device, dtype=construct_dtype, manual_cast_enabled=False, weight_format="gguf"):
                model = model_ctor(config_json)
        else:
            computation_dtype = memory_management.manager.compute_dtype_for_role(
                DeviceRole.CORE,
                supported=supported_dtypes,
                storage_dtype=storage_dtype if isinstance(storage_dtype, torch.dtype) else None,
            )
            
            prefer_gpu = bool(getattr(mem_config, "gpu_prefer_construct", False))
            construct_device = load_device if prefer_gpu else memory_management.manager.get_offload_device(DeviceRole.CORE)
            initial_device = construct_device
            construct_dtype = storage_dtype
            if load_device.type == "cpu" and construct_device.type == "cpu" and construct_dtype in (torch.bfloat16, torch.float16):
                _trace.event(
                    "construct_cpu_cast_override",
                    dtype=str(construct_dtype),
                    to="torch.float32",
                    component=module_name,
                    architecture=architecture_value,
                )
                construct_dtype = torch.float32

            need_manual_cast = construct_dtype != computation_dtype
            to_args = dict(device=construct_device, dtype=construct_dtype)
            _trace.event(
                "core_construct",
                component=module_name,
                architecture=architecture_value,
                device=str(construct_device),
                storage=str(construct_dtype),
                compute=str(computation_dtype),
            )
            try:
                with using_codex_operations(**to_args, manual_cast_enabled=need_manual_cast):
                    model = model_ctor(config_json).to(**to_args)
            except memory_management.manager.oom_exception as exc:
                policy = getattr(mem_config.swap, "policy", None)
                if hasattr(policy, "value"):
                    policy_value = policy.value
                elif policy is not None:
                    policy_value = str(policy)
                else:
                    policy_value = "cpu"
                _trace.event("construct_oom", policy=policy, component=module_name, architecture=architecture_value)
                raise RuntimeError(
                    "Core construction OOM for component={comp} (architecture={arch}) on device={dev} with dtype={dtype}. "
                    "Automatic fallback/offload is disabled. Reduce model precision/size or free VRAM and retry. "
                    "(swap_policy={policy}, gpu_prefer_construct={prefer})"
                .format(
                    comp=module_name,
                    arch=architecture_value,
                    dev=str(construct_device),
                    dtype=str(construct_dtype),
                    policy=str(policy_value),
                    prefer=str(bool(getattr(mem_config, "gpu_prefer_construct", False))),
                )) from exc

        if cls_name in {
            "UNet2DConditionModel",
            "FluxTransformer2DModel",
            "Flux2Transformer2DModel",
            "ChromaTransformer2DModel",
            "SD3Transformer2DModel",
            "ZImageTransformer2DModel",
        }:
            LOGGER.debug(
                "Core load: using parser/keymap output directly for %s (no legacy key-rename normalization).",
                cls_name,
            )


        _trace.event("load_state_dict", module=module_name, architecture=architecture_value, tensors=len(state_dict))
        
        if storage_dtype == "gguf":
            LOGGER.debug("Using strict PyTorch load_state_dict for GGUF model")
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception as exc:
                raise RuntimeError(
                    f"GGUF core load failed (strict): {core_label}. "
                    "Legacy permissive loading is disabled; add/update keymap or parser prefixes."
                ) from exc
        else:
            from .state_dict import safe_load_state_dict as _safe_load
            missing, unexpected = _safe_load(model, state_dict, log_name=core_label)
            if missing or unexpected:
                raise RuntimeError(
                    f"Core load failed (strict): {core_label} missing={len(missing)} unexpected={len(unexpected)} "
                    f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
                )

        # Avoid assigning to model.config (read-only on diffusers models)
        model.storage_dtype = storage_dtype
        model.computation_dtype = computation_dtype
        model.load_device = load_device
        model.initial_device = initial_device
        model.offload_device = offload_device
        model.architecture = core_arch

        return model

    _LOG.debug("Skipping component %s (%s.%s)", component_name, lib_name, cls_name)
    return None


def _apply_prediction_type(codex_components: Dict[str, Any], parsed: ParsedCheckpoint, yaml_prediction: str | None) -> None:
    scheduler = codex_components.get("scheduler")
    if not scheduler or not hasattr(scheduler, "config"):
        return
    desired = _prediction_type_value(parsed.signature.prediction)
    current = getattr(scheduler.config, "prediction_type", None)
    if yaml_prediction:
        scheduler.config.prediction_type = yaml_prediction
        _LOG.info("prediction_type overridden by YAML: %s -> %s", current, yaml_prediction)
        return
    if current and current != desired:
        _LOG.warning(
            "Scheduler prediction_type=%s differs from signature=%s; keeping scheduler value.",
            current,
            desired,
        )
        setattr(scheduler.config, "codex_signature_prediction_type", desired)
        return
    scheduler.config.prediction_type = desired or current or "epsilon"


@torch.no_grad()
def codex_loader(
    sd_path: str,
    additional_state_dicts=None,
    text_encoder_override: TextEncoderOverrideConfig | None = None,
    vae_path: str | None = None,
    tenc_path: str | list[str] | None = None,
    expected_family: ModelFamily | None = None,
) -> DiffusionModelBundle:
    try:
        parsed = _parse_checkpoint(
            sd_path,
            additional_state_dicts or [],
            expected_family=expected_family,
        )
    except ModelRegistryError as exc:
        raise ValueError("Failed to recognize model type!") from exc

    config = parsed.config
    signature = getattr(parsed, "signature", None)

    quant = getattr(signature, "quantization", None)
    extras = getattr(signature, "extras", {}) or {}
    is_flux_core_gguf = (
        isinstance(signature, ModelSignature)
        and signature.family in (ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT)
        and getattr(quant, "kind", None) is QuantizationKind.GGUF
        and bool(extras.get("gguf_core_only"))
    )
    is_flux2_core_only = (
        isinstance(signature, ModelSignature)
        and signature.family is ModelFamily.FLUX2
        and signature.vae is None
        and bool(extras.get("core_only"))
    )
    is_qwen_image_core_only = (
        isinstance(signature, ModelSignature)
        and signature.family is ModelFamily.QWEN_IMAGE
        and getattr(quant, "kind", None) is QuantizationKind.GGUF
        and bool(extras.get("core_only"))
    )
    needs_external_vae = is_flux_core_gguf or is_flux2_core_only or is_qwen_image_core_only
    if needs_external_vae:
        if not isinstance(vae_path, str) or not vae_path.strip():
            if is_qwen_image_core_only:
                family_label = "Qwen Image Edit-2511"
            elif is_flux2_core_only:
                family_label = "FLUX.2"
            else:
                family_label = "Flux GGUF"
            raise RuntimeError(
                f"{family_label} core-only checkpoint requires an external VAE (sha-selected). "
                "Provide a VAE via request extras.vae_sha so the API can pass a valid vae_path."
            )
        vae_path = os.path.expanduser(vae_path.strip())
        if is_qwen_image_core_only:
            vae_path = qwen_image_validate_external_vae_path(
                vae_path,
                context="Qwen Image Edit-2511 external VAE",
            )
        elif not os.path.isfile(vae_path):
            raise RuntimeError(f"Core-only VAE path not found: {vae_path}")
        _assert_core_only_vae_path_not_checkpoint(
            model_ref=sd_path,
            vae_path=vae_path,
            family_label=(
                "Qwen Image Edit-2511"
                if is_qwen_image_core_only
                else "FLUX.2"
                if is_flux2_core_only
                else "Flux GGUF"
            ),
        )

    te_override_cfg = text_encoder_override
    if te_override_cfg is None and tenc_path is not None and parsed.signature.family is not ModelFamily.ZIMAGE:
        # Shorthand: map `tenc_path` entries onto the signature-declared text encoders in order.
        if isinstance(tenc_path, str):
            paths = [tenc_path.strip()] if tenc_path.strip() else []
        elif isinstance(tenc_path, list):
            paths = []
            for entry in tenc_path:
                if not isinstance(entry, str):
                    raise TypeError("tenc_path must be a string or list[str] when provided.")
                item = entry.strip()
                if item:
                    paths.append(item)
        else:
            raise TypeError("tenc_path must be a string or list[str] when provided.")
        if not paths:
            raise RuntimeError("tenc_path was provided but empty after trimming.")

        aliases = tuple(te.name for te in parsed.signature.text_encoders)
        if not aliases:
            raise RuntimeError(
                "tenc_path override was provided, but this checkpoint declares no text encoders in the signature."
            )
        if len(paths) != len(aliases):
            raise RuntimeError(
                "tenc_path override expects exactly %d paths for this checkpoint (encoders=%s); got %d."
                % (len(aliases), ", ".join(aliases), len(paths))
            )

        te_override_cfg = TextEncoderOverrideConfig(
            family=_canonical_override_family(parsed.signature.family),
            root_label="tenc_path",
            components=aliases,
            explicit_paths={alias: path for alias, path in zip(aliases, paths, strict=True)},
        )

    try:
        te_override_paths = resolve_text_encoder_override_paths(
            signature=parsed.signature,
            estimated_config=config,
            override=te_override_cfg,
        )
    except TextEncoderOverrideError as exc:
        # Keep the surface error explicit and actionable; do not fall back silently.
        raise RuntimeError(str(exc)) from exc

    resolved_flux2_tenc_path: str | None = None
    resolved_qwen_image_tenc_path: str | None = None
    if is_flux2_core_only:
        if te_override_cfg is None:
            raise RuntimeError(
                "FLUX.2 core-only checkpoint requires an external text encoder (Qwen3-4B; sha-selected). "
                "Provide one via request extras.tenc_sha so the API can pass a valid tenc_path."
            )
        resolved_tenc_candidates = [
            value.strip() for value in te_override_paths.values() if isinstance(value, str) and value.strip()
        ]
        if len(resolved_tenc_candidates) != 1:
            raise RuntimeError(
                "FLUX.2 text encoder override resolution failed: expected exactly one non-empty external path, "
                f"got {len(resolved_tenc_candidates)}."
            )
        resolved_flux2_tenc_path = os.path.expanduser(resolved_tenc_candidates[0])
        if not os.path.isfile(resolved_flux2_tenc_path):
            raise RuntimeError(f"FLUX.2 text encoder path not found: {resolved_flux2_tenc_path}")

    if is_qwen_image_core_only:
        if te_override_cfg is None:
            raise RuntimeError(
                "Qwen Image Edit-2511 requires the exact external Qwen2.5-VL text encoder GGUF (sha-selected). "
                "Provide one via request extras.tenc_sha so the API can pass a valid tenc_path."
            )
        resolved_tenc_candidates = [
            value.strip() for value in te_override_paths.values() if isinstance(value, str) and value.strip()
        ]
        if len(resolved_tenc_candidates) != 1:
            raise RuntimeError(
                "Qwen Image Edit-2511 text encoder override resolution failed: "
                f"expected exactly one non-empty external path, got {len(resolved_tenc_candidates)}."
            )
        resolved_qwen_image_tenc_path = os.path.expanduser(resolved_tenc_candidates[0])
        if not os.path.isfile(resolved_qwen_image_tenc_path):
            raise RuntimeError(
                f"Qwen Image Edit-2511 text encoder path not found: {resolved_qwen_image_tenc_path}"
            )
        resolved_slot = classify_text_encoder_slot(resolved_qwen_image_tenc_path)
        if resolved_slot != "qwen2_5_vl_7b":
            raise RuntimeError(
                "Qwen Image Edit-2511 requires text encoder slot 'qwen2_5_vl_7b'; "
                f"got {resolved_slot!r} for {resolved_qwen_image_tenc_path!r}."
            )

    component_states = {name: comp.state_dict for name, comp in config.components.items()}

    if is_qwen_image_core_only:
        if resolved_qwen_image_tenc_path is None or not isinstance(vae_path, str):
            raise RuntimeError("Qwen Image Edit-2511 external component resolution did not complete.")
        return _build_diffusion_bundle(
            model_ref=sd_path,
            family=parsed.signature.family,
            estimated_config=config,
            components=component_states,
            signature=parsed.signature,
            source="state_dict",
            metadata={
                "engine_key": QWEN_IMAGE_ENGINE_ID,
                "core_only": True,
                "requires_vae": True,
                QWEN_IMAGE_VARIANT_KEY: QWEN_IMAGE_EDIT_VARIANT,
                "zero_cond_t": True,
                "tenc_override_paths": dict(te_override_paths),
                "tenc_path": resolved_qwen_image_tenc_path,
                "vae_path": vae_path,
                "metadata_root": str(
                    _BACKEND_ROOT / "huggingface" / "Qwen" / "Qwen-Image-Edit-2511"
                ),
            },
        )

    if parsed.signature.family is ModelFamily.ANIMA:
        # Anima is not a diffusers repository and must not rely on `model_index.json`/DiffusionPipeline config.
        # Return a minimal bundle containing the parsed core component state dicts; engines load external assets.
        if not isinstance(vae_path, str) or not vae_path.strip():
            raise RuntimeError(
                "Anima checkpoint requires an external VAE (sha-selected). "
                "Provide a VAE via request extras.vae_sha so the API can pass a valid vae_path."
            )
        resolved_vae_path = os.path.expanduser(vae_path.strip())
        if not os.path.isfile(resolved_vae_path):
            raise RuntimeError(f"Anima VAE path not found: {resolved_vae_path}")

        if te_override_cfg is None:
            raise RuntimeError(
                "Anima checkpoint requires an external text encoder (Qwen3-0.6B; sha-selected). "
                "Provide one via request extras.tenc_sha so the API can pass a valid tenc_path."
            )
        resolved_tenc_candidates = [
            value.strip() for value in te_override_paths.values() if isinstance(value, str) and value.strip()
        ]
        if len(resolved_tenc_candidates) != 1:
            raise RuntimeError(
                "Anima text encoder override resolution failed: expected exactly one non-empty external path, "
                f"got {len(resolved_tenc_candidates)}."
            )
        resolved_tenc_path = resolved_tenc_candidates[0]
        resolved_tenc_path = os.path.expanduser(resolved_tenc_path)
        if not os.path.isfile(resolved_tenc_path):
            raise RuntimeError(f"Anima text encoder path not found: {resolved_tenc_path}")

        return _build_diffusion_bundle(
            model_ref=sd_path,
            family=parsed.signature.family,
            estimated_config=config,
            components={name: comp.state_dict for name, comp in config.components.items()},
            signature=parsed.signature,
            source="state_dict",
            metadata={
                "engine_key": "anima",
                "tenc_override_paths": dict(te_override_paths),
                "tenc_path": resolved_tenc_path,
                "vae_path": resolved_vae_path,
            },
        )

    if parsed.signature.family is ModelFamily.ZIMAGE_L2P:
        if te_override_cfg is None:
            raise RuntimeError(
                "Z-Image L2P checkpoint requires an external text encoder (Qwen3-4B; sha-selected). "
                "Provide one via request extras.tenc_sha so the API can pass a valid tenc_path."
            )
        resolved_tenc_candidates = [
            value.strip() for value in te_override_paths.values() if isinstance(value, str) and value.strip()
        ]
        if len(resolved_tenc_candidates) != 1:
            raise RuntimeError(
                "Z-Image L2P text encoder override resolution failed: expected exactly one non-empty external path, "
                f"got {len(resolved_tenc_candidates)}."
            )
        resolved_tenc_path = os.path.expanduser(resolved_tenc_candidates[0])
        if not os.path.isfile(resolved_tenc_path):
            raise RuntimeError(f"Z-Image L2P text encoder path not found: {resolved_tenc_path}")
        return _build_diffusion_bundle(
            model_ref=sd_path,
            family=parsed.signature.family,
            estimated_config=config,
            components={name: comp.state_dict for name, comp in config.components.items()},
            signature=parsed.signature,
            source="state_dict",
            metadata={
                "engine_key": "zimage_l2p",
                "core_only": True,
                "requires_vae": False,
                "tenc_override_paths": dict(te_override_paths),
                "tenc_path": resolved_tenc_path,
            },
        )

    if parsed.signature.family is ModelFamily.LTX2:
        if te_override_cfg is None:
            raise RuntimeError(
                "LTX2 checkpoint requires an external text encoder (Gemma3-12B; sha-selected). "
                "Provide one via request extras.tenc_sha so the API can pass a valid tenc_path."
            )
        resolved_ltx2_tenc_candidates = [
            value.strip() for value in te_override_paths.values() if isinstance(value, str) and value.strip()
        ]
        if len(resolved_ltx2_tenc_candidates) != 1:
            raise RuntimeError(
                "LTX2 text encoder override resolution failed: expected exactly one non-empty external path, "
                f"got {len(resolved_ltx2_tenc_candidates)}."
            )
        ltx2_text_encoder_override_paths = {
            LTX2_REQUIRED_TEXT_ENCODER_SLOT: os.path.expanduser(resolved_ltx2_tenc_candidates[0])
        }
        ltx2_inputs = prepare_ltx2_bundle_inputs(
            model_ref=sd_path,
            estimated_config=config,
            signature=parsed.signature,
            text_encoder_override_paths=ltx2_text_encoder_override_paths,
            vae_path=vae_path,
            backend_root=_BACKEND_ROOT,
        )
        ltx2_component_states = {
            "transformer": ltx2_inputs.components.transformer,
            "connectors": ltx2_inputs.components.connectors,
            "vae": ltx2_inputs.components.vae,
            "audio_vae": ltx2_inputs.components.audio_vae,
            "vocoder": ltx2_inputs.components.vocoder,
        }
        estimated_component_names = tuple(ltx2_inputs.estimated_config.components.keys())
        if estimated_component_names != tuple(ltx2_component_states.keys()):
            raise RuntimeError(
                "LTX2 loader bundle assembly drifted from the rewritten component contract. "
                f"estimated_config.components={estimated_component_names!r} "
                f"bundle_components={tuple(ltx2_component_states.keys())!r}."
            )
        for component_name, component_state in ltx2_component_states.items():
            estimated_component_state = ltx2_inputs.estimated_config.components[component_name].state_dict
            if estimated_component_state is not component_state:
                raise RuntimeError(
                    "LTX2 loader bundle assembly detected stale component state after bundle planning rewrite. "
                    f"component={component_name!r}."
                )
        metadata = build_ltx2_bundle_metadata(ltx2_inputs)
        metadata["tenc_override_paths"] = dict(te_override_paths)
        return _build_diffusion_bundle(
            model_ref=sd_path,
            family=parsed.signature.family,
            estimated_config=ltx2_inputs.estimated_config,
            components=ltx2_component_states,
            signature=parsed.signature,
            source="state_dict",
            metadata=metadata,
        )

    repo_name = config.repo_id
    if not isinstance(repo_name, str) or not repo_name:
        raise ValueError("Codex model parser did not resolve a repository id")

    local_repo_path = os.path.join(str(_BACKEND_ROOT), "huggingface", repo_name)
    offline = bool(args.disable_online_tokenizer)
    include = ("config", "tokenizer", "scheduler")  # strictly minimal; no weights
    ensure_repo_minimal_files(repo_name, local_repo_path, offline=offline, include=include)

    from diffusers import DiffusionPipeline

    pipeline_config = DiffusionPipeline.load_config(local_repo_path)
    codex_components: Dict[str, Any] = {}

    for component_name, component_info in pipeline_config.items():
        if not (isinstance(component_info, list) and len(component_info) == 2):
            continue
        lib_name, cls_name = component_info
        component_sd = component_states.get(component_name)

        if component_sd is None and needs_external_vae and cls_name in {"AutoencoderKL", "AutoencoderKLFlux2"}:
            component_sd = _load_state_dict(vae_path)

        override_path = te_override_paths.get(component_name)
        if override_path is not None:
            if not os.path.isfile(override_path):
                raise RuntimeError(
                    "Text encoder override path for component %s does not exist: %s"
                    % (component_name, override_path)
                )
            component_sd = _load_state_dict(override_path)

        component_obj = _load_huggingface_component(
            parsed,
            component_name,
            lib_name,
            cls_name,
            local_repo_path,
            component_sd,
            weights_path=(
                override_path
                or (vae_path if needs_external_vae and cls_name in {"AutoencoderKL", "AutoencoderKLFlux2"} else sd_path)
            ),
        )
        if component_sd is not None:
            component_states.pop(component_name, None)
        if component_obj is not None:
            codex_components[component_name] = component_obj

    yaml_prediction = None
    config_filename = os.path.splitext(sd_path)[0] + ".yaml"
    if os.path.isfile(config_filename):
        try:
            import yaml
            with open(config_filename, "r", encoding="utf-8") as stream:
                yaml_config = yaml.safe_load(stream)
            yaml_prediction = (
                yaml_config.get("model", {}).get("params", {}).get("parameterization", "")
                or yaml_config.get("model", {})
                .get("params", {})
                .get("denoiser_config", {})
                .get("params", {})
                .get("scaling_config", {})
                .get("target", "")
            )
            if yaml_prediction == "v" or yaml_prediction.endswith(".VScaling"):
                yaml_prediction = "v_prediction"
            elif not yaml_prediction:
                yaml_prediction = None
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("Failed to parse YAML config %s: %s", config_filename, exc)

    _apply_prediction_type(codex_components, parsed, yaml_prediction)

    prediction_type_override = yaml_prediction
    forced_prediction = os.environ.get("CODEX_FORCE_PREDICTION_TYPE")
    if forced_prediction:
        forced_prediction = forced_prediction.strip().lower()
        if forced_prediction in {"epsilon", "eps"}:
            forced_prediction = "epsilon"
        elif forced_prediction in {"v_prediction", "v"}:
            forced_prediction = "v_prediction"
        else:
            raise ValueError(
                "CODEX_FORCE_PREDICTION_TYPE must be one of: epsilon, v_prediction (also accepts eps/v). "
                f"Got: {forced_prediction!r}"
            )

        scheduler = codex_components.get("scheduler")
        if not scheduler or not hasattr(scheduler, "config"):
            raise RuntimeError(
                "CODEX_FORCE_PREDICTION_TYPE is set, but the loaded pipeline has no scheduler/config to override."
            )
        current = getattr(scheduler.config, "prediction_type", None)
        scheduler.config.prediction_type = forced_prediction
        _LOG.warning(
            "prediction_type forced via CODEX_FORCE_PREDICTION_TYPE: %s -> %s",
            current,
            forced_prediction,
        )
        prediction_type_override = forced_prediction

    metadata = {"repo_id": repo_name}
    if prediction_type_override:
        metadata["prediction_type"] = prediction_type_override
    # Note: VAE selection is expressed via engine options (`vae_path` + `vae_source`).
    if te_override_paths:
        metadata["tenc_override_paths"] = dict(te_override_paths)
    if resolved_flux2_tenc_path is not None:
        metadata["tenc_path"] = resolved_flux2_tenc_path
    if is_flux2_core_only and isinstance(vae_path, str) and vae_path.strip():
        metadata["vae_path"] = os.path.expanduser(vae_path.strip())

    return _build_diffusion_bundle(
        model_ref=sd_path,
        family=parsed.signature.family,
        estimated_config=config,
        components=codex_components,
        signature=parsed.signature,
        source="state_dict",
        metadata=metadata,
    )


# ------------------------------ Native diffusers repo loader (no state dict)
class _SimpleEstimated:
    def __init__(
        self,
        *,
        huggingface_repo: str,
        core_config: dict,
        pipeline_class: str = "",
        text_encoder_map: Mapping[str, str] | None = None,
    ):
        self.huggingface_repo = huggingface_repo
        self.core_config = core_config
        self.pipeline_class = pipeline_class
        self.text_encoder_map = dict(text_encoder_map or {})


def resolve_sdxl_diffusers_surface(
    *,
    config: Mapping[str, Any],
    repo_dir: str,
    expected_family: ModelFamily | None = None,
) -> str | None:
    def _has_present_component(name: str) -> bool:
        value = config.get(name)
        return (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], str)
            and bool(value[0].strip())
            and isinstance(value[1], str)
            and bool(value[1].strip())
        )

    has_unet = _has_present_component("unet")
    has_text_encoder = _has_present_component("text_encoder")
    has_text_encoder_2 = _has_present_component("text_encoder_2")
    if not has_unet or (not has_text_encoder and not has_text_encoder_2):
        return None

    unet_config_path = Path(repo_dir) / "unet" / "config.json"
    unet_config = _read_json(unet_config_path) if unet_config_path.is_file() else {}
    pipeline_cls = str(config.get("_class_name") or "").strip().lower()
    text_encoder_spec = config.get("text_encoder")
    text_encoder_cls = (
        text_encoder_spec[1].strip()
        if isinstance(text_encoder_spec, list)
        and len(text_encoder_spec) == 2
        and isinstance(text_encoder_spec[1], str)
        else ""
    )
    cross_attention_dim_raw = unet_config.get("cross_attention_dim")
    cross_attention_dim = None
    if isinstance(cross_attention_dim_raw, (int, float, str)) and str(cross_attention_dim_raw).strip():
        cross_attention_dim = int(cross_attention_dim_raw)

    if has_text_encoder_2:
        detected_engine = "sdxl"
    elif (
        has_text_encoder
        and text_encoder_cls == "CLIPTextModelWithProjection"
        and (cross_attention_dim == 1280 or pipeline_cls == "stablediffusionxlimg2imgpipeline")
    ):
        detected_engine = "sdxl_refiner"
    elif cross_attention_dim == 1280:
        detected_engine = "sdxl_refiner"
    elif cross_attention_dim == 2048:
        detected_engine = "sdxl"
    elif expected_family is ModelFamily.SDXL_REFINER:
        detected_engine = "sdxl_refiner"
    elif expected_family is ModelFamily.SDXL:
        detected_engine = "sdxl"
    else:
        raise ValueError(
            f"Unable to determine SDXL diffusers surface from native metadata: repo={repo_dir} cross_attention_dim={cross_attention_dim!r}"
        )

    if expected_family is ModelFamily.SDXL and detected_engine != "sdxl":
        raise ValueError(f"Expected SDXL base diffusers repo, but '{repo_dir}' matches the SDXL refiner surface.")
    if expected_family is ModelFamily.SDXL_REFINER and detected_engine != "sdxl_refiner":
        raise ValueError(f"Expected SDXL refiner diffusers repo, but '{repo_dir}' matches the SDXL base surface.")
    return detected_engine


def _detect_engine_from_config(
    config: dict,
    *,
    repo_dir: str,
    expected_family: ModelFamily | None = None,
) -> str:
    pipeline_cls = str(config.get("_class_name") or "").strip().lower()
    if pipeline_cls in _QWEN_IMAGE_PIPELINE_VARIANTS:
        return QWEN_IMAGE_ENGINE_ID
    if pipeline_cls == "fluxkontextpipeline":
        return "flux1_kontext"
    if pipeline_cls == "flux2kleinpipeline":
        return "flux2"
    if pipeline_cls == "flux2pipeline":
        raise ValueError("Unsupported FLUX.2 pipeline config: only Flux2KleinPipeline (4B/base-4B) is supported.")
    comps = {k: v for k, v in config.items() if isinstance(v, list) and len(v) == 2}
    cls_by_name = {k: v[1] for k, v in comps.items()}
    sdxl_surface = resolve_sdxl_diffusers_surface(
        config=config,
        repo_dir=repo_dir,
        expected_family=expected_family,
    )
    if sdxl_surface is not None:
        return sdxl_surface
    if cls_by_name.get("transformer") in ("FluxTransformer2DModel",):
        return "flux1"
    if cls_by_name.get("transformer") in ("Flux2Transformer2DModel",):
        return "flux2"
    if cls_by_name.get("transformer") in ("QwenImageTransformer2DModel",):
        return QWEN_IMAGE_ENGINE_ID
    if cls_by_name.get("transformer") in ("SD3Transformer2DModel",):
        return "sd35"
    if cls_by_name.get("transformer") in ("ChromaTransformer2DModel",):
        return "flux1_chroma"
    if "unet" in comps and "text_encoder" in comps and "vae" in comps:
        te_cls = cls_by_name.get("text_encoder", "")
        if te_cls.endswith("WithProjection"):
            return "sd20"
        return "sd15"
    raise ValueError("Unable to determine engine from diffusers config")


def load_engine_from_diffusers(
    repo_dir: str,
    *,
    expected_family: ModelFamily | None = None,
) -> DiffusionModelBundle:
    config = _load_diffusers_model_index(repo_dir)
    if _is_qwen_image_model_index(config):
        return _qwen_image_bundle_from_diffusers_metadata(
            repo_dir=repo_dir,
            model_index=config,
            expected_family=expected_family,
        )
    comps = {}
    for name, (lib_name, cls_name) in (
        (k, v) for k, v in config.items() if isinstance(v, list) and len(v) == 2
    ):
        # Optional components are represented as [null, null] in model_index.json
        if lib_name is None and cls_name is None:
            continue
        if not isinstance(lib_name, str) or not isinstance(cls_name, str):
            raise TypeError(
                f"Invalid diffusers component spec for '{name}': expected [str, str] or [null, null], "
                f"got [{type(lib_name).__name__}, {type(cls_name).__name__}]"
            )
        cls = getattr(importlib.import_module(lib_name), cls_name)
        comps[name] = cls.from_pretrained(os.path.join(repo_dir, name), local_files_only=True)

    engine_key = _detect_engine_from_config(
        config,
        repo_dir=repo_dir,
        expected_family=expected_family,
    )
    family = ENGINE_KEY_TO_FAMILY.get(engine_key)
    if family is None:
        raise ValueError(f"Unsupported engine key from diffusers config: {engine_key}")
    core_config = {}
    try:
        for k in ("unet", "transformer"):
            cfg_dir = os.path.join(repo_dir, k)
            if os.path.isdir(cfg_dir):
                cfg_path = os.path.join(cfg_dir, "config.json")
                if os.path.isfile(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as fh:
                        core_config = json.load(fh)
                    break
    except Exception:
        core_config = {}

    if family is ModelFamily.FLUX2:
        _validate_supported_flux2_transformer_config(core_config, context=repo_dir)

    est = _SimpleEstimated(
        huggingface_repo=os.path.basename(repo_dir),
        core_config=core_config,
        pipeline_class=str(config.get("_class_name") or ""),
    )

    return _build_diffusion_bundle(
        model_ref=repo_dir,
        family=family,
        estimated_config=est,
        components=comps,
        source="diffusers",
        metadata={"engine_key": engine_key, "core_config": core_config},
    )


def resolve_diffusion_bundle(
    model_ref: str,
    *,
    additional_state_dicts: Optional[list[str]] = None,
    text_encoder_override: TextEncoderOverrideConfig | None = None,
    vae_path: str | None = None,
    tenc_path: str | list[str] | None = None,
    expected_family: ModelFamily | None = None,
) -> DiffusionModelBundle:
    """Resolve a diffusion model reference into a fully loaded bundle."""
    if os.path.isdir(model_ref):
        index = os.path.join(model_ref, "model_index.json")
        if os.path.isfile(index):
            return load_engine_from_diffusers(model_ref, expected_family=expected_family)
        raise ValueError(f"Not a diffusers repository (missing model_index.json): {model_ref}")

    if os.path.isfile(model_ref):
        return codex_loader(
            model_ref,
            additional_state_dicts=additional_state_dicts,
            text_encoder_override=text_encoder_override,
            vae_path=vae_path,
            tenc_path=tenc_path,
            expected_family=expected_family,
        )

    record = model_api.find_checkpoint(model_ref)
    if record is None:
        raise ValueError(f"Checkpoint not found: {model_ref}")

    # Determine format via metadata or filesystem inspection
    metadata = getattr(record, "metadata", {}) or {}
    if isinstance(metadata, dict) and metadata.get("format") == "diffusers":
        return load_engine_from_diffusers(record.path, expected_family=expected_family)

    repo_index = os.path.join(record.path, "model_index.json")
    if os.path.isfile(repo_index):
        return load_engine_from_diffusers(record.path, expected_family=expected_family)
    return codex_loader(
        record.filename,
        additional_state_dicts=additional_state_dicts,
        text_encoder_override=text_encoder_override,
        vae_path=vae_path,
        tenc_path=tenc_path,
        expected_family=expected_family,
    )

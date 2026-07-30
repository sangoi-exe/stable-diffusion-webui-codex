"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Header-only text encoder slot classification for sha-selected assets.
Provides a fast, import-light helper used by API request paths to classify a resolved text encoder weights file into
an explicit slot (`clip_l`, `clip_g`, `t5xxl`, `qwen3_4b`, `qwen3_06b`, `qwen2_5_vl_7b`, `gemma3_12b`) without loading tensor payloads.
Dedicated Z-Image L2P Qwen3-4B and Qwen Image Edit-2511 Qwen2.5-VL GGUFs require their exact Codex metadata and tensor topology.

Symbols (top-level; keep in sync; no ghosts):
- `TextEncoderSlotError` (class): Raised when a weights file cannot be classified into a known slot.
- `classify_text_encoder_slot` (function): Classify a weights file into a slot using header-only inspection.
- `map_text_encoder_paths_to_slots` (function): Classify paths into an expected slot set (order-independent).
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from apps.backend.quantization.gguf import GGMLQuantizationType, GGUFReader
from apps.backend.runtime.state_dict.keymap_qwen_text_encoder import resolve_qwen2_5_vl_multimodal_keyspace


class TextEncoderSlotError(ValueError):
    """Raised when a text encoder weights file cannot be classified."""


def _looks_like_mmproj_context(context: str) -> bool:
    lower = Path(str(context or "")).name.strip().lower()
    return "mmproj" in lower or "multimodal-projector" in lower


def _gguf_is_multimodal_projector(kv: Mapping[str, object]) -> bool:
    gguf_type = str(kv.get("general.type") or kv.get("model.type") or "").strip().lower()
    projector_type = str(kv.get("clip.projector_type") or "").strip().lower()
    has_vision_encoder = kv.get("clip.has_vision_encoder")
    return gguf_type in {"mmproj", "multimodal-projector", "projector"} or bool(projector_type) or bool(
        has_vision_encoder
    )


def classify_text_encoder_slot(path: str) -> str:
    """Classify a text encoder weights file into a known slot.

    This function must be safe to call in API request paths:
    - no torch imports
    - no tensor loading (header-only)
    - explicit errors (no guessing)
    """

    raw = str(path or "").strip()
    if not raw:
        raise TextEncoderSlotError("Text encoder path is required for slot classification.")

    p = Path(os.path.expanduser(raw)).resolve(strict=False)
    if not p.exists() or not p.is_file():
        raise TextEncoderSlotError(f"Text encoder path not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".safetensor", ".safetensors"}:
        header = _read_safetensors_header(p)
        return _classify_safetensors_header(header, context=str(p))
    if suffix == ".gguf":
        try:
            kv = _read_gguf_kv(p)
            descriptors = _read_gguf_tensor_descriptors(p)
        except TextEncoderSlotError:
            raise
        except Exception as exc:
            raise TextEncoderSlotError(f"Failed to read GGUF metadata for slot classification: {p}: {exc}") from exc
        try:
            return _classify_gguf_kv(kv, tensor_descriptors=descriptors, context=str(p))
        except TextEncoderSlotError:
            raise
        except Exception as exc:
            raise TextEncoderSlotError(f"Failed to classify GGUF text encoder slot for {p}: {exc}") from exc

    raise TextEncoderSlotError(
        "Unsupported text encoder weights extension %r (supported: .safetensor, .safetensors, .gguf)" % suffix
    )


def map_text_encoder_paths_to_slots(*, paths: Sequence[str], expected_slots: Sequence[str]) -> dict[str, str]:
    """Classify a set of text encoder weight paths into an expected slot set (order-independent)."""

    exp = tuple(str(s).strip() for s in expected_slots)
    if any(not s for s in exp):
        raise TextEncoderSlotError("Expected text encoder slots must not contain empty values.")
    if len(set(exp)) != len(exp):
        raise TextEncoderSlotError(f"Expected text encoder slots must be unique: {list(exp)}")

    cleaned = [str(p).strip() for p in paths if str(p).strip()]
    if len(cleaned) != len(exp):
        raise TextEncoderSlotError(
            f"Expected exactly {len(exp)} text encoder path(s) for slots={list(exp)}, got {len(cleaned)}."
        )

    slot_to_path: dict[str, str] = {}
    for raw in cleaned:
        slot = classify_text_encoder_slot(raw)
        if slot not in exp:
            raise TextEncoderSlotError(
                f"Text encoder slot mismatch: got slot={slot!r} for path={raw!r}, expected one of {list(exp)}."
            )
        if slot in slot_to_path:
            raise TextEncoderSlotError(
                f"Duplicate text encoder slot {slot!r} for slots={list(exp)} (path={raw!r})."
            )
        slot_to_path[slot] = raw

    missing = [slot for slot in exp if slot not in slot_to_path]
    if missing:
        raise TextEncoderSlotError(
            f"Missing required text encoder slot(s) {missing} for slots={list(exp)} (classified={sorted(slot_to_path)})."
        )

    return slot_to_path


def _classify_safetensors_header(header: Mapping[str, object], *, context: str) -> str:
    keys = [k for k in header.keys() if k != "__metadata__"]
    if not keys:
        raise TextEncoderSlotError(f"Empty safetensors header (no tensors): {context}")

    def _shape_for_suffix(suffixes: tuple[str, ...]) -> tuple[int, ...] | None:
        for k, meta in header.items():
            if k == "__metadata__":
                continue
            key = str(k)
            if not any(key.endswith(suf) for suf in suffixes):
                continue
            if not isinstance(meta, dict):
                continue
            shape = meta.get("shape")
            if isinstance(shape, (list, tuple)) and shape and all(isinstance(x, (int, float)) for x in shape):
                return tuple(int(x) for x in shape)
        return None

    def _has_key_prefix(prefix: str) -> bool:
        return any(str(key).startswith(prefix) for key in keys)

    # --- CLIP (CLIP-L / CLIP-G)
    clip_shape = _shape_for_suffix(
        (
            "text_model.embeddings.token_embedding.weight",
            "model.token_embedding.weight",
            "token_embedding.weight",
        )
    )
    if clip_shape and len(clip_shape) >= 2:
        hidden = int(clip_shape[-1])
        if hidden == 768:
            return "clip_l"
        if hidden == 1280:
            return "clip_g"
        raise TextEncoderSlotError(
            f"Unrecognized CLIP hidden size {hidden} for {context} (expected 768 or 1280)."
        )

    # --- T5 (T5-XXL)
    t5_shared = _shape_for_suffix(("shared.weight",))
    t5_attn = _shape_for_suffix(("encoder.block.0.layer.0.SelfAttention.q.weight",))
    if t5_shared and t5_attn and len(t5_shared) >= 2:
        hidden = int(t5_shared[-1])
        if hidden == 4096:
            return "t5xxl"
        raise TextEncoderSlotError(
            f"Unrecognized T5 hidden size {hidden} for {context} (expected 4096 for T5-XXL)."
        )

    # --- Qwen / Gemma (used by flow-based text encoder selection)
    qwen_embed = _shape_for_suffix(("model.embed_tokens.weight", "embed_tokens.weight"))
    if qwen_embed and len(qwen_embed) >= 2:
        hidden = int(qwen_embed[-1])
        if hidden == 3840:
            return "gemma3_12b"
        if hidden == 3584:
            if _has_key_prefix("visual."):
                return "qwen2_5_vl_7b"
            raise TextEncoderSlotError(
                f"Qwen2.5-VL-7B slot classification for {context} requires visual-tower tensor evidence "
                "(`visual.*`) in addition to embed dim 3584."
            )
        if hidden == 2560:
            return "qwen3_4b"
        if hidden == 1024:
            return "qwen3_06b"
        raise TextEncoderSlotError(
            f"Unrecognized LLM embed dim {hidden} for {context} "
            "(expected 3840 (Gemma3-12B), 3584 with visual tower (Qwen2.5-VL-7B), "
            "2560 (Qwen3-4B), or 1024 (Qwen3-0.6B))."
        )

    raise TextEncoderSlotError(
        "Could not classify text encoder slot from safetensors header for %s. "
        "Expected a CLIP (token_embedding), T5 (shared+encoder.block), or LLM (embed_tokens) weights file."
        % context
    )


_GGUF_MAGIC = 0x46554747  # 'GGUF' little-endian

_GGUF_VALUE_UINT8 = 0
_GGUF_VALUE_INT8 = 1
_GGUF_VALUE_UINT16 = 2
_GGUF_VALUE_INT16 = 3
_GGUF_VALUE_UINT32 = 4
_GGUF_VALUE_INT32 = 5
_GGUF_VALUE_FLOAT32 = 6
_GGUF_VALUE_BOOL = 7
_GGUF_VALUE_STRING = 8
_GGUF_VALUE_ARRAY = 9
_GGUF_VALUE_UINT64 = 10
_GGUF_VALUE_INT64 = 11
_GGUF_VALUE_FLOAT64 = 12

_SAFE_MAX_GGUF_ARRAY_ITEMS = 256


@dataclass(frozen=True, slots=True)
class _GGUFTensorDescriptor:
    shape: tuple[int, ...]
    tensor_type: GGMLQuantizationType
    n_bytes: int


def _read_exact(handle, n: int) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise EOFError(f"Unexpected EOF (wanted {n} bytes, got {len(data)}).")
    return data


def _read_u32(handle) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]

def _read_i32(handle) -> int:
    return struct.unpack("<i", _read_exact(handle, 4))[0]


def _read_u64(handle) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]

def _read_i64(handle) -> int:
    return struct.unpack("<q", _read_exact(handle, 8))[0]

def _read_f32(handle) -> float:
    return struct.unpack("<f", _read_exact(handle, 4))[0]

def _read_f64(handle) -> float:
    return struct.unpack("<d", _read_exact(handle, 8))[0]


def _read_string(handle) -> str:
    n = _read_u64(handle)
    raw = _read_exact(handle, int(n))
    return raw.decode("utf-8", errors="replace")


def _read_gguf_value(handle, vtype: int) -> object:
    if vtype == _GGUF_VALUE_UINT8:
        return _read_exact(handle, 1)[0]
    if vtype == _GGUF_VALUE_INT8:
        return struct.unpack("<b", _read_exact(handle, 1))[0]
    if vtype == _GGUF_VALUE_UINT16:
        return struct.unpack("<H", _read_exact(handle, 2))[0]
    if vtype == _GGUF_VALUE_INT16:
        return struct.unpack("<h", _read_exact(handle, 2))[0]
    if vtype == _GGUF_VALUE_UINT32:
        return _read_u32(handle)
    if vtype == _GGUF_VALUE_INT32:
        return _read_i32(handle)
    if vtype == _GGUF_VALUE_FLOAT32:
        return _read_f32(handle)
    if vtype == _GGUF_VALUE_BOOL:
        return bool(_read_exact(handle, 1)[0])
    if vtype == _GGUF_VALUE_UINT64:
        return _read_u64(handle)
    if vtype == _GGUF_VALUE_INT64:
        return _read_i64(handle)
    if vtype == _GGUF_VALUE_FLOAT64:
        return _read_f64(handle)
    if vtype == _GGUF_VALUE_STRING:
        return _read_string(handle)
    if vtype == _GGUF_VALUE_ARRAY:
        elem_type = _read_u32(handle)
        count = _read_u64(handle)
        preview: list[object] = []
        for i in range(int(count)):
            value = _read_gguf_value(handle, int(elem_type))
            if i < _SAFE_MAX_GGUF_ARRAY_ITEMS:
                preview.append(value)
        if count > _SAFE_MAX_GGUF_ARRAY_ITEMS:
            return {"__truncated__": True, "count": int(count), "preview": preview, "elem_type": int(elem_type)}
        return preview
    raise ValueError(f"Unsupported GGUF value type: {vtype}")


def _read_gguf_kv(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        magic = _read_u32(handle)
        if magic != _GGUF_MAGIC:
            raise TextEncoderSlotError(f"Not a GGUF file (magic={hex(magic)}): {path}")
        _ = _read_u32(handle)  # version
        _ = _read_u64(handle)  # n_tensors
        n_kv = _read_u64(handle)

        kv: dict[str, object] = {}
        for _ in range(int(n_kv)):
            key = _read_string(handle)
            vtype = _read_u32(handle)
            kv[key] = _read_gguf_value(handle, int(vtype))

    return kv


def _read_gguf_tensor_descriptors(path: Path) -> dict[str, _GGUFTensorDescriptor]:
    reader = GGUFReader(path)
    descriptors: dict[str, _GGUFTensorDescriptor] = {}
    for tensor in reader.tensors:
        descriptors[tensor.name] = _GGUFTensorDescriptor(
            shape=tuple(int(dim) for dim in reversed(tensor.shape.tolist())),
            tensor_type=tensor.tensor_type,
            n_bytes=int(tensor.n_bytes),
        )
    return descriptors


def _metadata_int(kv: Mapping[str, object], key: str) -> int | None:
    raw = kv.get(key)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, float) and float(raw).is_integer():
        return int(raw)
    if isinstance(raw, str):
        value = raw.strip()
        if value.isdigit():
            return int(value)
    return None


def _validate_qwen_image_qwen2_5_vl_gguf(
    kv: Mapping[str, object],
    tensor_descriptors: Mapping[str, _GGUFTensorDescriptor],
    *,
    context: str,
) -> None:
    expected_strings = {
        "model.architecture": "qwen2_5_vl",
        "model.name": "qwen_image_qwen2_5_vl_text_encoder",
        "codex.quant_policy": "qwen_image_tenc_mq_v2",
        "codex.quant_policy_preset": "MQ",
        "codex.quant_base_type": "Q8_0",
        "codex.quant_recipe": "Q8_0",
    }
    for key, expected in expected_strings.items():
        actual = str(kv.get(key) or "").strip()
        if actual != expected:
            raise TextEncoderSlotError(
                f"Qwen Image Qwen2.5-VL GGUF metadata mismatch for {context}: "
                f"{key}={actual or '<missing>'!r}, expected {expected!r}."
            )
    expected_ints = {
        "general.file_type": 7,
        "model.attention.head_count": 28,
        "model.attention.head_count_kv": 4,
        "model.block_count": 28,
        "model.context_length": 128000,
        "model.embedding_length": 3584,
    }
    for key, expected in expected_ints.items():
        actual = _metadata_int(kv, key)
        if actual != expected:
            raise TextEncoderSlotError(
                f"Qwen Image Qwen2.5-VL GGUF metadata mismatch for {context}: "
                f"{key}={actual!r}, expected {expected!r}."
            )

    descriptor_mapping = dict(tensor_descriptors)
    try:
        resolved = resolve_qwen2_5_vl_multimodal_keyspace(descriptor_mapping)
    except Exception as exc:
        raise TextEncoderSlotError(
            f"Qwen Image Qwen2.5-VL GGUF tensor topology is invalid for {context}: {exc}"
        ) from exc
    if len(resolved.view) != 728:
        raise TextEncoderSlotError(
            f"Qwen Image Qwen2.5-VL runtime key count mismatch for {context}: "
            f"got={len(resolved.view)} expected=728."
        )

    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        descriptor = tensor_descriptors.get(key)
        if descriptor is None:
            raise TextEncoderSlotError(f"Qwen Image Qwen2.5-VL GGUF is missing {key!r}: {context}")
        if descriptor.tensor_type is not GGMLQuantizationType.Q8_0:
            raise TextEncoderSlotError(
                f"Qwen Image Qwen2.5-VL GGUF requires {key!r} in Q8_0 for {context}; "
                f"got={descriptor.tensor_type.name}."
            )
        if descriptor.n_bytes != 579_059_712:
            raise TextEncoderSlotError(
                f"Qwen Image Qwen2.5-VL GGUF byte-size mismatch for {key!r} in {context}: "
                f"got={descriptor.n_bytes} expected=579059712."
            )


def _classify_gguf_kv(
    kv: Mapping[str, object],
    *,
    tensor_descriptors: Mapping[str, _GGUFTensorDescriptor],
    context: str,
) -> str:
    if _looks_like_mmproj_context(context) or _gguf_is_multimodal_projector(kv):
        raise TextEncoderSlotError(f"GGUF multimodal projector files are not supported text encoder slots: {context}")

    l2p_profile_id = str(kv.get("codex.zimage_l2p.profile_id") or "").strip()
    l2p_component = str(kv.get("codex.zimage_l2p.component") or "").strip()
    l2p_family = str(kv.get("codex.zimage_l2p.family") or "").strip()
    l2p_slot = str(kv.get("codex.zimage_l2p.tenc_slot") or "").strip()
    if l2p_profile_id or l2p_component or l2p_family or l2p_slot:
        if (
            l2p_profile_id == "zimage_l2p_tenc"
            and l2p_component == "tenc"
            and l2p_family == "zimage_l2p"
            and l2p_slot == "qwen3_4b"
        ):
            return "qwen3_4b"
        raise TextEncoderSlotError(
            "Invalid Z-Image L2P GGUF text-encoder metadata for %s "
            "(expected profile_id='zimage_l2p_tenc', component='tenc', family='zimage_l2p', tenc_slot='qwen3_4b')."
            % context
        )

    arch = str(kv.get("general.architecture") or kv.get("model.architecture") or "").strip().lower()
    tok_model = str(kv.get("tokenizer.ggml.model") or "").strip().lower()
    hint = arch or tok_model

    def _resolve_embed_dim() -> int | None:
        return (
            _metadata_int(kv, "llama.embedding_length")
            or _metadata_int(kv, "qwen.embedding_length")
            or _metadata_int(kv, "gemma3.embedding_length")
            or _metadata_int(kv, "gemma.embedding_length")
            or _metadata_int(kv, "model.embedding_length")
            or _metadata_int(kv, "general.embedding_length")
        )

    if "t5" in hint:
        return "t5xxl"
    if "qwen" in hint:
        embed = _resolve_embed_dim()
        if embed is None:
            raise TextEncoderSlotError(
                f"GGUF Qwen slot disambiguation requires an embed dim (expected 2560 or 1024): {context}"
            )
        if embed == 2560:
            return "qwen3_4b"
        if embed == 1024:
            return "qwen3_06b"
        if embed == 3584:
            _validate_qwen_image_qwen2_5_vl_gguf(kv, tensor_descriptors, context=context)
            return "qwen2_5_vl_7b"
        raise TextEncoderSlotError(
            f"Unrecognized GGUF Qwen embed dim {embed} for {context} "
            "(expected 3584 (exact Qwen Image Qwen2.5-VL), 2560 (Qwen3-4B), or 1024 (Qwen3-0.6B))."
        )
    if "gemma" in hint:
        embed = _resolve_embed_dim()
        if embed is None:
            raise TextEncoderSlotError(
                f"GGUF Gemma slot disambiguation requires an embed dim (expected 3840): {context}"
            )
        if embed == 3840:
            return "gemma3_12b"
        raise TextEncoderSlotError(
            f"Unrecognized GGUF Gemma embed dim {embed} for {context} (expected 3840 for Gemma3-12B)."
        )
    if "clip" in hint:
        # GGUF CLIP variants are not slot-disambiguated today; fail if we ever need this.
        raise TextEncoderSlotError(f"GGUF CLIP slot disambiguation is not supported yet: {context}")

    raise TextEncoderSlotError(
        "Could not classify GGUF text encoder slot for %s (missing/unknown architecture metadata)." % context
    )


def _read_safetensors_header(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise TextEncoderSlotError(f"Invalid safetensors header (short): {path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len <= 0 or header_len > 64 * 1024 * 1024:
            raise TextEncoderSlotError(f"Invalid safetensors header length: {header_len} ({path})")
        raw = _read_exact(handle, int(header_len))
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise TextEncoderSlotError(f"Invalid safetensors header JSON: {path}") from exc
    if not isinstance(data, dict):
        raise TextEncoderSlotError(f"Invalid safetensors header (expected JSON object): {path}")
    return data


__all__ = [
    "TextEncoderSlotError",
    "classify_text_encoder_slot",
    "map_text_encoder_paths_to_slots",
]

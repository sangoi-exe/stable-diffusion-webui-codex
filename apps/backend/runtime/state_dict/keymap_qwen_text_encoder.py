"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical key-style detection + explicit source-style mapping for Qwen text-encoder state_dict keys.
Provides the generic Qwen text-backbone resolver plus an exact Qwen Image Edit-2511 multimodal resolver that maps stored
`model.*` keys to the runtime `language_model.*` lookup space, keeps stored `visual.*` keys intact, validates topology and every
logical source shape lazily, and excludes only the separately validated stored `lm_head.weight` tensor.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_qwen_text_encoder_keyspace` (function): Resolves Qwen text-checkpoint source styles into canonical backbone keys (`model.*`) and drops known auxiliary heads.
- `resolve_qwen2_5_vl_multimodal_keyspace` (function): Resolves the exact 729-key Qwen Image Qwen2.5-VL GGUF into the 728-key runtime model view.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping, Sequence
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    KeyMappingError,
    KeySentinel,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    ResolvedKeyspace,
    SentinelKind,
)
from apps.backend.runtime.state_dict.views import KeyspaceLookupView

_T = TypeVar("_T")


_WRAPPER_PREFIXES = (
    "module.",
    "text_encoder.",
    "language_model.",
    "text_model.",
)

_REQUIRED_BACKBONE_KEYS = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.norm.weight",
)

_IGNORED_META_KEYS = frozenset(
    {
        "__metadata__",
    }
)

_DETECTOR = KeyStyleDetector(
    name="qwen_text_encoder_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.HF,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "model."),
                KeySentinel(SentinelKind.PREFIX, "lm_head."),
                KeySentinel(SentinelKind.PREFIX, "visual."),
            ),
            min_sentinel_hits=1,
        ),
    ),
)

_QWEN2_5_VL_EXPECTED_TENSOR_COUNT = 729
_QWEN2_5_VL_LANGUAGE_LAYER_COUNT = 28
_QWEN2_5_VL_VISUAL_BLOCK_COUNT = 32
_QWEN2_5_VL_LANGUAGE_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_QWEN2_5_VL_VISUAL_BLOCK_PATTERN = re.compile(r"^visual\.blocks\.(\d+)\.(.+)$")
_QWEN2_5_VL_TEXT_HIDDEN_SIZE = 3584
_QWEN2_5_VL_TEXT_INTERMEDIATE_SIZE = 18944
_QWEN2_5_VL_TEXT_KV_SIZE = 512
_QWEN2_5_VL_VISUAL_HIDDEN_SIZE = 1280
_QWEN2_5_VL_VISUAL_INTERMEDIATE_SIZE = 3420
_QWEN2_5_VL_VISUAL_QKV_SIZE = 3840
_QWEN2_5_VL_VISUAL_MERGED_SIZE = 5120
_QWEN2_5_VL_LANGUAGE_ROOT_SHAPES = {
    "model.embed_tokens.weight": (152064, 3584),
    "model.norm.weight": (3584,),
}
_QWEN2_5_VL_VISUAL_ROOT_SHAPES = {
    "visual.merger.ln_q.weight": (_QWEN2_5_VL_VISUAL_HIDDEN_SIZE,),
    "visual.merger.mlp.0.bias": (_QWEN2_5_VL_VISUAL_MERGED_SIZE,),
    "visual.merger.mlp.0.weight": (
        _QWEN2_5_VL_VISUAL_MERGED_SIZE,
        _QWEN2_5_VL_VISUAL_MERGED_SIZE,
    ),
    "visual.merger.mlp.2.bias": (_QWEN2_5_VL_TEXT_HIDDEN_SIZE,),
    "visual.merger.mlp.2.weight": (
        _QWEN2_5_VL_TEXT_HIDDEN_SIZE,
        _QWEN2_5_VL_VISUAL_MERGED_SIZE,
    ),
    "visual.patch_embed.proj.weight": (1280, 3, 2, 14, 14),
}
_QWEN2_5_VL_LANGUAGE_LAYER_SUFFIX_SHAPES = {
    "input_layernorm.weight": (_QWEN2_5_VL_TEXT_HIDDEN_SIZE,),
    "mlp.down_proj.weight": (
        _QWEN2_5_VL_TEXT_HIDDEN_SIZE,
        _QWEN2_5_VL_TEXT_INTERMEDIATE_SIZE,
    ),
    "mlp.gate_proj.weight": (
        _QWEN2_5_VL_TEXT_INTERMEDIATE_SIZE,
        _QWEN2_5_VL_TEXT_HIDDEN_SIZE,
    ),
    "mlp.up_proj.weight": (
        _QWEN2_5_VL_TEXT_INTERMEDIATE_SIZE,
        _QWEN2_5_VL_TEXT_HIDDEN_SIZE,
    ),
    "post_attention_layernorm.weight": (_QWEN2_5_VL_TEXT_HIDDEN_SIZE,),
    "self_attn.k_proj.bias": (_QWEN2_5_VL_TEXT_KV_SIZE,),
    "self_attn.k_proj.weight": (
        _QWEN2_5_VL_TEXT_KV_SIZE,
        _QWEN2_5_VL_TEXT_HIDDEN_SIZE,
    ),
    "self_attn.o_proj.weight": (
        _QWEN2_5_VL_TEXT_HIDDEN_SIZE,
        _QWEN2_5_VL_TEXT_HIDDEN_SIZE,
    ),
    "self_attn.q_proj.bias": (_QWEN2_5_VL_TEXT_HIDDEN_SIZE,),
    "self_attn.q_proj.weight": (
        _QWEN2_5_VL_TEXT_HIDDEN_SIZE,
        _QWEN2_5_VL_TEXT_HIDDEN_SIZE,
    ),
    "self_attn.v_proj.bias": (_QWEN2_5_VL_TEXT_KV_SIZE,),
    "self_attn.v_proj.weight": (
        _QWEN2_5_VL_TEXT_KV_SIZE,
        _QWEN2_5_VL_TEXT_HIDDEN_SIZE,
    ),
}
_QWEN2_5_VL_VISUAL_BLOCK_SUFFIX_SHAPES = {
    "attn.proj.bias": (_QWEN2_5_VL_VISUAL_HIDDEN_SIZE,),
    "attn.proj.weight": (
        _QWEN2_5_VL_VISUAL_HIDDEN_SIZE,
        _QWEN2_5_VL_VISUAL_HIDDEN_SIZE,
    ),
    "attn.qkv.bias": (_QWEN2_5_VL_VISUAL_QKV_SIZE,),
    "attn.qkv.weight": (
        _QWEN2_5_VL_VISUAL_QKV_SIZE,
        _QWEN2_5_VL_VISUAL_HIDDEN_SIZE,
    ),
    "mlp.down_proj.bias": (_QWEN2_5_VL_VISUAL_HIDDEN_SIZE,),
    "mlp.down_proj.weight": (
        _QWEN2_5_VL_VISUAL_HIDDEN_SIZE,
        _QWEN2_5_VL_VISUAL_INTERMEDIATE_SIZE,
    ),
    "mlp.gate_proj.bias": (_QWEN2_5_VL_VISUAL_INTERMEDIATE_SIZE,),
    "mlp.gate_proj.weight": (
        _QWEN2_5_VL_VISUAL_INTERMEDIATE_SIZE,
        _QWEN2_5_VL_VISUAL_HIDDEN_SIZE,
    ),
    "mlp.up_proj.bias": (_QWEN2_5_VL_VISUAL_INTERMEDIATE_SIZE,),
    "mlp.up_proj.weight": (
        _QWEN2_5_VL_VISUAL_INTERMEDIATE_SIZE,
        _QWEN2_5_VL_VISUAL_HIDDEN_SIZE,
    ),
    "norm1.weight": (_QWEN2_5_VL_VISUAL_HIDDEN_SIZE,),
    "norm2.weight": (_QWEN2_5_VL_VISUAL_HIDDEN_SIZE,),
}
_QWEN2_5_VL_AUX_SHAPES = {
    "lm_head.weight": (152064, 3584),
}
_QWEN2_5_VL_LANGUAGE_ROOT_KEYS = frozenset(_QWEN2_5_VL_LANGUAGE_ROOT_SHAPES)
_QWEN2_5_VL_VISUAL_ROOT_KEYS = frozenset(_QWEN2_5_VL_VISUAL_ROOT_SHAPES)
_QWEN2_5_VL_LANGUAGE_LAYER_SUFFIXES = frozenset(_QWEN2_5_VL_LANGUAGE_LAYER_SUFFIX_SHAPES)
_QWEN2_5_VL_VISUAL_BLOCK_SUFFIXES = frozenset(_QWEN2_5_VL_VISUAL_BLOCK_SUFFIX_SHAPES)
_QWEN2_5_VL_AUX_KEYS = frozenset(_QWEN2_5_VL_AUX_SHAPES)


def _is_supported_qwen_root_key(key: str) -> bool:
    return key in _IGNORED_META_KEYS or key.startswith(("model.", "lm_head.", "visual."))


def _map_source_key_to_backbone_key(key: str) -> str:
    source_key = str(key)
    if _is_supported_qwen_root_key(source_key):
        return source_key
    for wrapper_prefix in _WRAPPER_PREFIXES:
        if source_key.startswith(wrapper_prefix):
            candidate_key = source_key[len(wrapper_prefix) :]
            if _is_supported_qwen_root_key(candidate_key):
                return candidate_key
            break
    return source_key


def _validate_required_backbone_keys(keys: Sequence[str], *, context: str) -> None:
    missing = [key for key in _REQUIRED_BACKBONE_KEYS if key not in keys]
    if missing:
        preview = ", ".join(sorted(keys)[:10])
        raise KeyMappingError(
            f"{context}: missing required Qwen backbone keys: {missing}. sample_keys=[{preview}]"
        )


def resolve_qwen_text_encoder_keyspace(
    state_dict: MutableMapping[str, _T],
    *,
    allow_lm_head_aux: bool = True,
    allow_visual_aux: bool = True,
    require_backbone_keys: bool = True,
) -> ResolvedKeyspace[_T]:
    """Resolve Qwen text-encoder source styles into canonical backbone keys.

    Supported upstream styles:
    - HF: `model.*` (plus optional `lm_head.*`, `visual.*`)
    - Wrapped HF: `module.*`, `text_encoder.*`, `language_model.*`, `text_model.*`

    Resolver behavior:
    - Keeps canonical backbone weights under `model.*`.
    - Drops known auxiliary heads (`lm_head.*`, optional `visual.*`).
    - Ignores known metadata-only sentinel keys (currently `__metadata__`).
    - Raises on unknown keys, ambiguous style detection, key collisions, or missing required backbone keys.
    """

    keys = [str(key) for key in state_dict.keys()]
    if not keys:
        raise KeyMappingError("qwen_text_encoder_key_style: empty key list; cannot detect key style")

    lookup_keys_for_detection: list[str] = []
    for key in keys:
        lookup_key = _map_source_key_to_backbone_key(key)
        if lookup_key in _IGNORED_META_KEYS:
            continue
        lookup_keys_for_detection.append(lookup_key)
    if not lookup_keys_for_detection:
        raise KeyMappingError(
            "qwen_text_encoder_key_style: no tensor keys remained after metadata filtering; cannot detect key style"
        )
    style = _DETECTOR.detect(lookup_keys_for_detection)

    canonical_to_source: dict[str, str] = {}
    unsupported_keys: list[str] = []
    for source_key in keys:
        lookup_key = _map_source_key_to_backbone_key(source_key)
        if lookup_key in _IGNORED_META_KEYS:
            continue
        if lookup_key.startswith("model."):
            previous = canonical_to_source.get(lookup_key)
            if previous is not None and previous != source_key:
                raise KeyMappingError(
                    "qwen_text_encoder_key_style: multiple source keys map to the same destination key: "
                    f"dst={lookup_key!r} srcs={previous!r},{source_key!r}"
                )
            canonical_to_source[lookup_key] = source_key
            continue

        if lookup_key.startswith("lm_head."):
            if not allow_lm_head_aux:
                unsupported_keys.append(lookup_key)
            continue

        if lookup_key.startswith("visual."):
            if not allow_visual_aux:
                unsupported_keys.append(lookup_key)
            continue

        unsupported_keys.append(lookup_key)

    if unsupported_keys:
        sample = ", ".join(unsupported_keys[:10])
        raise KeyMappingError(
            "qwen_text_encoder_key_style: unsupported source keys after explicit source-style mapping. "
            "Allowed destinations are `model.*` plus optional aux branches "
            f"`lm_head.*`/`visual.*`. offenders_sample=[{sample}]"
        )

    canonical_keys = list(canonical_to_source.keys())
    if not canonical_keys:
        raise KeyMappingError(
            "qwen_text_encoder_key_style: no canonical backbone keys (`model.*`) were produced after explicit source-style mapping"
        )

    if require_backbone_keys:
        _validate_required_backbone_keys(
            canonical_keys,
            context="qwen_text_encoder_key_style",
        )

    return ResolvedKeyspace(
        style=style,
        canonical_to_source=canonical_to_source,
        metadata={
            "resolver": "qwen_text_encoder",
            "allow_lm_head_aux": bool(allow_lm_head_aux),
            "allow_visual_aux": bool(allow_visual_aux),
            "require_backbone_keys": bool(require_backbone_keys),
        },
        view=KeyspaceLookupView(state_dict, canonical_to_source),
    )


def _logical_shape(state_dict: MutableMapping[str, _T], key: str) -> tuple[int, ...] | None:
    shape_getter = getattr(state_dict, "shape_of", None)
    if callable(shape_getter):
        try:
            shape = shape_getter(key)
        except Exception as exc:
            raise KeyMappingError(
                "qwen2_5_vl_multimodal: lazy logical-shape inspection failed. "
                f"key={key!r}"
            ) from exc
        if shape is None:
            return None
        return tuple(int(dim) for dim in shape)
    try:
        value = state_dict[key]
    except Exception:
        return None
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _validate_exact_qwen2_5_vl_keyspace(state_dict: MutableMapping[str, _T]) -> list[str]:
    raw_keys = list(state_dict.keys())
    non_string = [type(key).__name__ for key in raw_keys if not isinstance(key, str)]
    if non_string:
        raise KeyMappingError(
            "qwen2_5_vl_multimodal: every stored tensor key must be a string; "
            f"offender_types={non_string[:10]}"
        )
    keys = [str(key) for key in raw_keys]
    if len(keys) != _QWEN2_5_VL_EXPECTED_TENSOR_COUNT:
        raise KeyMappingError(
            "qwen2_5_vl_multimodal: exact Qwen Image text-encoder tensor count mismatch. "
            f"got={len(keys)} expected={_QWEN2_5_VL_EXPECTED_TENSOR_COUNT}"
        )

    language_roots: set[str] = set()
    visual_roots: set[str] = set()
    aux_keys: set[str] = set()
    language_layers: dict[int, set[str]] = {}
    visual_blocks: dict[int, set[str]] = {}
    unsupported: list[str] = []
    for key in keys:
        if key in _QWEN2_5_VL_LANGUAGE_ROOT_KEYS:
            language_roots.add(key)
            continue
        if key in _QWEN2_5_VL_VISUAL_ROOT_KEYS:
            visual_roots.add(key)
            continue
        if key in _QWEN2_5_VL_AUX_KEYS:
            aux_keys.add(key)
            continue

        language_match = _QWEN2_5_VL_LANGUAGE_LAYER_PATTERN.fullmatch(key)
        if language_match is not None:
            layer_index = int(language_match.group(1))
            suffix = language_match.group(2)
            if (
                layer_index < 0
                or layer_index >= _QWEN2_5_VL_LANGUAGE_LAYER_COUNT
                or suffix not in _QWEN2_5_VL_LANGUAGE_LAYER_SUFFIXES
            ):
                unsupported.append(key)
            else:
                language_layers.setdefault(layer_index, set()).add(suffix)
            continue

        visual_match = _QWEN2_5_VL_VISUAL_BLOCK_PATTERN.fullmatch(key)
        if visual_match is not None:
            block_index = int(visual_match.group(1))
            suffix = visual_match.group(2)
            if (
                block_index < 0
                or block_index >= _QWEN2_5_VL_VISUAL_BLOCK_COUNT
                or suffix not in _QWEN2_5_VL_VISUAL_BLOCK_SUFFIXES
            ):
                unsupported.append(key)
            else:
                visual_blocks.setdefault(block_index, set()).add(suffix)
            continue
        unsupported.append(key)

    if unsupported:
        raise KeyMappingError(
            "qwen2_5_vl_multimodal: unsupported stored keys; wrapper prefixes and alternate layouts are not accepted. "
            f"offenders_sample={sorted(unsupported)[:10]}"
        )
    for label, found, expected in (
        ("language roots", language_roots, _QWEN2_5_VL_LANGUAGE_ROOT_KEYS),
        ("visual roots", visual_roots, _QWEN2_5_VL_VISUAL_ROOT_KEYS),
        ("auxiliary keys", aux_keys, _QWEN2_5_VL_AUX_KEYS),
    ):
        if found != expected:
            raise KeyMappingError(
                f"qwen2_5_vl_multimodal: {label} mismatch. "
                f"missing={sorted(expected - found)} extra={sorted(found - expected)}"
            )

    expected_language_indices = set(range(_QWEN2_5_VL_LANGUAGE_LAYER_COUNT))
    if set(language_layers) != expected_language_indices:
        raise KeyMappingError(
            "qwen2_5_vl_multimodal: language layer index set mismatch. "
            f"missing={sorted(expected_language_indices - set(language_layers))} "
            f"extra={sorted(set(language_layers) - expected_language_indices)}"
        )
    for layer_index in range(_QWEN2_5_VL_LANGUAGE_LAYER_COUNT):
        found = language_layers[layer_index]
        if found != _QWEN2_5_VL_LANGUAGE_LAYER_SUFFIXES:
            raise KeyMappingError(
                "qwen2_5_vl_multimodal: language layer tensor topology mismatch. "
                f"layer={layer_index} missing={sorted(_QWEN2_5_VL_LANGUAGE_LAYER_SUFFIXES - found)} "
                f"extra={sorted(found - _QWEN2_5_VL_LANGUAGE_LAYER_SUFFIXES)}"
            )

    expected_visual_indices = set(range(_QWEN2_5_VL_VISUAL_BLOCK_COUNT))
    if set(visual_blocks) != expected_visual_indices:
        raise KeyMappingError(
            "qwen2_5_vl_multimodal: visual block index set mismatch. "
            f"missing={sorted(expected_visual_indices - set(visual_blocks))} "
            f"extra={sorted(set(visual_blocks) - expected_visual_indices)}"
        )
    for block_index in range(_QWEN2_5_VL_VISUAL_BLOCK_COUNT):
        found = visual_blocks[block_index]
        if found != _QWEN2_5_VL_VISUAL_BLOCK_SUFFIXES:
            raise KeyMappingError(
                "qwen2_5_vl_multimodal: visual block tensor topology mismatch. "
                f"block={block_index} missing={sorted(_QWEN2_5_VL_VISUAL_BLOCK_SUFFIXES - found)} "
                f"extra={sorted(found - _QWEN2_5_VL_VISUAL_BLOCK_SUFFIXES)}"
            )

    for key in keys:
        expected_shape = _QWEN2_5_VL_LANGUAGE_ROOT_SHAPES.get(key)
        if expected_shape is None:
            expected_shape = _QWEN2_5_VL_VISUAL_ROOT_SHAPES.get(key)
        if expected_shape is None:
            expected_shape = _QWEN2_5_VL_AUX_SHAPES.get(key)
        if expected_shape is None:
            language_match = _QWEN2_5_VL_LANGUAGE_LAYER_PATTERN.fullmatch(key)
            if language_match is not None:
                expected_shape = _QWEN2_5_VL_LANGUAGE_LAYER_SUFFIX_SHAPES.get(
                    language_match.group(2)
                )
        if expected_shape is None:
            visual_match = _QWEN2_5_VL_VISUAL_BLOCK_PATTERN.fullmatch(key)
            if visual_match is not None:
                expected_shape = _QWEN2_5_VL_VISUAL_BLOCK_SUFFIX_SHAPES.get(
                    visual_match.group(2)
                )
        if expected_shape is None:
            raise KeyMappingError(
                "qwen2_5_vl_multimodal: validated topology has no logical-shape contract. "
                f"key={key!r}"
            )
        actual_shape = _logical_shape(state_dict, key)
        if actual_shape != expected_shape:
            raise KeyMappingError(
                "qwen2_5_vl_multimodal: source logical shape mismatch. "
                f"key={key!r} got={actual_shape!r} expected={expected_shape!r}"
            )
    return keys


def resolve_qwen2_5_vl_multimodal_keyspace(
    state_dict: MutableMapping[str, _T],
) -> ResolvedKeyspace[_T]:
    """Resolve the exact Qwen Image Qwen2.5-VL GGUF into the runtime model lookup space."""

    keys = _validate_exact_qwen2_5_vl_keyspace(state_dict)
    canonical_to_source: dict[str, str] = {}
    for source_key in keys:
        if source_key in _QWEN2_5_VL_AUX_KEYS:
            continue
        destination_key = (
            f"language_model.{source_key[len('model.') :]}"
            if source_key.startswith("model.")
            else source_key
        )
        previous = canonical_to_source.get(destination_key)
        if previous is not None:
            raise KeyMappingError(
                "qwen2_5_vl_multimodal: multiple stored keys map to the same runtime lookup key. "
                f"destination={destination_key!r} sources={previous!r},{source_key!r}"
            )
        canonical_to_source[destination_key] = source_key

    expected_runtime_count = _QWEN2_5_VL_EXPECTED_TENSOR_COUNT - len(_QWEN2_5_VL_AUX_KEYS)
    if len(canonical_to_source) != expected_runtime_count:
        raise KeyMappingError(
            "qwen2_5_vl_multimodal: runtime key count mismatch after excluding lm_head.weight. "
            f"got={len(canonical_to_source)} expected={expected_runtime_count}"
        )
    return ResolvedKeyspace(
        style=KeyStyle.HF,
        canonical_to_source=canonical_to_source,
        metadata={
            "resolver": "qwen2_5_vl_multimodal",
            "source_keys": len(keys),
            "canonical_keys": len(canonical_to_source),
            "language_layers": _QWEN2_5_VL_LANGUAGE_LAYER_COUNT,
            "visual_blocks": _QWEN2_5_VL_VISUAL_BLOCK_COUNT,
            "excluded_auxiliary_keys": tuple(sorted(_QWEN2_5_VL_AUX_KEYS)),
            "validated_source_shapes": len(keys),
        },
        view=KeyspaceLookupView(state_dict, canonical_to_source),
    )


__all__ = [
    "resolve_qwen2_5_vl_multimodal_keyspace",
    "resolve_qwen_text_encoder_keyspace",
]

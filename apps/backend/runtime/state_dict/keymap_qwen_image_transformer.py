"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Exact native-keyspace resolver for the Qwen Image Edit-2511 transformer GGUF.
Validates the 1,933 stored tensor names, 60-block topology, and every executable logical shape before exposing an identity
`KeyspaceLookupView`; stored checkpoint keys are never renamed, stripped, normalized, or materialized into a remapped state dict.

Symbols (top-level; keep in sync; no ghosts):
- `resolve_qwen_image_edit_transformer_keyspace` (function): Validate and expose the exact Edit-2511 transformer keyspace lazily.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import KeyMappingError, KeyStyle, ResolvedKeyspace
from apps.backend.runtime.state_dict.views import KeyspaceLookupView

_T = TypeVar("_T")

_EXPECTED_TENSOR_COUNT = 1933
_EXPECTED_BLOCK_COUNT = 60
_BLOCK_PATTERN = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
_MODEL_DIM = 3072
_ATTENTION_HEAD_DIM = 128
_FEED_FORWARD_DIM = 12288
_MODULATION_DIM = 18432
_NON_BLOCK_SHAPES = {
    "img_in.weight": (3072, 64),
    "img_in.bias": (3072,),
    "txt_in.weight": (3072, 3584),
    "txt_in.bias": (3072,),
    "txt_norm.weight": (3584,),
    "norm_out.linear.weight": (6144, 3072),
    "norm_out.linear.bias": (6144,),
    "proj_out.weight": (64, 3072),
    "proj_out.bias": (64,),
    "time_text_embed.timestep_embedder.linear_1.weight": (3072, 256),
    "time_text_embed.timestep_embedder.linear_1.bias": (3072,),
    "time_text_embed.timestep_embedder.linear_2.weight": (3072, 3072),
    "time_text_embed.timestep_embedder.linear_2.bias": (3072,),
}
_BLOCK_SUFFIX_SHAPES = {
    "attn.add_k_proj.bias": (_MODEL_DIM,),
    "attn.add_k_proj.weight": (_MODEL_DIM, _MODEL_DIM),
    "attn.add_q_proj.bias": (_MODEL_DIM,),
    "attn.add_q_proj.weight": (_MODEL_DIM, _MODEL_DIM),
    "attn.add_v_proj.bias": (_MODEL_DIM,),
    "attn.add_v_proj.weight": (_MODEL_DIM, _MODEL_DIM),
    "attn.norm_added_k.weight": (_ATTENTION_HEAD_DIM,),
    "attn.norm_added_q.weight": (_ATTENTION_HEAD_DIM,),
    "attn.norm_k.weight": (_ATTENTION_HEAD_DIM,),
    "attn.norm_q.weight": (_ATTENTION_HEAD_DIM,),
    "attn.to_add_out.bias": (_MODEL_DIM,),
    "attn.to_add_out.weight": (_MODEL_DIM, _MODEL_DIM),
    "attn.to_k.bias": (_MODEL_DIM,),
    "attn.to_k.weight": (_MODEL_DIM, _MODEL_DIM),
    "attn.to_out.0.bias": (_MODEL_DIM,),
    "attn.to_out.0.weight": (_MODEL_DIM, _MODEL_DIM),
    "attn.to_q.bias": (_MODEL_DIM,),
    "attn.to_q.weight": (_MODEL_DIM, _MODEL_DIM),
    "attn.to_v.bias": (_MODEL_DIM,),
    "attn.to_v.weight": (_MODEL_DIM, _MODEL_DIM),
    "img_mlp.net.0.proj.bias": (_FEED_FORWARD_DIM,),
    "img_mlp.net.0.proj.weight": (_FEED_FORWARD_DIM, _MODEL_DIM),
    "img_mlp.net.2.bias": (_MODEL_DIM,),
    "img_mlp.net.2.weight": (_MODEL_DIM, _FEED_FORWARD_DIM),
    "img_mod.1.bias": (_MODULATION_DIM,),
    "img_mod.1.weight": (_MODULATION_DIM, _MODEL_DIM),
    "txt_mlp.net.0.proj.bias": (_FEED_FORWARD_DIM,),
    "txt_mlp.net.0.proj.weight": (_FEED_FORWARD_DIM, _MODEL_DIM),
    "txt_mlp.net.2.bias": (_MODEL_DIM,),
    "txt_mlp.net.2.weight": (_MODEL_DIM, _FEED_FORWARD_DIM),
    "txt_mod.1.bias": (_MODULATION_DIM,),
    "txt_mod.1.weight": (_MODULATION_DIM, _MODEL_DIM),
}
_NON_BLOCK_KEYS = frozenset(_NON_BLOCK_SHAPES)
_BLOCK_SUFFIXES = frozenset(_BLOCK_SUFFIX_SHAPES)


def _logical_shape(state_dict: MutableMapping[str, _T], key: str) -> tuple[int, ...] | None:
    shape_getter = getattr(state_dict, "shape_of", None)
    if callable(shape_getter):
        try:
            shape = shape_getter(key)
        except Exception as exc:
            raise KeyMappingError(
                "qwen_image_edit_transformer: lazy logical-shape inspection failed. "
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


def _validate_exact_keyspace(state_dict: MutableMapping[str, _T]) -> list[str]:
    raw_keys = list(state_dict.keys())
    non_string = [type(key).__name__ for key in raw_keys if not isinstance(key, str)]
    if non_string:
        raise KeyMappingError(
            "qwen_image_edit_transformer: every stored tensor key must be a string; "
            f"offender_types={non_string[:10]}"
        )
    keys = [str(key) for key in raw_keys]
    if len(keys) != _EXPECTED_TENSOR_COUNT:
        raise KeyMappingError(
            "qwen_image_edit_transformer: exact Edit-2511 tensor count mismatch. "
            f"got={len(keys)} expected={_EXPECTED_TENSOR_COUNT}"
        )

    seen_non_block: set[str] = set()
    block_suffixes: dict[int, set[str]] = {}
    unsupported: list[str] = []
    for key in keys:
        if key in _NON_BLOCK_KEYS:
            seen_non_block.add(key)
            continue
        match = _BLOCK_PATTERN.fullmatch(key)
        if match is None:
            unsupported.append(key)
            continue
        block_index = int(match.group(1))
        suffix = match.group(2)
        if block_index < 0 or block_index >= _EXPECTED_BLOCK_COUNT or suffix not in _BLOCK_SUFFIXES:
            unsupported.append(key)
            continue
        block_suffixes.setdefault(block_index, set()).add(suffix)

    if unsupported:
        raise KeyMappingError(
            "qwen_image_edit_transformer: unsupported stored keys; wrapper prefixes and alternate layouts are not accepted. "
            f"offenders_sample={sorted(unsupported)[:10]}"
        )
    missing_non_block = sorted(_NON_BLOCK_KEYS.difference(seen_non_block))
    if missing_non_block:
        raise KeyMappingError(
            "qwen_image_edit_transformer: missing required non-block tensors. "
            f"missing={missing_non_block}"
        )

    expected_indices = set(range(_EXPECTED_BLOCK_COUNT))
    found_indices = set(block_suffixes)
    if found_indices != expected_indices:
        raise KeyMappingError(
            "qwen_image_edit_transformer: transformer block index set mismatch. "
            f"missing={sorted(expected_indices - found_indices)} extra={sorted(found_indices - expected_indices)}"
        )
    for block_index in range(_EXPECTED_BLOCK_COUNT):
        found = block_suffixes[block_index]
        if found != _BLOCK_SUFFIXES:
            raise KeyMappingError(
                "qwen_image_edit_transformer: block tensor topology mismatch. "
                f"block={block_index} missing={sorted(_BLOCK_SUFFIXES - found)} extra={sorted(found - _BLOCK_SUFFIXES)}"
            )

    for key in keys:
        expected_shape = _NON_BLOCK_SHAPES.get(key)
        if expected_shape is None:
            match = _BLOCK_PATTERN.fullmatch(key)
            if match is None:
                raise KeyMappingError(
                    "qwen_image_edit_transformer: validated topology has no logical-shape owner. "
                    f"key={key!r}"
                )
            expected_shape = _BLOCK_SUFFIX_SHAPES.get(match.group(2))
        if expected_shape is None:
            raise KeyMappingError(
                "qwen_image_edit_transformer: validated topology has no logical-shape contract. "
                f"key={key!r}"
            )
        actual_shape = _logical_shape(state_dict, key)
        if actual_shape != expected_shape:
            raise KeyMappingError(
                "qwen_image_edit_transformer: executable logical shape mismatch. "
                f"key={key!r} got={actual_shape!r} expected={expected_shape!r}"
            )
    return keys


def resolve_qwen_image_edit_transformer_keyspace(
    state_dict: MutableMapping[str, _T],
) -> ResolvedKeyspace[_T]:
    """Validate and expose the exact Edit-2511 transformer keyspace without changing stored names."""

    keys = _validate_exact_keyspace(state_dict)
    canonical_to_source = {key: key for key in keys}
    return ResolvedKeyspace(
        style=KeyStyle.DIFFUSERS,
        canonical_to_source=canonical_to_source,
        metadata={
            "resolver": "qwen_image_edit_transformer",
            "variant": "edit_2511",
            "source_keys": len(keys),
            "canonical_keys": len(canonical_to_source),
            "block_count": _EXPECTED_BLOCK_COUNT,
            "validated_logical_shapes": len(keys),
        },
        view=KeyspaceLookupView(state_dict, canonical_to_source),
    )


__all__ = ["resolve_qwen_image_edit_transformer_keyspace"]

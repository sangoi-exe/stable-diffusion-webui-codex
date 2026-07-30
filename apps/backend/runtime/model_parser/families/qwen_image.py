"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Parser plan builder for the exact core-only Qwen Image Edit-2511 transformer GGUF.
Keeps the whole native checkpoint under the transformer component, validates it through the canonical lazy keymap, and
registers the single external `qwen2_5_vl_7b` text-encoder alias for deterministic override resolution.

Symbols (top-level; keep in sync; no ghosts):
- `_register_qwen_image_text_encoder` (function): Register the exact external Qwen2.5-VL text-encoder alias.
- `_validate_qwen_image_transformer` (function): Validate the native Edit-2511 transformer keyspace.
- `build_plan` (function): Build the Qwen Image Edit-2511 `ParserPlanBundle`.
"""

from __future__ import annotations

from apps.backend.runtime.model_registry.specs import ModelSignature
from apps.backend.runtime.state_dict.keymap_qwen_image_transformer import (
    resolve_qwen_image_edit_transformer_keyspace,
)

from ..builders import build_estimated_config, register_text_encoder
from ..errors import ValidationError
from ..quantization import validate_component_dtypes
from ..specs import ParserPlan, ParserPlanBundle, SplitSpec, ValidationSpec


def _register_qwen_image_text_encoder(context) -> None:  # type: ignore[no-untyped-def]
    register_text_encoder(context, "qwen2_5_vl_7b", "text_encoder")


def _validate_qwen_image_transformer(context) -> None:  # type: ignore[no-untyped-def]
    raw_transformer = context.require("transformer").tensors
    try:
        resolve_qwen_image_edit_transformer_keyspace(raw_transformer)
    except Exception as exc:
        raise ValidationError(
            f"Qwen Image Edit-2511 transformer keyspace resolution failed: {exc}",
            component="transformer",
        ) from exc


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    plan = ParserPlan(
        splits=[
            SplitSpec(name="transformer", prefixes=("",)),
        ],
        converters=(),
        validations=(
            ValidationSpec(name="register_qwen_image_text_encoder", function=_register_qwen_image_text_encoder),
            ValidationSpec(name="qwen_image_transformer", function=_validate_qwen_image_transformer),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda context: build_estimated_config(context, signature))


__all__ = ["build_plan"]

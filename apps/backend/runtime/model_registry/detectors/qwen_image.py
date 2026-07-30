"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Exact core-only Qwen Image Edit-2511 GGUF detector for the Codex model registry.
Requires the dedicated converter metadata profile plus the native 1,933-key transformer topology, rejects embedded or alternate
layouts, and emits an Edit-only signature with explicit external Qwen2.5-VL and VAE requirements.

Symbols (top-level; keep in sync; no ghosts):
- `validate_qwen_image_edit_gguf_metadata` (function): Validate the exact Edit-2511 transformer GGUF metadata contract.
- `qwen_image_edit_gguf_metadata_valid` (function): Boolean wrapper around exact transformer metadata validation.
- `QwenImageDetector` (class): Detect and describe the exact core-only Edit-2511 transformer GGUF.
"""

from __future__ import annotations

from typing import Any, Mapping

from apps.backend.runtime.model_registry.detectors.base import ModelDetector, REGISTRY
from apps.backend.runtime.model_registry.signals import SignalBundle
from apps.backend.runtime.model_registry.specs import (
    CodexCoreArchitecture,
    CodexCoreSignature,
    LatentFormat,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationHint,
    QuantizationKind,
    TextEncoderSignature,
)
from apps.backend.runtime.state_dict.keymap_qwen_image_transformer import (
    resolve_qwen_image_edit_transformer_keyspace,
)

_EXPECTED_STRING_METADATA = {
    "model.architecture": "qwen_image",
    "model.name": "Qwen/Qwen-Image-Edit-2511",
    "codex.quant_policy": "qwen_image_mq_v2",
    "codex.quant_policy_preset": "MQ",
    "codex.qwen_image.variant": "edit_2511",
}
_EXPECTED_INTEGER_METADATA = {
    "GGUF.version": 3,
    "GGUF.tensor_count": 1933,
    "model.attention.head_count": 24,
    "model.attention.head_count_kv": 24,
    "model.block_count": 60,
    "model.context_length": 4096,
    "model.embedding_length": 3072,
    "codex.qwen_image.in_channels": 64,
    "codex.qwen_image.joint_attention_dim": 3584,
    "codex.qwen_image.out_channels": 16,
    "codex.qwen_image.patch_size": 2,
}


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int | None:
    raw = metadata.get(key)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, float) and float(raw).is_integer():
        return int(raw)
    return None


def validate_qwen_image_edit_gguf_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate the dedicated Edit-2511 transformer GGUF metadata profile."""

    if not isinstance(metadata, Mapping):
        raise ValueError("Qwen Image Edit-2511 GGUF metadata must be a mapping.")
    for key, expected in _EXPECTED_STRING_METADATA.items():
        actual = str(metadata.get(key) or "").strip()
        if actual != expected:
            raise ValueError(
                f"Qwen Image Edit-2511 GGUF metadata mismatch for {key!r}: "
                f"got={actual or '<missing>'!r} expected={expected!r}."
            )
    for key, expected in _EXPECTED_INTEGER_METADATA.items():
        actual = _metadata_int(metadata, key)
        if actual != expected:
            raise ValueError(
                f"Qwen Image Edit-2511 GGUF metadata mismatch for {key!r}: "
                f"got={actual!r} expected={expected!r}."
            )
    if metadata.get("codex.qwen_image.zero_cond_t") is not True:
        raise ValueError("Qwen Image Edit-2511 GGUF metadata requires codex.qwen_image.zero_cond_t=true.")
    axes = metadata.get("codex.qwen_image.axes_dims_rope")
    if not isinstance(axes, (list, tuple)) or tuple(int(value) for value in axes) != (16, 56, 56):
        raise ValueError(
            "Qwen Image Edit-2511 GGUF metadata requires codex.qwen_image.axes_dims_rope=[16, 56, 56]."
        )


def qwen_image_edit_gguf_metadata_valid(metadata: Mapping[str, Any] | None) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    try:
        validate_qwen_image_edit_gguf_metadata(metadata)
    except Exception:
        return False
    return True


class QwenImageDetector(ModelDetector):
    priority = 182

    def matches(self, bundle: SignalBundle) -> bool:  # type: ignore[override]
        if str(bundle.source_format or "").strip().lower() != "gguf":
            return False
        if not qwen_image_edit_gguf_metadata_valid(bundle.metadata):
            return False
        try:
            resolve_qwen_image_edit_transformer_keyspace(bundle.state_dict)  # type: ignore[arg-type]
        except Exception:
            return False
        return True

    def build_signature(self, bundle: SignalBundle) -> ModelSignature:  # type: ignore[override]
        validate_qwen_image_edit_gguf_metadata(bundle.metadata or {})
        resolve_qwen_image_edit_transformer_keyspace(bundle.state_dict)  # type: ignore[arg-type]
        return ModelSignature(
            family=ModelFamily.QWEN_IMAGE,
            repo_hint="Qwen/Qwen-Image-Edit-2511",
            prediction=PredictionKind.FLOW,
            latent_format=LatentFormat.QWEN_IMAGE,
            quantization=QuantizationHint(
                kind=QuantizationKind.GGUF,
                detail="qwen_image_mq_v2",
            ),
            core=CodexCoreSignature(
                architecture=CodexCoreArchitecture.FLOW_TRANSFORMER,
                channels_in=64,
                channels_out=16,
                context_dim=3584,
                temporal=False,
                depth=60,
                key_prefixes=["transformer_blocks."],
            ),
            text_encoders=[
                TextEncoderSignature(
                    name="qwen2_5_vl_7b",
                    key_prefix="text_encoder.",
                    expected_dim=3584,
                    tokenizer_hint="Qwen/Qwen-Image-Edit-2511/processor",
                )
            ],
            vae=None,
            extras={
                "core_only": True,
                "requires_vae": True,
                "qwen_image_variant": "edit_2511",
                "zero_cond_t": True,
                "signature_source": "qwen_image_edit_2511_gguf_profile",
            },
        )


REGISTRY.register(QwenImageDetector())


__all__ = [
    "QwenImageDetector",
    "qwen_image_edit_gguf_metadata_valid",
    "validate_qwen_image_edit_gguf_metadata",
]

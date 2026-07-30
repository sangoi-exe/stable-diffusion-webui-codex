"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image Edit-2511 VAE metadata validation and latent normalization helpers.
Owns the vendored `AutoencoderKLQwenImage` config contract, exact external SafeTensors header identity, and per-channel
latent mean/std math used by Qwen Image encode/decode seams without loading tensor payloads during admission.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageVaeConfig` (dataclass): Strict metadata contract for `AutoencoderKLQwenImage`.
- `qwen_image_denormalize_latents` (function): Apply inverse per-channel Qwen Image VAE latent normalization.
- `qwen_image_normalize_latents` (function): Apply per-channel Qwen Image VAE latent normalization.
- `qwen_image_validate_external_vae_path` (function): Validate a selected external Qwen Image VAE asset path/root/config.
- `qwen_image_vae_config_from_mapping` (function): Validate and convert a VAE config mapping.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_header

if TYPE_CHECKING:  # pragma: no cover
    import torch

from .config import QWEN_IMAGE_LATENT_CHANNELS

_VENDORED_VAE_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "huggingface"
    / "Qwen"
    / "Qwen-Image-Edit-2511"
    / "vae"
    / "config.json"
)
_EXPECTED_VAE_TENSOR_COUNT = 194
_EXPECTED_VAE_DTYPE = "BF16"
_CRITICAL_VAE_SHAPES = {
    "encoder.conv_in.weight": (96, 3, 3, 3, 3),
    "decoder.conv_out.weight": (3, 96, 3, 3, 3),
    "quant_conv.weight": (32, 32, 1, 1, 1),
    "post_quant_conv.weight": (16, 16, 1, 1, 1),
}


@dataclass(frozen=True, slots=True)
class QwenImageVaeConfig:
    class_name: str
    z_dim: int
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.class_name != "AutoencoderKLQwenImage":
            raise ValueError("Qwen Image VAE class must be AutoencoderKLQwenImage")
        if self.z_dim != QWEN_IMAGE_LATENT_CHANNELS:
            raise ValueError(f"Qwen Image VAE z_dim must be {QWEN_IMAGE_LATENT_CHANNELS}")
        if len(self.latents_mean) != QWEN_IMAGE_LATENT_CHANNELS:
            raise ValueError(f"Qwen Image VAE latents_mean must have {QWEN_IMAGE_LATENT_CHANNELS} values")
        if len(self.latents_std) != QWEN_IMAGE_LATENT_CHANNELS:
            raise ValueError(f"Qwen Image VAE latents_std must have {QWEN_IMAGE_LATENT_CHANNELS} values")
        for index, value in enumerate(self.latents_mean):
            if not math.isfinite(float(value)):
                raise ValueError(f"Qwen Image VAE latents_mean[{index}] must be finite")
        for index, value in enumerate(self.latents_std):
            if not math.isfinite(float(value)) or float(value) == 0.0:
                raise ValueError(f"Qwen Image VAE latents_std[{index}] must be finite and non-zero")


def _float_tuple(values: object, *, field: str, context: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise RuntimeError(f"{context}: {field} must be a sequence of {QWEN_IMAGE_LATENT_CHANNELS} numbers.")
    result: list[float] = []
    for index, value in enumerate(values):
        try:
            result.append(float(value))
        except Exception as exc:  # noqa: BLE001 - strict metadata validation
            raise RuntimeError(f"{context}: {field}[{index}] must be numeric; got {value!r}.") from exc
    return tuple(result)


def qwen_image_vae_config_from_mapping(
    config: Mapping[str, object],
    *,
    context: str = "Qwen Image VAE metadata",
) -> QwenImageVaeConfig:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"{context}: VAE config must be a mapping.")
    try:
        vae_config = QwenImageVaeConfig(
            class_name=str(config.get("_class_name") or "").strip(),
            z_dim=int(config.get("z_dim") or 0),
            latents_mean=_float_tuple(config.get("latents_mean"), field="latents_mean", context=context),
            latents_std=_float_tuple(config.get("latents_std"), field="latents_std", context=context),
        )
    except ValueError as exc:
        raise RuntimeError(f"{context}: {exc}") from exc
    return vae_config


def _read_vae_config(path: Path, *, context: str) -> Mapping[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{context}: Qwen Image VAE config not found: {path}") from exc
    except Exception as exc:  # noqa: BLE001 - strict metadata validation
        raise RuntimeError(f"{context}: invalid Qwen Image VAE config JSON at {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise RuntimeError(f"{context}: Qwen Image VAE config must be a JSON object: {path}")
    return data


def _validate_vae_safetensors_header(path: Path, *, context: str) -> None:
    try:
        header = read_safetensors_header(path)
    except Exception as exc:
        raise RuntimeError(f"{context}: failed to read SafeTensors header from {path}: {exc}") from exc

    tensor_entries = {
        str(name): metadata
        for name, metadata in header.items()
        if name != "__metadata__" and isinstance(metadata, Mapping)
    }
    if len(tensor_entries) != _EXPECTED_VAE_TENSOR_COUNT:
        raise RuntimeError(
            f"{context}: exact Edit-2511 VAE tensor count mismatch for {path}. "
            f"got={len(tensor_entries)} expected={_EXPECTED_VAE_TENSOR_COUNT}."
        )

    wrong_dtypes: list[str] = []
    for name, metadata in tensor_entries.items():
        dtype = metadata.get("dtype")
        if dtype != _EXPECTED_VAE_DTYPE:
            wrong_dtypes.append(f"{name}:{dtype!r}")
    if wrong_dtypes:
        raise RuntimeError(
            f"{context}: Edit-2511 VAE tensors must all use {_EXPECTED_VAE_DTYPE}. "
            f"offenders_sample={wrong_dtypes[:10]}"
        )

    for name, expected_shape in _CRITICAL_VAE_SHAPES.items():
        metadata = tensor_entries.get(name)
        raw_shape = metadata.get("shape") if isinstance(metadata, Mapping) else None
        try:
            actual_shape = tuple(int(dim) for dim in raw_shape) if isinstance(raw_shape, (list, tuple)) else None
        except Exception:
            actual_shape = None
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{context}: Edit-2511 VAE critical shape mismatch for {name!r}. "
                f"got={actual_shape!r} expected={expected_shape!r}."
            )


def _is_under_allowed_root(path: Path, roots: Sequence[object]) -> bool:
    resolved_path = path.resolve()
    for raw_root in roots:
        if not isinstance(raw_root, str) or not raw_root.strip():
            continue
        root = Path(raw_root.strip()).expanduser().resolve()
        if root.is_file():
            if resolved_path == root:
                return True
            continue
        try:
            resolved_path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def qwen_image_validate_external_vae_path(
    raw_path: object,
    *,
    allowed_roots: Sequence[object] = (),
    context: str = "Qwen Image external VAE",
) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError(f"{context}: path must be a non-empty string.")
    path = Path(raw_path.strip()).expanduser()
    if not path.exists():
        raise RuntimeError(f"{context}: path not found: {path}")
    if not path.is_file():
        raise RuntimeError(f"{context}: Edit-2511 requires one exact VAE SafeTensors file; got {path}.")
    if path.suffix.lower() not in {".safetensor", ".safetensors"}:
        raise RuntimeError(f"{context}: Edit-2511 VAE must be a SafeTensors file; got {path}.")
    if allowed_roots and not _is_under_allowed_root(path, allowed_roots):
        roots_text = ", ".join(str(root) for root in allowed_roots if isinstance(root, str) and root.strip())
        raise RuntimeError(
            f"{context}: path must be under qwen_image_vae roots; got {path}. Roots: {roots_text or '<none>'}."
        )

    _validate_vae_safetensors_header(path, context=context)
    qwen_image_vae_config_from_mapping(
        _read_vae_config(_VENDORED_VAE_CONFIG_PATH, context=context),
        context=str(_VENDORED_VAE_CONFIG_PATH),
    )
    return str(path)


def _latent_stat_tensor(latents: torch.Tensor, values: tuple[float, ...], *, field: str) -> torch.Tensor:
    import torch

    if not isinstance(latents, torch.Tensor):
        raise TypeError("latents must be a torch.Tensor")
    if latents.ndim < 3:
        raise RuntimeError(f"Qwen Image VAE latents must have at least 3 dimensions; got shape={tuple(latents.shape)}.")
    channels = int(latents.shape[1])
    if channels != QWEN_IMAGE_LATENT_CHANNELS:
        raise RuntimeError(
            f"Qwen Image VAE latent channel mismatch: expected {QWEN_IMAGE_LATENT_CHANNELS}, got {channels}."
        )
    stat = torch.tensor(values, device=latents.device, dtype=latents.dtype)
    shape = (1, QWEN_IMAGE_LATENT_CHANNELS, *([1] * (latents.ndim - 2)))
    if stat.numel() != QWEN_IMAGE_LATENT_CHANNELS:
        raise RuntimeError(f"Qwen Image VAE {field} must have {QWEN_IMAGE_LATENT_CHANNELS} values.")
    return stat.reshape(shape)


def qwen_image_normalize_latents(latents: torch.Tensor, config: QwenImageVaeConfig) -> torch.Tensor:
    mean = _latent_stat_tensor(latents, config.latents_mean, field="latents_mean")
    std = _latent_stat_tensor(latents, config.latents_std, field="latents_std")
    return (latents - mean) / std


def qwen_image_denormalize_latents(latents: torch.Tensor, config: QwenImageVaeConfig) -> torch.Tensor:
    mean = _latent_stat_tensor(latents, config.latents_mean, field="latents_mean")
    std = _latent_stat_tensor(latents, config.latents_std, field="latents_std")
    return latents * std + mean


__all__ = [
    "QwenImageVaeConfig",
    "qwen_image_denormalize_latents",
    "qwen_image_normalize_latents",
    "qwen_image_validate_external_vae_path",
    "qwen_image_vae_config_from_mapping",
]

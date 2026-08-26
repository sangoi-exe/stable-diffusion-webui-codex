"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared native 3D VAE runtime lane for temporal video autoencoders.
Defines `AutoencoderCodex3D` with causal 3D convolutions, temporal cache-aware
encode/decode chunk execution, knob-selected exact query-tiled spatial attention,
explicit temporal down/upsampling, and strict diffusers->codex keyspace resolution
helpers for WAN-like checkpoints without importing diffusers model classes. WAN22
keyspace ownership lives in `runtime/state_dict/keymap_wan22_vae.py`.

Symbols (top-level; keep in sync; no ghosts):
- `_CACHE_T` (constant): Temporal cache window used by causal-conv cache plumbing in chunked encode/decode passes.
- `_CODEX3D_TILED_ATTENTION_QUERY_TOKENS` (constant): Maximum spatial attention query tokens processed per exact tiled chunk when VAE tiling is enabled.
- `Codex3DAttentionBlock` (class): Frame-wise native spatial attention block with optional exact query tiling.
- `AutoencoderCodex3D` (class): Native causal-3D KL VAE with codex keyspace (`encoder.downsamples.*`, `decoder.upsamples.*`, `conv1/conv2`).
- `sanitize_codex3d_vae_config` (function): Normalizes alias config fields into `AutoencoderCodex3D` constructor arguments.
- `resolve_codex3d_vae_keyspace` (function): Normalizes wrapper prefixes and resolves diffusers WAN3D VAE keys into codex lookup space (strict/fail-loud).
- `is_codex3d_vae_instance` (function): Returns True when a model instance belongs to the native codex 3D VAE lane.
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch import nn

from apps.backend.infra.config.env_flags import env_flag
from apps.backend.runtime.attention import attention_function_single_head_spatial
from apps.backend.runtime.common.vae_ldm import DiagonalGaussianDistribution
from apps.backend.runtime.state_dict.keymap_wan22_vae import resolve_wan22_vae_3d_keyspace

_CACHE_T = 2
_CODEX3D_TILED_ATTENTION_QUERY_TOKENS = 1024


def _vae_trace_verbose_enabled() -> bool:
    return env_flag("CODEX_TRACE_INFERENCE_DEBUG", default=False)


def _cuda_mem_snapshot_str(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cuda_mem=n/a"
    try:
        alloc_mb = float(torch.cuda.memory_allocated(device)) / (1024**2)
        reserved_mb = float(torch.cuda.memory_reserved(device)) / (1024**2)
        max_alloc_mb = float(torch.cuda.max_memory_allocated(device)) / (1024**2)
        max_reserved_mb = float(torch.cuda.max_memory_reserved(device)) / (1024**2)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_mb = float(free_bytes) / (1024**2)
        total_mb = float(total_bytes) / (1024**2)
        return (
            f"alloc={alloc_mb:.0f}MB reserved={reserved_mb:.0f}MB free={free_mb:.0f}MB total={total_mb:.0f}MB "
            f"max_alloc={max_alloc_mb:.0f}MB max_reserved={max_reserved_mb:.0f}MB"
        )
    except Exception:
        return "cuda_mem=unavailable"


def resolve_codex3d_vae_keyspace(
    state_dict: MutableMapping[str, Any],
) -> tuple[str, MutableMapping[str, Any]]:
    resolved = resolve_wan22_vae_3d_keyspace(state_dict)
    style = resolved.style
    style_label = style.value if hasattr(style, "value") else str(style)
    return style_label, resolved.view


class Codex3DCausalConv(nn.Conv3d):
    """Causal Conv3d with left-only temporal padding."""

    _dispatch_workaround_enabled_cache: bool | None = None

    @classmethod
    def _dispatch_workaround_enabled(cls) -> bool:
        cached = cls._dispatch_workaround_enabled_cache
        if cached is not None:
            return bool(cached)

        enabled = False
        try:
            version_str = str(torch.__version__).split("+", 1)[0]
            version_parts = version_str.split(".")
            major = int(version_parts[0]) if len(version_parts) > 0 else 0
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            if major > 2 or (major == 2 and minor >= 9):
                cudnn_backend = getattr(torch.backends, "cudnn", None)
                cudnn_version = (
                    cudnn_backend.version()
                    if cudnn_backend is not None and hasattr(cudnn_backend, "version")
                    else None
                )
                enabled = (
                    hasattr(torch, "cudnn_convolution")
                    and cudnn_version is not None
                    and int(cudnn_version) >= 91002
                )
        except Exception:
            enabled = False

        cls._dispatch_workaround_enabled_cache = bool(enabled)
        return bool(enabled)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self._padding = (
            int(self.padding[2]),
            int(self.padding[2]),
            int(self.padding[1]),
            int(self.padding[1]),
            int(2 * self.padding[0]),
            0,
        )
        self.padding = (0, 0, 0)

    def _conv_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            self._dispatch_workaround_enabled()
            and input.device.type == "cuda"
            and weight.device.type == "cuda"
            and weight.dtype in (torch.float16, torch.bfloat16)
        ):
            cudnn_backend = getattr(torch.backends, "cudnn", None)
            cudnn_available = (
                cudnn_backend is not None
                and hasattr(cudnn_backend, "is_available")
                and bool(cudnn_backend.is_available())
                and bool(getattr(cudnn_backend, "enabled", True))
            )
            if cudnn_available:
                try:
                    out = torch.cudnn_convolution(
                        input,
                        weight,
                        self.padding,
                        self.stride,
                        self.dilation,
                        self.groups,
                        benchmark=bool(getattr(cudnn_backend, "benchmark", False)),
                        deterministic=bool(getattr(cudnn_backend, "deterministic", False)),
                        allow_tf32=bool(getattr(cudnn_backend, "allow_tf32", True)),
                    )
                    if bias is not None:
                        out = out + bias.reshape((1, -1) + (1,) * (out.ndim - 2))
                    return out
                except RuntimeError:
                    pass
        return super()._conv_forward(input, weight, bias)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 5:
            raise RuntimeError(
                "AutoencoderCodex3D causal conv expects 5D input [B,C,T,H,W], "
                f"got shape={tuple(x.shape)}."
            )
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= int(cache_x.shape[2])
        x = F.pad(x, padding)
        return super().forward(x)


class Codex3DRMSNorm(nn.Module):
    def __init__(self, dim: int, *, images: bool = True, channel_first: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcast_dims = (1, 1) if images else (1, 1, 1)
        shape = (int(dim), *broadcast_dims) if channel_first else (int(dim),)
        self._channel_first = bool(channel_first)
        self._scale = float(dim) ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self._channel_first else -1
        out = F.normalize(x, dim=dim) * self._scale
        gamma = self.gamma.to(dtype=x.dtype, device=x.device)
        if gamma.ndim < x.ndim:
            gamma = gamma.view((1,) * (x.ndim - gamma.ndim) + tuple(gamma.shape))
        out = out * gamma
        if self.bias is not None:
            bias = self.bias.to(dtype=x.dtype, device=x.device)
            if bias.ndim < x.ndim:
                bias = bias.view((1,) * (x.ndim - bias.ndim) + tuple(bias.shape))
            out = out + bias
        return out


class Codex3DUpsample(nn.Upsample):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return super().forward(x)
        except Exception:
            return super().forward(x.float()).to(dtype=x.dtype)


class Codex3DResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.residual = nn.Sequential(
            Codex3DRMSNorm(self.in_dim, images=False),
            nn.SiLU(),
            Codex3DCausalConv(self.in_dim, self.out_dim, 3, padding=1),
            Codex3DRMSNorm(self.out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            Codex3DCausalConv(self.out_dim, self.out_dim, 3, padding=1),
        )
        self.shortcut = Codex3DCausalConv(self.in_dim, self.out_dim, 1) if self.in_dim != self.out_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        if feat_cache is not None and feat_idx is None:
            raise RuntimeError("AutoencoderCodex3D residual block cache requires feat_idx.")

        identity = self.shortcut(x)

        x = self.residual[0](x)
        x = self.residual[1](x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            cached = feat_cache[idx]
            if cache_x.shape[2] < _CACHE_T and torch.is_tensor(cached):
                cache_x = torch.cat([cached[:, :, -1:, :, :], cache_x], dim=2)
            x = self.residual[2](x, cached if torch.is_tensor(cached) else None)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.residual[2](x)

        x = self.residual[3](x)
        x = self.residual[4](x)
        x = self.residual[5](x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            cached = feat_cache[idx]
            if cache_x.shape[2] < _CACHE_T and torch.is_tensor(cached):
                cache_x = torch.cat([cached[:, :, -1:, :, :], cache_x], dim=2)
            x = self.residual[6](x, cached if torch.is_tensor(cached) else None)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.residual[6](x)

        return x + identity


class Codex3DAttentionBlock(nn.Module):
    """Single-head spatial attention applied frame-wise."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = Codex3DRMSNorm(self.dim, images=True)
        self.to_qkv = nn.Conv2d(self.dim, self.dim * 3, 1)
        self.proj = nn.Conv2d(self.dim, self.dim, 1)
        self._query_chunk_size: int | None = None

    def enable_query_tiling(self, *, query_chunk_size: int) -> None:
        if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size <= 0:
            raise RuntimeError(
                "AutoencoderCodex3D attention query chunk size must be a positive integer; "
                f"got {query_chunk_size!r}."
            )
        self._query_chunk_size = int(query_chunk_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise RuntimeError(
                "AutoencoderCodex3D attention expects 5D input [B,C,T,H,W], "
                f"got shape={tuple(x.shape)}."
            )
        batch, channels, frames, height, width = x.shape
        residual = x
        normed = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        normed = self.norm(normed)
        qkv = self.to_qkv(normed)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        if self._query_chunk_size is None:
            q = q.reshape(batch * frames, channels, height * width).transpose(1, 2)
            k = k.reshape(batch * frames, channels, height * width)
            v = v.reshape(batch * frames, channels, height * width).transpose(1, 2)
            attn = torch.matmul(q, k) * (channels ** -0.5)
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(batch * frames, channels, height, width)
        else:
            out = attention_function_single_head_spatial(
                q,
                k,
                v,
                query_chunk_size=self._query_chunk_size,
            )
        out = self.proj(out).view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return residual + out


class Codex3DResample(nn.Module):
    def __init__(self, dim: int, *, mode: str) -> None:
        super().__init__()
        self.dim = int(dim)
        self.mode = str(mode)

        if self.mode == "upsample2d":
            self.resample = nn.Sequential(Codex3DUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif self.mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif self.mode == "upsample3d":
            self.resample = nn.Sequential(Codex3DUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = Codex3DCausalConv(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif self.mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = Codex3DCausalConv(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        elif self.mode == "none":
            self.resample = nn.Identity()
        else:
            raise RuntimeError(f"AutoencoderCodex3D unsupported resample mode={self.mode!r}.")

    def forward(
        self,
        x: torch.Tensor,
        *,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise RuntimeError(
                "AutoencoderCodex3D resample expects 5D input [B,C,T,H,W], "
                f"got shape={tuple(x.shape)}."
            )
        if feat_cache is not None and feat_idx is None:
            raise RuntimeError("AutoencoderCodex3D resample cache requires feat_idx.")

        batch, channels, frames, height, width = x.shape
        work = x
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                cached = feat_cache[idx]
                if cached is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = work[:, :, -_CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < _CACHE_T and cached != "Rep" and torch.is_tensor(cached):
                        cache_x = torch.cat([cached[:, :, -1:, :, :], cache_x], dim=2)
                    if cache_x.shape[2] < _CACHE_T and cached == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                    if cached == "Rep":
                        work = self.time_conv(work)
                    else:
                        work = self.time_conv(work, cached if torch.is_tensor(cached) else None)
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    mixed = work.reshape(batch, 2, channels, frames, height, width)
                    work = torch.stack((mixed[:, 0], mixed[:, 1]), dim=3).reshape(batch, channels, frames * 2, height, width)

        flat = work.permute(0, 2, 1, 3, 4).reshape(batch * int(work.shape[2]), channels, int(work.shape[3]), int(work.shape[4]))
        flat = self.resample(flat)
        out_channels = int(flat.shape[1])
        out_height = int(flat.shape[2])
        out_width = int(flat.shape[3])
        work = flat.reshape(batch, int(work.shape[2]), out_channels, out_height, out_width).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                cached = feat_cache[idx]
                if cached is None:
                    feat_cache[idx] = work[:, :, -1:, :, :].clone()
                    feat_idx[0] += 1
                else:
                    cache_x = work[:, :, -1:, :, :].clone()
                    if not torch.is_tensor(cached):
                        raise RuntimeError("AutoencoderCodex3D downsample cache is corrupted.")
                    work = self.time_conv(torch.cat([cached[:, :, -1:, :, :], work], dim=2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
            else:
                work = self.time_conv(work)
        return work


class Codex3DEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        z_dim: int,
        input_channels: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        attn_scales: Sequence[float],
        temperal_downsample: Sequence[bool],
        dropout: float,
    ) -> None:
        super().__init__()
        dims = [int(dim) * m for m in (1, *tuple(int(x) for x in dim_mult))]
        scale = 1.0
        self.conv1 = Codex3DCausalConv(int(input_channels), int(dims[0]), 3, padding=1)
        downsamples: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            in_dim = int(in_dim)
            out_dim = int(out_dim)
            for _ in range(int(num_res_blocks)):
                downsamples.append(Codex3DResidualBlock(in_dim, out_dim, dropout=float(dropout)))
                if float(scale) in tuple(float(v) for v in attn_scales):
                    downsamples.append(Codex3DAttentionBlock(out_dim))
                in_dim = out_dim
            if index != len(tuple(dim_mult)) - 1:
                mode = "downsample3d" if bool(tuple(temperal_downsample)[index]) else "downsample2d"
                downsamples.append(Codex3DResample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            Codex3DResidualBlock(out_dim, out_dim, dropout=float(dropout)),
            Codex3DAttentionBlock(out_dim),
            Codex3DResidualBlock(out_dim, out_dim, dropout=float(dropout)),
        )
        self.head = nn.Sequential(
            Codex3DRMSNorm(out_dim, images=False),
            nn.SiLU(),
            Codex3DCausalConv(out_dim, int(z_dim), 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        if feat_cache is not None and feat_idx is None:
            raise RuntimeError("AutoencoderCodex3D encoder cache requires feat_idx.")

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            cached = feat_cache[idx]
            if cache_x.shape[2] < _CACHE_T and torch.is_tensor(cached):
                cache_x = torch.cat([cached[:, :, -1:, :, :], cache_x], dim=2)
            x = self.conv1(x, cached if torch.is_tensor(cached) else None)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if isinstance(layer, (Codex3DResidualBlock, Codex3DResample)):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        for layer in self.middle:
            if isinstance(layer, Codex3DResidualBlock):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        x = self.head[0](x)
        x = self.head[1](x)
        conv_out = self.head[2]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            cached = feat_cache[idx]
            if cache_x.shape[2] < _CACHE_T and torch.is_tensor(cached):
                cache_x = torch.cat([cached[:, :, -1:, :, :], cache_x], dim=2)
            x = conv_out(x, cached if torch.is_tensor(cached) else None)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
            return x
        return conv_out(x)


class Codex3DDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        z_dim: int,
        output_channels: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        attn_scales: Sequence[float],
        temperal_upsample: Sequence[bool],
        dropout: float,
    ) -> None:
        super().__init__()
        dim_mult_tuple = tuple(int(x) for x in dim_mult)
        dims = [int(dim) * int(v) for v in (dim_mult_tuple[-1], *reversed(dim_mult_tuple))]
        scale = 1.0 / (2 ** max(len(tuple(dim_mult)) - 2, 0))
        self.conv1 = Codex3DCausalConv(int(z_dim), int(dims[0]), 3, padding=1)
        self.middle = nn.Sequential(
            Codex3DResidualBlock(int(dims[0]), int(dims[0]), dropout=float(dropout)),
            Codex3DAttentionBlock(int(dims[0])),
            Codex3DResidualBlock(int(dims[0]), int(dims[0]), dropout=float(dropout)),
        )
        upsamples: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            in_dim_i = int(in_dim)
            if index in (1, 2, 3):
                in_dim_i = in_dim_i // 2
            for _ in range(int(num_res_blocks) + 1):
                upsamples.append(Codex3DResidualBlock(in_dim_i, int(out_dim), dropout=float(dropout)))
                if float(scale) in tuple(float(v) for v in attn_scales):
                    upsamples.append(Codex3DAttentionBlock(int(out_dim)))
                in_dim_i = int(out_dim)
            if index != len(tuple(dim_mult)) - 1:
                mode = "upsample3d" if bool(tuple(temperal_upsample)[index]) else "upsample2d"
                upsamples.append(Codex3DResample(int(out_dim), mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            Codex3DRMSNorm(int(out_dim), images=False),
            nn.SiLU(),
            Codex3DCausalConv(int(out_dim), int(output_channels), 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        _ = first_chunk
        if feat_cache is not None and feat_idx is None:
            raise RuntimeError("AutoencoderCodex3D decoder cache requires feat_idx.")
        trace_logger = get_backend_logger("backend.runtime.wan22.vae_codex3d")
        trace_enabled = _vae_trace_verbose_enabled() and trace_logger.isEnabledFor(logging.DEBUG)
        middle_count = int(len(self.middle))
        upsample_count = int(len(self.upsamples))
        block_total = middle_count + upsample_count

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            cached = feat_cache[idx]
            if cache_x.shape[2] < _CACHE_T and torch.is_tensor(cached):
                cache_x = torch.cat([cached[:, :, -1:, :, :], cache_x], dim=2)
            x = self.conv1(x, cached if torch.is_tensor(cached) else None)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        if trace_enabled:
            trace_logger.debug(
                "[wan22.vae.trace] decoder.conv1: x_shape=%s x_dtype=%s x_device=%s %s",
                tuple(x.shape),
                str(x.dtype),
                str(x.device),
                _cuda_mem_snapshot_str(x.device),
            )

        for layer_index, layer in enumerate(self.middle):
            block_number = int(layer_index) + 1
            if trace_enabled:
                trace_logger.debug(
                    "[wan22.vae.trace] decoder.block[%d/%d] pre: section=middle layer=%s x_shape=%s x_dtype=%s x_device=%s %s",
                    block_number,
                    block_total,
                    layer.__class__.__name__,
                    tuple(x.shape),
                    str(x.dtype),
                    str(x.device),
                    _cuda_mem_snapshot_str(x.device),
                )
            if isinstance(layer, Codex3DResidualBlock):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)
            if trace_enabled:
                trace_logger.debug(
                    "[wan22.vae.trace] decoder.block[%d/%d] post: section=middle layer=%s x_shape=%s x_dtype=%s x_device=%s %s",
                    block_number,
                    block_total,
                    layer.__class__.__name__,
                    tuple(x.shape),
                    str(x.dtype),
                    str(x.device),
                    _cuda_mem_snapshot_str(x.device),
                )

        for layer_index, layer in enumerate(self.upsamples):
            block_number = middle_count + int(layer_index) + 1
            if trace_enabled:
                trace_logger.debug(
                    "[wan22.vae.trace] decoder.block[%d/%d] pre: section=upsample layer=%s x_shape=%s x_dtype=%s x_device=%s %s",
                    block_number,
                    block_total,
                    layer.__class__.__name__,
                    tuple(x.shape),
                    str(x.dtype),
                    str(x.device),
                    _cuda_mem_snapshot_str(x.device),
                )
            if isinstance(layer, (Codex3DResidualBlock, Codex3DResample)):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)
            if trace_enabled:
                trace_logger.debug(
                    "[wan22.vae.trace] decoder.block[%d/%d] post: section=upsample layer=%s x_shape=%s x_dtype=%s x_device=%s %s",
                    block_number,
                    block_total,
                    layer.__class__.__name__,
                    tuple(x.shape),
                    str(x.dtype),
                    str(x.device),
                    _cuda_mem_snapshot_str(x.device),
                )

        x = self.head[0](x)
        x = self.head[1](x)
        if trace_enabled:
            trace_logger.debug(
                "[wan22.vae.trace] decoder.head: x_shape=%s x_dtype=%s x_device=%s %s",
                tuple(x.shape),
                str(x.dtype),
                str(x.device),
                _cuda_mem_snapshot_str(x.device),
            )
        conv_out = self.head[2]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            cached = feat_cache[idx]
            if cache_x.shape[2] < _CACHE_T and torch.is_tensor(cached):
                cache_x = torch.cat([cached[:, :, -1:, :, :], cache_x], dim=2)
            x = conv_out(x, cached if torch.is_tensor(cached) else None)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
            if trace_enabled:
                trace_logger.debug(
                    "[wan22.vae.trace] decoder.out: x_shape=%s x_dtype=%s x_device=%s %s",
                    tuple(x.shape),
                    str(x.dtype),
                    str(x.device),
                    _cuda_mem_snapshot_str(x.device),
                )
            return x
        out = conv_out(x)
        if trace_enabled:
            trace_logger.debug(
                "[wan22.vae.trace] decoder.out: x_shape=%s x_dtype=%s x_device=%s %s",
                tuple(out.shape),
                str(out.dtype),
                str(out.device),
                _cuda_mem_snapshot_str(out.device),
            )
        return out


@dataclass(frozen=True, slots=True)
class _Codex3DConfigView:
    z_dim: int
    scaling_factor: float
    shift_factor: float | None


class AutoencoderCodex3D(nn.Module, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        *,
        base_dim: int = 96,
        decoder_base_dim: int | None = None,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence[float] = (),
        temperal_downsample: Sequence[bool] = (False, True, True),
        dropout: float = 0.0,
        in_channels: int = 3,
        out_channels: int = 3,
        scaling_factor: float = 1.0,
        shift_factor: float | None = None,
        latents_mean: Sequence[float] | None = None,
        latents_std: Sequence[float] | None = None,
        scale_factor_temporal: int | None = 4,
        scale_factor_spatial: int | None = 8,
    ) -> None:
        super().__init__()
        if decoder_base_dim is not None and int(decoder_base_dim) != int(base_dim):
            raise RuntimeError(
                "AutoencoderCodex3D requires decoder_base_dim to match base_dim "
                f"(got base_dim={int(base_dim)} decoder_base_dim={int(decoder_base_dim)})."
            )
        self.z_dim = int(z_dim)
        self.temperal_downsample = tuple(bool(x) for x in temperal_downsample)
        self.temperal_upsample = tuple(reversed(self.temperal_downsample))
        self.encoder = Codex3DEncoder(
            dim=int(base_dim),
            z_dim=int(self.z_dim) * 2,
            input_channels=int(in_channels),
            dim_mult=tuple(int(v) for v in dim_mult),
            num_res_blocks=int(num_res_blocks),
            attn_scales=tuple(float(v) for v in attn_scales),
            temperal_downsample=self.temperal_downsample,
            dropout=float(dropout),
        )
        self.conv1 = Codex3DCausalConv(int(self.z_dim) * 2, int(self.z_dim) * 2, 1)
        self.conv2 = Codex3DCausalConv(int(self.z_dim), int(self.z_dim), 1)
        self.decoder = Codex3DDecoder(
            dim=int(base_dim),
            z_dim=int(self.z_dim),
            output_channels=int(out_channels),
            dim_mult=tuple(int(v) for v in dim_mult),
            num_res_blocks=int(num_res_blocks),
            attn_scales=tuple(float(v) for v in attn_scales),
            temperal_upsample=self.temperal_upsample,
            dropout=float(dropout),
        )
        self.scaling_factor = float(scaling_factor)
        self.shift_factor = None if shift_factor is None else float(shift_factor)
        self._shift_factor_value = 0.0 if self.shift_factor is None else float(self.shift_factor)
        self.scale_factor_temporal = int(scale_factor_temporal) if scale_factor_temporal is not None else 4
        self.scale_factor_spatial = int(scale_factor_spatial) if scale_factor_spatial is not None else 8
        self.latents_mean = None if latents_mean is None else tuple(float(v) for v in latents_mean)
        self.latents_std = None if latents_std is None else tuple(float(v) for v in latents_std)
        self.use_tiling = False
        self._cached_conv_counts = {
            "decoder": sum(isinstance(module, Codex3DCausalConv) for module in self.decoder.modules()) if self.decoder is not None else 0,
            "encoder": sum(isinstance(module, Codex3DCausalConv) for module in self.encoder.modules()) if self.encoder is not None else 0,
        }
        self._conv_num = 0
        self._conv_idx: list[int] = [0]
        self._feat_map: list[torch.Tensor | str | None] = []
        self._enc_conv_num = 0
        self._enc_conv_idx: list[int] = [0]
        self._enc_feat_map: list[torch.Tensor | str | None] = []

    def enable_tiling(self) -> None:
        self.use_tiling = True
        attention_blocks = tuple(module for module in self.modules() if isinstance(module, Codex3DAttentionBlock))
        if not attention_blocks:
            raise RuntimeError("AutoencoderCodex3D tiling requires at least one spatial attention block.")
        for attention_block in attention_blocks:
            attention_block.enable_query_tiling(query_chunk_size=_CODEX3D_TILED_ATTENTION_QUERY_TOKENS)

    def clear_cache(self) -> None:
        self._conv_num = int(self._cached_conv_counts["decoder"])
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = int(self._cached_conv_counts["encoder"])
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def encode(self, x: torch.Tensor, return_dict: bool = True, regulation: Any = None) -> Any:
        squeeze_t = False
        if x.ndim == 4:
            squeeze_t = True
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise RuntimeError(
                "AutoencoderCodex3D encode expects 4D or 5D input ([B,C,H,W] or [B,C,T,H,W]), "
                f"got shape={tuple(x.shape)}."
            )
        _, _, num_frame, _, _ = x.shape
        self.clear_cache()
        chunk_count = 1 + (int(num_frame) - 1) // 4
        encoded: torch.Tensor | None = None
        for index in range(int(chunk_count)):
            self._enc_conv_idx = [0]
            if index == 0:
                encoded_chunk = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                encoded_chunk = self.encoder(
                    x[:, :, 1 + 4 * (index - 1) : 1 + 4 * index, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            if encoded_chunk.ndim != 5:
                raise RuntimeError(
                    "AutoencoderCodex3D encoder produced non-5D chunk "
                    f"(chunk={index} shape={tuple(encoded_chunk.shape)})."
                )
            if int(encoded_chunk.shape[2]) != 1:
                raise RuntimeError(
                    "AutoencoderCodex3D encode expected chunk temporal size of 1 "
                    f"(chunk={index} got={int(encoded_chunk.shape[2])})."
                )
            if encoded is None:
                encoded = encoded_chunk.new_empty(
                    (
                        int(encoded_chunk.shape[0]),
                        int(encoded_chunk.shape[1]),
                        int(chunk_count),
                        int(encoded_chunk.shape[3]),
                        int(encoded_chunk.shape[4]),
                    )
                )
            elif (
                int(encoded_chunk.shape[0]) != int(encoded.shape[0])
                or int(encoded_chunk.shape[1]) != int(encoded.shape[1])
                or int(encoded_chunk.shape[3]) != int(encoded.shape[3])
                or int(encoded_chunk.shape[4]) != int(encoded.shape[4])
            ):
                raise RuntimeError(
                    "AutoencoderCodex3D encode chunk shape mismatch; refusing to assemble heterogeneous chunks "
                    f"(chunk={index} shape={tuple(encoded_chunk.shape)} expected=(B={int(encoded.shape[0])},"
                    f" C={int(encoded.shape[1])}, T=1, H={int(encoded.shape[3])}, W={int(encoded.shape[4])}))."
                )
            encoded[:, :, index : index + 1, :, :] = encoded_chunk
        if encoded is None:
            raise RuntimeError("AutoencoderCodex3D encode produced no chunks.")
        moments = self.conv1(encoded)
        self.clear_cache()
        posterior = DiagonalGaussianDistribution(moments)
        latents = regulation(posterior) if regulation is not None else posterior.sample()
        if squeeze_t and latents.ndim == 5 and int(latents.shape[2]) == 1:
            latents = latents.squeeze(2)
        if not return_dict:
            return (posterior,)
        return SimpleNamespace(latent_dist=posterior, latents=latents)

    def decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
        *,
        chunk_callback: Callable[[torch.Tensor, int], None] | None = None,
    ) -> Any:
        squeeze_t = False
        if z.ndim == 4:
            squeeze_t = True
            z = z.unsqueeze(2)
        if z.ndim != 5:
            raise RuntimeError(
                "AutoencoderCodex3D decode expects 4D or 5D latents ([B,C,H,W] or [B,C,T,H,W]), "
                f"got shape={tuple(z.shape)}."
            )
        trace_logger = get_backend_logger("backend.runtime.wan22.vae_codex3d")
        trace_enabled = _vae_trace_verbose_enabled() and trace_logger.isEnabledFor(logging.DEBUG)
        _, _, num_frame, _, _ = z.shape
        conv2_cache: torch.Tensor | None = None
        conv2_padding = getattr(self.conv2, "_padding", (0, 0, 0, 0, 0, 0))
        conv2_cache_t = int(conv2_padding[4]) if len(conv2_padding) >= 5 else 0
        if conv2_cache_t < 0:
            raise RuntimeError(
                "AutoencoderCodex3D decode encountered invalid conv2 temporal cache size "
                f"(cache_t={conv2_cache_t})."
            )
        if trace_enabled:
            trace_logger.debug(
                "[wan22.vae.trace] decode.start: z_shape=%s z_dtype=%s z_device=%s frames=%d conv2_cache_t=%d %s",
                tuple(z.shape),
                str(z.dtype),
                str(z.device),
                int(num_frame),
                conv2_cache_t,
                _cuda_mem_snapshot_str(z.device),
            )
        self.clear_cache()
        out_chunks: list[torch.Tensor] = []
        chunk_count = 0
        try:
            for index in range(int(num_frame)):
                self._conv_idx = [0]
                z_chunk = z[:, :, index : index + 1, :, :]
                if trace_enabled:
                    trace_logger.debug(
                        "[wan22.vae.trace] decode.frame[%d/%d] pre: z_chunk_shape=%s z_chunk_dtype=%s z_chunk_device=%s conv2_cache_present=%s %s",
                        int(index) + 1,
                        int(num_frame),
                        tuple(z_chunk.shape),
                        str(z_chunk.dtype),
                        str(z_chunk.device),
                        str(conv2_cache is not None),
                        _cuda_mem_snapshot_str(z_chunk.device),
                    )
                if conv2_cache_t > 0:
                    x_chunk = self.conv2(z_chunk, conv2_cache)
                    conv2_cache_next = z_chunk[:, :, -conv2_cache_t:, :, :].clone()
                    if conv2_cache_next.shape[2] < conv2_cache_t and torch.is_tensor(conv2_cache):
                        missing = conv2_cache_t - int(conv2_cache_next.shape[2])
                        conv2_cache_next = torch.cat([conv2_cache[:, :, -missing:, :, :], conv2_cache_next], dim=2)
                    conv2_cache = conv2_cache_next
                else:
                    x_chunk = self.conv2(z_chunk)
                if index == 0:
                    out_chunk = self.decoder(
                        x_chunk,
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                        first_chunk=True,
                    )
                else:
                    out_chunk = self.decoder(
                        x_chunk,
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                    )
                chunk_count += 1
                if chunk_callback is not None:
                    chunk_callback(torch.clamp(out_chunk, min=-1.0, max=1.0), int(index))
                else:
                    out_chunks.append(out_chunk)
                if trace_enabled:
                    trace_logger.debug(
                        "[wan22.vae.trace] decode.frame[%d/%d] post: out_chunk_shape=%s out_chunk_dtype=%s out_chunk_device=%s %s",
                        int(index) + 1,
                        int(num_frame),
                        tuple(out_chunk.shape),
                        str(out_chunk.dtype),
                        str(out_chunk.device),
                        _cuda_mem_snapshot_str(out_chunk.device),
                    )
                del x_chunk
                del z_chunk
                del out_chunk
        finally:
            self.clear_cache()
            if trace_enabled:
                trace_logger.debug("[wan22.vae.trace] decode.finalize: cache cleared")
        if chunk_count < 1:
            raise RuntimeError("AutoencoderCodex3D decode produced no chunks.")
        if chunk_callback is not None:
            if not return_dict:
                return (None,)
            return SimpleNamespace(sample=None)
        out = out_chunks[0] if len(out_chunks) == 1 else torch.cat(out_chunks, dim=2)
        out = torch.clamp(out, min=-1.0, max=1.0)
        if squeeze_t and out.ndim == 5 and int(out.shape[2]) == 1:
            out = out.squeeze(2)
        if not return_dict:
            return (out,)
        return SimpleNamespace(sample=out)

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self._shift_factor_value) * self.scaling_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent / self.scaling_factor) + self._shift_factor_value


def sanitize_codex3d_vae_config(config: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(config)
    cleaned: dict[str, Any] = {}

    base_dim = source.get("base_dim")
    if base_dim is None:
        block_channels = source.get("block_out_channels")
        if isinstance(block_channels, (list, tuple)) and block_channels:
            base_dim = int(block_channels[0])
    if base_dim is None:
        base_dim = 96
    cleaned["base_dim"] = int(base_dim)

    decoder_base_dim = source.get("decoder_base_dim")
    if decoder_base_dim is not None:
        cleaned["decoder_base_dim"] = int(decoder_base_dim)

    z_dim = source.get("z_dim")
    if z_dim is None:
        z_dim = source.get("latent_channels")
    if z_dim is None:
        z_dim = 16
    cleaned["z_dim"] = int(z_dim)

    dim_mult = source.get("dim_mult")
    if dim_mult is None:
        block_channels = source.get("block_out_channels")
        if isinstance(block_channels, (list, tuple)) and block_channels:
            try:
                dim_mult = tuple(int(int(ch) // int(cleaned["base_dim"])) for ch in block_channels)
            except Exception:
                dim_mult = None
    cleaned["dim_mult"] = tuple(int(v) for v in (dim_mult if dim_mult is not None else (1, 2, 4, 4)))

    num_res_blocks = source.get("num_res_blocks")
    if num_res_blocks is None:
        num_res_blocks = source.get("layers_per_block")
    cleaned["num_res_blocks"] = int(num_res_blocks) if num_res_blocks is not None else 2

    attn_scales = source.get("attn_scales", ())
    cleaned["attn_scales"] = tuple(float(v) for v in attn_scales)

    temporal_flags = source.get("temperal_downsample")
    if temporal_flags is None:
        temporal_flags = source.get("temporal_downsample")
    if temporal_flags is None:
        temporal_flags = (False, True, True)
    cleaned["temperal_downsample"] = tuple(bool(v) for v in temporal_flags)

    cleaned["dropout"] = float(source.get("dropout", 0.0))
    cleaned["in_channels"] = int(source.get("in_channels", 3))
    cleaned["out_channels"] = int(source.get("out_channels", 3))
    cleaned["scaling_factor"] = float(source.get("scaling_factor", 1.0))
    cleaned["shift_factor"] = source.get("shift_factor")
    if source.get("latents_mean") is not None:
        cleaned["latents_mean"] = tuple(float(v) for v in source["latents_mean"])
    if source.get("latents_std") is not None:
        cleaned["latents_std"] = tuple(float(v) for v in source["latents_std"])
    if source.get("scale_factor_temporal") is not None:
        cleaned["scale_factor_temporal"] = int(source["scale_factor_temporal"])
    if source.get("scale_factor_spatial") is not None:
        cleaned["scale_factor_spatial"] = int(source["scale_factor_spatial"])
    return cleaned


def is_codex3d_vae_instance(model: object) -> bool:
    return isinstance(model, AutoencoderCodex3D)


__all__ = [
    "AutoencoderCodex3D",
    "is_codex3d_vae_instance",
    "resolve_codex3d_vae_keyspace",
    "sanitize_codex3d_vae_config",
]

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LTX2 video VAE with parser-state-compatible parameter names.
Implements the LTX2 video autoencoder under `apps/**` without importing the official Diffusers LTX2 class while
preserving the parser-produced raw/original state-dict surface (`per_channel_statistics.*`, sequential `down_blocks`
/ `up_blocks`, `res_blocks`, `last_time_embedder`, `last_scale_shift_table`). The public `decode(...)` path
enforces the explicit decode-timestep contract required by `config.timestep_conditioning` and does not expose
unowned slicing, tiling, or framewise execution controls.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2VideoAutoencoder` (class): Config/state-driven native LTX2 video KL autoencoder.
- `__all__` (constant): Explicit public export list for runtime imports.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["Ltx2VideoAutoencoder"]


_CONFIG_META_KEYS = frozenset({"_class_name", "_diffusers_version", "_name_or_path", "architectures"})
_ALLOWED_CLASS_NAMES = frozenset({"AutoencoderKLLTX2Video", "Ltx2VideoAutoencoder"})


def _require_mapping(config: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise RuntimeError(f"LTX2 {label} config must be a mapping, got {type(config).__name__}.")
    return dict(config)


def _validate_class_name(raw: dict[str, Any], *, allowed: Sequence[str], label: str) -> None:
    class_name = raw.get("_class_name")
    if class_name is None:
        return
    if str(class_name) not in allowed:
        raise RuntimeError(
            f"LTX2 {label} config `_class_name` must be one of {tuple(allowed)!r}, got {class_name!r}."
        )


def _reject_unexpected_keys(raw: dict[str, Any], *, allowed: set[str], label: str) -> None:
    unexpected = sorted(set(raw) - allowed - _CONFIG_META_KEYS)
    if unexpected:
        raise RuntimeError(f"LTX2 {label} config has unsupported keys: {unexpected!r}.")


def _as_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"LTX2 config field {name!r} must be an int, got bool.")
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 config field {name!r} must be an int, got {value!r}.") from exc


def _as_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 config field {name!r} must be a float, got {value!r}.") from exc


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise RuntimeError(f"LTX2 config field {name!r} must be a bool, got {value!r}.")


def _as_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"LTX2 config field {name!r} must be a string, got {type(value).__name__}.")
    return value


def _as_tuple(value: Any, *, name: str, item_type: type[int] | type[bool] | type[str]) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"LTX2 config field {name!r} must be a list/tuple, got {type(value).__name__}.")
    caster = {int: _as_int, bool: _as_bool, str: _as_str}[item_type]
    return tuple(caster(v, name=f"{name}[]") for v in value)


def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise RuntimeError(f"LTX2 timestep embedding expects a 1D tensor, got shape={tuple(timesteps.shape)!r}.")
    half_dim = embedding_dim // 2
    if half_dim <= 0:
        raise RuntimeError(f"LTX2 timestep embedding dimension must be >= 2, got {embedding_dim!r}.")

    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class _Timesteps(nn.Module):
    def __init__(self, num_channels: int, *, flip_sin_to_cos: bool, downscale_freq_shift: float) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.flip_sin_to_cos = bool(flip_sin_to_cos)
        self.downscale_freq_shift = float(downscale_freq_shift)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return _get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class _TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class _PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, size_emb_dim: int) -> None:
        super().__init__()
        self.outdim = int(size_emb_dim)
        self.time_proj = _Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = _TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        *,
        timestep: torch.Tensor,
        resolution: torch.Tensor | None,
        aspect_ratio: torch.Tensor | None,
        batch_size: int | None,
        hidden_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        del resolution, aspect_ratio, batch_size
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype or timesteps_proj.dtype))


@dataclass(slots=True)
class _AutoencoderKLOutput:
    latent_dist: "_DiagonalGaussianDistribution"


@dataclass(slots=True)
class _DecoderOutput:
    sample: torch.Tensor


class _DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = bool(deterministic)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean


class _PerChannelStatistics(nn.Module):
    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.register_buffer("channel", torch.arange(latent_channels, dtype=torch.int64), persistent=True)
        self.register_buffer("mean-of-means", torch.zeros(latent_channels, dtype=torch.float32), persistent=True)
        self.register_buffer("mean-of-stds", torch.zeros(latent_channels, dtype=torch.float32), persistent=True)
        self.register_buffer("std-of-means", torch.ones(latent_channels, dtype=torch.float32), persistent=True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        return (x * std) + mean


class _PerChannelRMSNorm(nn.Module):
    def __init__(self, channel_dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.channel_dim = int(channel_dim)
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(hidden_states**2, dim=self.channel_dim, keepdim=True)
        return hidden_states / torch.sqrt(mean_sq + self.eps)


class _VideoCausalConv3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.kernel_size = (
            tuple(int(v) for v in kernel_size)
            if isinstance(kernel_size, (list, tuple))
            else (int(kernel_size), int(kernel_size), int(kernel_size))
        )
        dilation = (
            tuple(int(v) for v in dilation)
            if isinstance(dilation, (list, tuple))
            else (int(dilation), 1, 1)
        )
        stride = (
            tuple(int(v) for v in stride)
            if isinstance(stride, (list, tuple))
            else (int(stride), int(stride), int(stride))
        )
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        time_kernel = self.kernel_size[0]
        if causal:
            pad_left = hidden_states[:, :, :1].repeat(1, 1, time_kernel - 1, 1, 1)
            hidden_states = torch.cat([pad_left, hidden_states], dim=2)
        else:
            half = (time_kernel - 1) // 2
            pad_left = hidden_states[:, :, :1].repeat(1, 1, half, 1, 1)
            pad_right = hidden_states[:, :, -1:].repeat(1, 1, half, 1, 1)
            hidden_states = torch.cat([pad_left, hidden_states, pad_right], dim=2)
        return self.conv(hidden_states)


class _VideoResnetBlock3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        if non_linearity != "swish":
            raise RuntimeError(f"LTX2 video ResNet only supports non_linearity='swish', got {non_linearity!r}.")

        self.nonlinearity = nn.SiLU()
        self.norm1 = _PerChannelRMSNorm()
        self.conv1 = _VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = _PerChannelRMSNorm()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = _VideoCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.norm3: nn.Module | None = None
        self.conv_shortcut: nn.Module | None = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True)
            self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

        self.per_channel_scale1: nn.Parameter | None = None
        self.per_channel_scale2: nn.Parameter | None = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))

        self.scale_shift_table: nn.Parameter | None = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        temb: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        *,
        causal: bool = True,
    ) -> torch.Tensor:
        hidden_states = self.norm1(inputs)
        if self.scale_shift_table is not None:
            if temb is None:
                raise RuntimeError("LTX2 video ResNet block expected timestep conditioning but received temb=None.")
            temb = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
            shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states, causal=causal)

        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape,
                generator=generator,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]

        hidden_states = self.norm2(hidden_states)
        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, causal=causal)

        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape,
                generator=generator,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]

        residual = inputs
        if self.norm3 is not None:
            residual = self.norm3(residual.movedim(1, -1)).movedim(-1, 1)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return hidden_states + residual


class _VideoDownsampler3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int | Sequence[int] = 1,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.stride = (
            tuple(int(v) for v in stride)
            if isinstance(stride, (list, tuple))
            else (int(stride), int(stride), int(stride))
        )
        self.group_size = (in_channels * self.stride[0] * self.stride[1] * self.stride[2]) // out_channels
        conv_out_channels = out_channels // (self.stride[0] * self.stride[1] * self.stride[2])
        self.conv = _VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        hidden_states = torch.cat([hidden_states[:, :, : self.stride[0] - 1], hidden_states], dim=2)
        residual = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        residual = residual.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        residual = residual.unflatten(1, (-1, self.group_size)).mean(dim=2)

        hidden_states = self.conv(hidden_states, causal=causal)
        hidden_states = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        return hidden_states + residual


class _VideoUpsampler3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        stride: int | Sequence[int] = 1,
        residual: bool = False,
        upscale_factor: int = 1,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.stride = (
            tuple(int(v) for v in stride)
            if isinstance(stride, (list, tuple))
            else (int(stride), int(stride), int(stride))
        )
        self.residual = bool(residual)
        self.upscale_factor = int(upscale_factor)
        out_channels = (in_channels * self.stride[0] * self.stride[1] * self.stride[2]) // self.upscale_factor
        self.conv = _VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        batch, channels, frames, height, width = hidden_states.shape
        residual: torch.Tensor | None = None
        if self.residual:
            residual = hidden_states.reshape(
                batch,
                -1,
                self.stride[0],
                self.stride[1],
                self.stride[2],
                frames,
                height,
                width,
            )
            residual = residual.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
            repeats = (self.stride[0] * self.stride[1] * self.stride[2]) // self.upscale_factor
            residual = residual.repeat(1, repeats, 1, 1, 1)
            residual = residual[:, :, self.stride[0] - 1 :]

        hidden_states = self.conv(hidden_states, causal=causal)
        hidden_states = hidden_states.reshape(
            batch,
            -1,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            frames,
            height,
            width,
        )
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        hidden_states = hidden_states[:, :, self.stride[0] - 1 :]
        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states


class _VideoDownBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                _VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        temb: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        for res_block in self.res_blocks:
            hidden_states = res_block(hidden_states, temb, generator, causal=causal)
        return hidden_states


class _VideoMidBlock3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.time_embedder: _PixArtAlphaCombinedTimestepSizeEmbeddings | None = None
        if timestep_conditioning:
            self.time_embedder = _PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)
        self.res_blocks = nn.ModuleList(
            [
                _VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        temb: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        if self.time_embedder is not None:
            if temb is None:
                raise RuntimeError("LTX2 video mid block expected timestep conditioning but received temb=None.")
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)
        for res_block in self.res_blocks:
            hidden_states = res_block(hidden_states, temb, generator, causal=causal)
        return hidden_states


class _VideoUpBlock3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.time_embedder: _PixArtAlphaCombinedTimestepSizeEmbeddings | None = None
        if timestep_conditioning:
            self.time_embedder = _PixArtAlphaCombinedTimestepSizeEmbeddings(out_channels * 4, 0)
        self.conv_in: _VideoResnetBlock3d | None = None
        if in_channels != out_channels:
            self.conv_in = _VideoResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
            )
        self.res_blocks = nn.ModuleList(
            [
                _VideoResnetBlock3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        temb: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        if self.conv_in is not None:
            hidden_states = self.conv_in(hidden_states, temb, generator, causal=causal)
        if self.time_embedder is not None:
            if temb is None:
                raise RuntimeError("LTX2 video up block expected timestep conditioning but received temb=None.")
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)
        for res_block in self.res_blocks:
            hidden_states = res_block(hidden_states, temb, generator, causal=causal)
        return hidden_states


class _VideoEncoder3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 128,
        block_out_channels: Sequence[int] = (256, 512, 1024, 2048),
        down_block_types: Sequence[str] = (
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
        ),
        spatio_temporal_scaling: Sequence[bool] = (True, True, True, True),
        layers_per_block: Sequence[int] = (4, 6, 6, 2, 2),
        downsample_type: Sequence[str] = ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        block_out_channels = tuple(int(v) for v in block_out_channels)
        down_block_types = tuple(str(v) for v in down_block_types)
        spatio_temporal_scaling = tuple(bool(v) for v in spatio_temporal_scaling)
        layers_per_block = tuple(int(v) for v in layers_per_block)
        downsample_type = tuple(str(v) for v in downsample_type)
        if not (
            len(block_out_channels) == len(down_block_types) == len(spatio_temporal_scaling) == len(downsample_type)
        ):
            raise RuntimeError(
                "LTX2 video encoder config mismatch: block_out_channels, down_block_types, "
                "spatio_temporal_scaling, and downsample_type must have equal length."
            )
        if len(layers_per_block) != len(block_out_channels) + 1:
            raise RuntimeError(
                "LTX2 video encoder config mismatch: layers_per_block must have exactly one more entry than block_out_channels."
            )
        for block_type in down_block_types:
            if block_type != "LTX2VideoDownBlock3D":
                raise RuntimeError(f"LTX2 video encoder unsupported down block type: {block_type!r}.")

        self.patch_size = int(patch_size)
        self.patch_size_t = int(patch_size_t)
        self.in_channels = int(in_channels) * self.patch_size**2
        self.is_causal = bool(is_causal)
        output_channel = int(out_channels)
        self.conv_in = _VideoCausalConv3d(
            in_channels=self.in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.down_blocks = nn.ModuleList([])
        current_channels = output_channel
        for index, next_channels in enumerate(block_out_channels):
            self.down_blocks.append(
                _VideoDownBlock3D(
                    in_channels=current_channels,
                    num_layers=layers_per_block[index],
                    resnet_eps=resnet_norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
            if spatio_temporal_scaling[index]:
                pattern = downsample_type[index]
                if pattern == "conv":
                    if current_channels != next_channels:
                        raise RuntimeError(
                            "LTX2 native video encoder does not support conv downsampling when channels change; "
                            f"got in={current_channels} out={next_channels}."
                        )
                    self.down_blocks.append(
                        _VideoCausalConv3d(
                            in_channels=current_channels,
                            out_channels=current_channels,
                            kernel_size=3,
                            stride=(2, 2, 2),
                            spatial_padding_mode=spatial_padding_mode,
                        )
                    )
                elif pattern == "spatial":
                    self.down_blocks.append(
                        _VideoDownsampler3d(
                            in_channels=current_channels,
                            out_channels=next_channels,
                            stride=(1, 2, 2),
                            spatial_padding_mode=spatial_padding_mode,
                        )
                    )
                    current_channels = next_channels
                elif pattern == "temporal":
                    self.down_blocks.append(
                        _VideoDownsampler3d(
                            in_channels=current_channels,
                            out_channels=next_channels,
                            stride=(2, 1, 1),
                            spatial_padding_mode=spatial_padding_mode,
                        )
                    )
                    current_channels = next_channels
                elif pattern == "spatiotemporal":
                    self.down_blocks.append(
                        _VideoDownsampler3d(
                            in_channels=current_channels,
                            out_channels=next_channels,
                            stride=(2, 2, 2),
                            spatial_padding_mode=spatial_padding_mode,
                        )
                    )
                    current_channels = next_channels
                else:
                    raise RuntimeError(f"LTX2 video encoder unsupported downsample_type: {pattern!r}.")
            else:
                if current_channels != next_channels:
                    raise RuntimeError(
                        "LTX2 native video encoder requires matching channels when spatio_temporal_scaling is disabled; "
                        f"got in={current_channels} out={next_channels} at block_index={index}."
                    )
        self.down_blocks.append(
            _VideoMidBlock3d(
                in_channels=current_channels,
                num_layers=layers_per_block[-1],
                resnet_eps=resnet_norm_eps,
                spatial_padding_mode=spatial_padding_mode,
            )
        )
        self.norm_out = _PerChannelRMSNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = _VideoCausalConv3d(
            in_channels=current_channels,
            out_channels=int(out_channels) + 1,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor, *, causal: bool | None = None) -> torch.Tensor:
        patch_size = self.patch_size
        patch_size_t = self.patch_size_t
        batch, channels, frames, height, width = hidden_states.shape
        post_patch_frames = frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        causal = self.is_causal if causal is None else causal
        hidden_states = hidden_states.reshape(
            batch,
            channels,
            post_patch_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        hidden_states = self.conv_in(hidden_states, causal=causal)
        for block in self.down_blocks:
            if isinstance(block, (_VideoDownBlock3D, _VideoMidBlock3d)):
                hidden_states = block(hidden_states, temb=None, generator=None, causal=causal)
            else:
                hidden_states = block(hidden_states, causal=causal)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal)
        last_channel = hidden_states[:, -1:].repeat(1, hidden_states.size(1) - 2, 1, 1, 1)
        return torch.cat([hidden_states, last_channel], dim=1)


class _VideoDecoder3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 128,
        out_channels: int = 3,
        block_out_channels: Sequence[int] = (256, 512, 1024),
        spatio_temporal_scaling: Sequence[bool] = (True, True, True),
        layers_per_block: Sequence[int] = (5, 5, 5, 5),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = False,
        inject_noise: Sequence[bool] = (False, False, False, False),
        timestep_conditioning: bool = False,
        upsample_residual: Sequence[bool] = (True, True, True),
        upsample_factor: Sequence[int] = (2, 2, 2),
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        block_out_channels = tuple(reversed(tuple(int(v) for v in block_out_channels)))
        spatio_temporal_scaling = tuple(reversed(tuple(bool(v) for v in spatio_temporal_scaling)))
        layers_per_block = tuple(reversed(tuple(int(v) for v in layers_per_block)))
        inject_noise = tuple(reversed(tuple(bool(v) for v in inject_noise)))
        upsample_residual = tuple(reversed(tuple(bool(v) for v in upsample_residual)))
        upsample_factor = tuple(reversed(tuple(int(v) for v in upsample_factor)))
        if not (
            len(block_out_channels) == len(spatio_temporal_scaling) == len(upsample_residual) == len(upsample_factor)
        ):
            raise RuntimeError(
                "LTX2 video decoder config mismatch: block_out_channels, spatio_temporal_scaling, "
                "upsample_residual, and upsample_factor must have equal length."
            )
        if len(layers_per_block) != len(block_out_channels) + 1:
            raise RuntimeError(
                "LTX2 video decoder config mismatch: decoder_layers_per_block must have exactly one more entry than decoder_block_out_channels."
            )
        if len(inject_noise) != len(block_out_channels) + 1:
            raise RuntimeError(
                "LTX2 video decoder config mismatch: decoder_inject_noise must have exactly one more entry than decoder_block_out_channels."
            )

        self.patch_size = int(patch_size)
        self.patch_size_t = int(patch_size_t)
        self.out_channels = int(out_channels) * self.patch_size**2
        self.is_causal = bool(is_causal)

        output_channel = block_out_channels[0]
        self.conv_in = _VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.up_blocks = nn.ModuleList(
            [
                _VideoMidBlock3d(
                    in_channels=output_channel,
                    num_layers=layers_per_block[0],
                    resnet_eps=resnet_norm_eps,
                    inject_noise=inject_noise[0],
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            ]
        )
        for index, block_channel in enumerate(block_out_channels):
            if not spatio_temporal_scaling[index]:
                raise RuntimeError(
                    "LTX2 native video decoder currently requires decoder_spatio_temporal_scaling=True for every block; "
                    f"got False at block_index={index}."
                )
            input_channel = output_channel // upsample_factor[index]
            next_output_channel = block_channel // upsample_factor[index]
            if spatio_temporal_scaling[index]:
                self.up_blocks.append(
                    _VideoUpsampler3d(
                        in_channels=next_output_channel * upsample_factor[index],
                        stride=(2, 2, 2),
                        residual=upsample_residual[index],
                        upscale_factor=upsample_factor[index],
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            self.up_blocks.append(
                _VideoUpBlock3d(
                    in_channels=input_channel,
                    out_channels=next_output_channel,
                    num_layers=layers_per_block[index + 1],
                    resnet_eps=resnet_norm_eps,
                    inject_noise=inject_noise[index + 1],
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
            output_channel = next_output_channel
        self.norm_out = _PerChannelRMSNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = _VideoCausalConv3d(
            in_channels=output_channel,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.last_time_embedder: _PixArtAlphaCombinedTimestepSizeEmbeddings | None = None
        self.last_scale_shift_table: nn.Parameter | None = None
        self.timestep_scale_multiplier: nn.Parameter | None = None
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0, dtype=torch.float32))
            self.last_time_embedder = _PixArtAlphaCombinedTimestepSizeEmbeddings(output_channel * 2, 0)
            self.last_scale_shift_table = nn.Parameter(torch.randn(2, output_channel) / output_channel**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        temb: torch.Tensor | None = None,
        causal: bool | None = None,
    ) -> torch.Tensor:
        causal = self.is_causal if causal is None else causal
        hidden_states = self.conv_in(hidden_states, causal=causal)
        if self.timestep_scale_multiplier is not None:
            if temb is None:
                raise RuntimeError("LTX2 video decoder expected timestep conditioning but received temb=None.")
            temb = temb * self.timestep_scale_multiplier
        for block in self.up_blocks:
            if isinstance(block, (_VideoMidBlock3d, _VideoUpBlock3d)):
                hidden_states = block(hidden_states, temb=temb, generator=None, causal=causal)
            else:
                hidden_states = block(hidden_states, causal=causal)
        hidden_states = self.norm_out(hidden_states)
        if self.last_time_embedder is not None:
            if temb is None or self.last_scale_shift_table is None:
                raise RuntimeError("LTX2 video decoder output conditioning is configured but missing state.")
            temb_out = self.last_time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb_out = temb_out.view(hidden_states.size(0), -1, 1, 1, 1).unflatten(1, (2, -1))
            temb_out = temb_out + self.last_scale_shift_table[None, ..., None, None, None]
            shift, scale = temb_out.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal)

        patch_size = self.patch_size
        patch_size_t = self.patch_size_t
        batch, _, frames, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(batch, -1, patch_size_t, patch_size, patch_size, frames, height, width)
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 4, 7, 3).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return hidden_states


class Ltx2VideoAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 128,
        block_out_channels: Sequence[int] = (256, 512, 1024, 2048),
        down_block_types: Sequence[str] = (
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
        ),
        decoder_block_out_channels: Sequence[int] = (256, 512, 1024),
        layers_per_block: Sequence[int] = (4, 6, 6, 2, 2),
        decoder_layers_per_block: Sequence[int] = (5, 5, 5, 5),
        spatio_temporal_scaling: Sequence[bool] = (True, True, True, True),
        decoder_spatio_temporal_scaling: Sequence[bool] = (True, True, True),
        decoder_inject_noise: Sequence[bool] = (False, False, False, False),
        downsample_type: Sequence[str] = ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
        upsample_residual: Sequence[bool] = (True, True, True),
        upsample_factor: Sequence[int] = (2, 2, 2),
        timestep_conditioning: bool = False,
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        scaling_factor: float = 1.0,
        encoder_causal: bool = True,
        decoder_causal: bool = True,
        encoder_spatial_padding_mode: str = "zeros",
        decoder_spatial_padding_mode: str = "reflect",
        spatial_compression_ratio: int | None = None,
        temporal_compression_ratio: int | None = None,
    ) -> None:
        super().__init__()
        block_out_channels = tuple(int(v) for v in block_out_channels)
        down_block_types = tuple(str(v) for v in down_block_types)
        decoder_block_out_channels = tuple(int(v) for v in decoder_block_out_channels)
        layers_per_block = tuple(int(v) for v in layers_per_block)
        decoder_layers_per_block = tuple(int(v) for v in decoder_layers_per_block)
        spatio_temporal_scaling = tuple(bool(v) for v in spatio_temporal_scaling)
        decoder_spatio_temporal_scaling = tuple(bool(v) for v in decoder_spatio_temporal_scaling)
        decoder_inject_noise = tuple(bool(v) for v in decoder_inject_noise)
        downsample_type = tuple(str(v) for v in downsample_type)
        upsample_residual = tuple(bool(v) for v in upsample_residual)
        upsample_factor = tuple(int(v) for v in upsample_factor)
        self.config = SimpleNamespace(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            latent_channels=int(latent_channels),
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            decoder_block_out_channels=decoder_block_out_channels,
            layers_per_block=layers_per_block,
            decoder_layers_per_block=decoder_layers_per_block,
            spatio_temporal_scaling=spatio_temporal_scaling,
            decoder_spatio_temporal_scaling=decoder_spatio_temporal_scaling,
            decoder_inject_noise=decoder_inject_noise,
            downsample_type=downsample_type,
            upsample_residual=upsample_residual,
            upsample_factor=upsample_factor,
            timestep_conditioning=bool(timestep_conditioning),
            patch_size=int(patch_size),
            patch_size_t=int(patch_size_t),
            resnet_norm_eps=float(resnet_norm_eps),
            scaling_factor=float(scaling_factor),
            encoder_causal=bool(encoder_causal),
            decoder_causal=bool(decoder_causal),
            encoder_spatial_padding_mode=str(encoder_spatial_padding_mode),
            decoder_spatial_padding_mode=str(decoder_spatial_padding_mode),
            spatial_compression_ratio=(
                patch_size * 2 ** sum(spatio_temporal_scaling)
                if spatial_compression_ratio is None
                else int(spatial_compression_ratio)
            ),
            temporal_compression_ratio=(
                patch_size_t * 2 ** sum(spatio_temporal_scaling)
                if temporal_compression_ratio is None
                else int(temporal_compression_ratio)
            ),
        )
        self.encoder = _VideoEncoder3d(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            spatio_temporal_scaling=spatio_temporal_scaling,
            layers_per_block=layers_per_block,
            downsample_type=downsample_type,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_eps=resnet_norm_eps,
            is_causal=encoder_causal,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )
        self.decoder = _VideoDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=decoder_block_out_channels,
            spatio_temporal_scaling=decoder_spatio_temporal_scaling,
            layers_per_block=decoder_layers_per_block,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_eps=resnet_norm_eps,
            is_causal=decoder_causal,
            timestep_conditioning=timestep_conditioning,
            inject_noise=decoder_inject_noise,
            upsample_residual=upsample_residual,
            upsample_factor=upsample_factor,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )
        self.per_channel_statistics = _PerChannelStatistics(latent_channels=int(latent_channels))
        self.spatial_compression_ratio = self.config.spatial_compression_ratio
        self.temporal_compression_ratio = self.config.temporal_compression_ratio

    @property
    def latents_mean(self) -> torch.Tensor:
        return self.per_channel_statistics.get_buffer("mean-of-means")

    @property
    def latents_std(self) -> torch.Tensor:
        return self.per_channel_statistics.get_buffer("std-of-means")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Ltx2VideoAutoencoder":
        raw = _require_mapping(config, label="video VAE")
        _validate_class_name(raw, allowed=_ALLOWED_CLASS_NAMES, label="video VAE")
        expected = {
            "in_channels",
            "out_channels",
            "latent_channels",
            "block_out_channels",
            "down_block_types",
            "decoder_block_out_channels",
            "layers_per_block",
            "decoder_layers_per_block",
            "spatio_temporal_scaling",
            "decoder_spatio_temporal_scaling",
            "decoder_inject_noise",
            "downsample_type",
            "upsample_residual",
            "upsample_factor",
            "timestep_conditioning",
            "patch_size",
            "patch_size_t",
            "resnet_norm_eps",
            "scaling_factor",
            "encoder_causal",
            "decoder_causal",
            "encoder_spatial_padding_mode",
            "decoder_spatial_padding_mode",
            "spatial_compression_ratio",
            "temporal_compression_ratio",
        }
        _reject_unexpected_keys(raw, allowed=expected, label="video VAE")
        return cls(
            in_channels=_as_int(raw.get("in_channels", 3), name="in_channels"),
            out_channels=_as_int(raw.get("out_channels", 3), name="out_channels"),
            latent_channels=_as_int(raw.get("latent_channels", 128), name="latent_channels"),
            block_out_channels=_as_tuple(
                raw.get("block_out_channels", (256, 512, 1024, 2048)),
                name="block_out_channels",
                item_type=int,
            ),
            down_block_types=_as_tuple(
                raw.get(
                    "down_block_types",
                    ("LTX2VideoDownBlock3D", "LTX2VideoDownBlock3D", "LTX2VideoDownBlock3D", "LTX2VideoDownBlock3D"),
                ),
                name="down_block_types",
                item_type=str,
            ),
            decoder_block_out_channels=_as_tuple(
                raw.get("decoder_block_out_channels", (256, 512, 1024)),
                name="decoder_block_out_channels",
                item_type=int,
            ),
            layers_per_block=_as_tuple(
                raw.get("layers_per_block", (4, 6, 6, 2, 2)),
                name="layers_per_block",
                item_type=int,
            ),
            decoder_layers_per_block=_as_tuple(
                raw.get("decoder_layers_per_block", (5, 5, 5, 5)),
                name="decoder_layers_per_block",
                item_type=int,
            ),
            spatio_temporal_scaling=_as_tuple(
                raw.get("spatio_temporal_scaling", (True, True, True, True)),
                name="spatio_temporal_scaling",
                item_type=bool,
            ),
            decoder_spatio_temporal_scaling=_as_tuple(
                raw.get("decoder_spatio_temporal_scaling", (True, True, True)),
                name="decoder_spatio_temporal_scaling",
                item_type=bool,
            ),
            decoder_inject_noise=_as_tuple(
                raw.get("decoder_inject_noise", (False, False, False, False)),
                name="decoder_inject_noise",
                item_type=bool,
            ),
            downsample_type=_as_tuple(
                raw.get("downsample_type", ("spatial", "temporal", "spatiotemporal", "spatiotemporal")),
                name="downsample_type",
                item_type=str,
            ),
            upsample_residual=_as_tuple(
                raw.get("upsample_residual", (True, True, True)),
                name="upsample_residual",
                item_type=bool,
            ),
            upsample_factor=_as_tuple(
                raw.get("upsample_factor", (2, 2, 2)),
                name="upsample_factor",
                item_type=int,
            ),
            timestep_conditioning=_as_bool(raw.get("timestep_conditioning", False), name="timestep_conditioning"),
            patch_size=_as_int(raw.get("patch_size", 4), name="patch_size"),
            patch_size_t=_as_int(raw.get("patch_size_t", 1), name="patch_size_t"),
            resnet_norm_eps=_as_float(raw.get("resnet_norm_eps", 1e-6), name="resnet_norm_eps"),
            scaling_factor=_as_float(raw.get("scaling_factor", 1.0), name="scaling_factor"),
            encoder_causal=_as_bool(raw.get("encoder_causal", True), name="encoder_causal"),
            decoder_causal=_as_bool(raw.get("decoder_causal", True), name="decoder_causal"),
            encoder_spatial_padding_mode=_as_str(
                raw.get("encoder_spatial_padding_mode", "zeros"),
                name="encoder_spatial_padding_mode",
            ),
            decoder_spatial_padding_mode=_as_str(
                raw.get("decoder_spatial_padding_mode", "reflect"),
                name="decoder_spatial_padding_mode",
            ),
            spatial_compression_ratio=(
                None
                if raw.get("spatial_compression_ratio") is None
                else _as_int(raw.get("spatial_compression_ratio"), name="spatial_compression_ratio")
            ),
            temporal_compression_ratio=(
                None
                if raw.get("temporal_compression_ratio") is None
                else _as_int(raw.get("temporal_compression_ratio"), name="temporal_compression_ratio")
            ),
        )

    def load_strict_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(f"LTX2 video VAE strict load expects a mapping, got {type(state_dict).__name__}.")
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "LTX2 video VAE strict load failed: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
            )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor, *, return_dict: bool = True) -> _AutoencoderKLOutput | tuple[_DiagonalGaussianDistribution]:
        hidden_states = self._encode(x)
        posterior = _DiagonalGaussianDistribution(hidden_states)
        if not return_dict:
            return (posterior,)
        return _AutoencoderKLOutput(latent_dist=posterior)

    def _normalize_decode_timestep(
        self,
        timestep: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        expects_timestep = bool(getattr(self.config, "timestep_conditioning", False))
        if timestep is None:
            if expects_timestep:
                raise RuntimeError(
                    "LTX2 video VAE decode requires `timestep` when `config.timestep_conditioning=True`."
                )
            return None
        if not expects_timestep:
            raise RuntimeError(
                "LTX2 video VAE decode received `timestep`, but `config.timestep_conditioning=False`."
            )
        if not isinstance(timestep, torch.Tensor):
            raise RuntimeError(
                f"LTX2 video VAE decode expected `timestep` as torch.Tensor, got {type(timestep).__name__}."
            )
        if timestep.ndim == 0:
            timestep = timestep.reshape(1)
        if timestep.ndim != 1:
            raise RuntimeError(
                f"LTX2 video VAE decode expected `timestep` to be 1D, got shape={tuple(timestep.shape)!r}."
            )
        if int(timestep.shape[0]) != int(batch_size):
            raise RuntimeError(
                "LTX2 video VAE decode expected `timestep` batch size to match latents: "
                f"batch={int(batch_size)} timestep={int(timestep.shape[0])}."
            )
        return timestep.to(device=device, dtype=dtype)

    def _decode(self, z: torch.Tensor, *, timestep: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(z, temb=timestep)

    def decode(
        self,
        z: torch.Tensor,
        timestep: torch.Tensor | None = None,
        *,
        return_dict: bool = True,
    ) -> _DecoderOutput | tuple[torch.Tensor]:
        decode_timestep = self._normalize_decode_timestep(
            timestep,
            batch_size=int(z.shape[0]),
            device=z.device,
            dtype=z.dtype,
        )
        decoded = self._decode(z, timestep=decode_timestep)
        if not return_dict:
            return (decoded,)
        return _DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        *,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> _DecoderOutput | tuple[torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        latent = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        decode_timestep = None
        if bool(getattr(self.config, "timestep_conditioning", False)):
            decode_timestep = torch.zeros(int(latent.shape[0]), device=latent.device, dtype=latent.dtype)
        decoded = self.decode(latent, timestep=decode_timestep)
        if not return_dict:
            return (decoded.sample,)
        return decoded

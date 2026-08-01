"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Family-agnostic native LDM 2D VAE runtime lane for latent encode/decode.
Defines shared LDM VAE blocks (ResNet/Attention/Encoder/Decoder), the canonical `AutoencoderKL_LDM` class, and config sanitizers used across
SDXL/Flux/ZImage/WAN lanes without WAN-family ownership coupling. Posterior variance preserves exact `exp(logvar)` semantics through lazy evaluation.

Symbols (top-level; keep in sync; no ghosts):
- `nonlinearity` (function): Activation helper (SiLU-like: `x * sigmoid(x)`) used across blocks.
- `Normalize` (function): GroupNorm helper used across residual/attention blocks.
- `DiagonalGaussianDistribution` (class): Gaussian distribution wrapper for VAE posterior/prior (sampling + mode), used by KL-style VAEs.
- `Upsample` (class): Upsampling block (optionally with convolution) used in the decoder path.
- `Downsample` (class): Downsampling block (optionally with convolution) used in the encoder path.
- `ResnetBlock` (class): Residual block used in encoder/decoder stacks.
- `AttnBlock` (class): Spatial attention block used at configured resolutions (uses single-head attention helper).
- `Encoder` (class): VAE encoder mapping images → latent moments (mu/logvar) via conv/resnet/attn stacks.
- `Decoder` (class): VAE decoder mapping latents → images via conv/resnet/attn/upsample stacks.
- `AutoencoderKL_LDM` (class): Full VAE module (`ConfigMixin`); wires encoder/decoder + quantization layers and exposes
  encode/decode APIs (contains nested helpers for config defaults and tensor plumbing).
- `sanitize_ldm_vae_config` (function): Removes unsupported diffusers-only keys and normalizes alias fields for native constructor parity.
- `is_ldm_native_vae_instance` (function): Returns True when a model instance belongs to the native LDM VAE lane.
- `__all__` (constant): Explicit export list.
"""

import logging
from typing import Any, Mapping
from apps.backend.runtime.logging import get_backend_logger

import torch
import numpy as np
from apps.backend.runtime.attention import attention_function_single_head_spatial
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch import nn

_log = get_backend_logger("backend.runtime.vae.ldm")


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        if self.deterministic:
            self.std = torch.zeros_like(self.mean)

    @property
    def var(self):
        if self.deterministic:
            return torch.zeros_like(self.mean)
        return torch.exp(self.logvar)

    def sample(self):
        noise = torch.randn(
            self.mean.shape,
            device=self.parameters.device,
            dtype=self.mean.dtype,
        )
        x = self.mean + self.std * noise
        return x

    def mode(self):
        return self.mean


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        try:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        except Exception:
            b, c, h, w = x.shape
            out = torch.empty((b, c, h * 2, w * 2), dtype=x.dtype, layout=x.layout, device=x.device)
            split = 8
            channel_chunk = max(out.shape[1] // split, 1)
            for i in range(0, out.shape[1], channel_chunk):
                out[:, i:i + channel_chunk] = torch.nn.functional.interpolate(
                    x[:, i:i + channel_chunk].to(torch.float32),
                    scale_factor=2.0,
                    mode="nearest",
                ).to(x.dtype)
            del x
            x = out

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.swish = torch.nn.SiLU(inplace=True)
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(self.swish(temb))[:, :, None, None]
        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        h_ = attention_function_single_head_spatial(q, k, v)
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla", **kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False, **kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        _log.info("Working with z of shape %s = %d dimensions.", self.z_shape, np.prod(self.z_shape))

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z, **kwargs):
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class AutoencoderKL_LDM(nn.Module, ConfigMixin):
    config_name = 'config.json'

    @register_to_config
    def __init__(self, in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",), block_out_channels=(64,), layers_per_block=1, act_fn="silu", latent_channels=4, norm_num_groups=32, sample_size=32, scaling_factor=0.18215, shift_factor=None, latents_mean=None, latents_std=None, force_upcast=True, use_quant_conv=True, use_post_quant_conv=True):
        super().__init__()
        ch = block_out_channels[0]
        ch_mult = [x // ch for x in block_out_channels]
        self.encoder = Encoder(double_z=True, z_channels=latent_channels, resolution=256, in_channels=in_channels, out_ch=out_channels, ch=ch, ch_mult=ch_mult, num_res_blocks=layers_per_block, attn_resolutions=[], dropout=0.0)
        self.decoder = Decoder(double_z=True, z_channels=latent_channels, resolution=256, in_channels=in_channels, out_ch=out_channels, ch=ch, ch_mult=ch_mult, num_res_blocks=layers_per_block, attn_resolutions=[], dropout=0.0)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None
        self.embed_dim = latent_channels
        self.scaling_factor = scaling_factor
        if shift_factor is None:
            self.shift_factor = None
        else:
            try:
                self.shift_factor = float(shift_factor)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid VAE shift_factor: {shift_factor!r}.") from exc
            if not np.isfinite(self.shift_factor):
                raise RuntimeError(f"Invalid VAE shift_factor: {shift_factor!r} (must be finite).")
        self._shift_factor_value = 0.0 if self.shift_factor is None else self.shift_factor

    def encode(self, x, regulation=None):
        z = self.encoder(x)

        if self.quant_conv is not None:
            z = self.quant_conv(z)

        posterior = DiagonalGaussianDistribution(z)
        if regulation is not None:
            return regulation(posterior)
        else:
            return posterior.sample()

    def decode(self, z):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        x = self.decoder(z)
        return x

    def process_in(self, latent):
        return (latent - self._shift_factor_value) * self.scaling_factor

    def process_out(self, latent):
        return (latent / self.scaling_factor) + self._shift_factor_value


def sanitize_ldm_vae_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(config)
    # LDM native lane is always mid-attn capable; this diffusers flag is redundant
    # and unsupported by the native constructor.
    cleaned.pop("mid_block_add_attention", None)
    if "latent_channels" not in cleaned and "z_dim" in cleaned:
        try:
            cleaned["latent_channels"] = int(cleaned["z_dim"])
        except (TypeError, ValueError):
            pass
    if "block_out_channels" not in cleaned and "base_dim" in cleaned and "dim_mult" in cleaned:
        try:
            base_dim = int(cleaned["base_dim"])
            multipliers = tuple(int(value) for value in cleaned["dim_mult"])
            if base_dim > 0 and multipliers:
                cleaned["block_out_channels"] = tuple(base_dim * value for value in multipliers)
        except (TypeError, ValueError):
            pass
    if "layers_per_block" not in cleaned and "num_res_blocks" in cleaned:
        try:
            layers = int(cleaned["num_res_blocks"])
            if layers > 0:
                cleaned["layers_per_block"] = layers
        except (TypeError, ValueError):
            pass
    return cleaned


def is_ldm_native_vae_instance(model: object) -> bool:
    return isinstance(model, AutoencoderKL_LDM)


__all__ = [
    "AttnBlock",
    "AutoencoderKL_LDM",
    "Decoder",
    "DiagonalGaussianDistribution",
    "Downsample",
    "Encoder",
    "Normalize",
    "ResnetBlock",
    "Upsample",
    "is_ldm_native_vae_instance",
    "nonlinearity",
    "sanitize_ldm_vae_config",
]

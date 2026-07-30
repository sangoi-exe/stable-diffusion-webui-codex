"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native Qwen Image Edit-2511 weighted transformer layers.
Implements the exact dual-stream modulation, Q/K RMS normalization, joint attention, feed-forward, timestep embedding,
and final adaptive normalization topology used by the 1,933-key Edit-2511 transformer checkpoint.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageAdaLayerNormContinuous` (class): Final timestep-conditioned layer normalization.
- `QwenImageAttention` (class): Native Qwen joint text/image attention with exact checkpoint-owned projections.
- `QwenImageFeedForward` (class): Exact GELU-approximate feed-forward topology with Diffusers-compatible key names.
- `QwenImageRMSNorm` (class): RMS normalization with operation-context-aware parameter construction.
- `QwenImageTimestepEmbedding` (class): Sinusoidal timestep projection and two-layer embedding MLP.
- `QwenImageTransformerBlock` (class): One exact dual-stream Qwen transformer block.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from apps.backend.runtime.attention import attention_function_pre_shaped
from apps.backend.runtime.ops.operations import get_operation_context

from .transformer_rope import apply_qwen_image_rotary_embedding


def _operation_parameter(
    shape: int | tuple[int, ...],
    *,
    fill: float,
) -> nn.Parameter:
    context = get_operation_context()
    device = context.device
    dtype = context.dtype or torch.float32
    if fill == 0.0:
        value = torch.zeros(shape, device=device, dtype=dtype)
    elif fill == 1.0:
        value = torch.ones(shape, device=device, dtype=dtype)
    else:
        value = torch.full(shape, fill, device=device, dtype=dtype)
    return nn.Parameter(value)


class QwenImageRMSNorm(nn.Module):
    """RMS normalization matching the Qwen Image checkpoint contract."""

    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = _operation_parameter(int(dim), fill=1.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_floating_point():
            raise TypeError(
                "Qwen Image RMSNorm expects floating-point hidden states; "
                f"got dtype={hidden_states.dtype}."
            )
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        weight = self.weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return normalized.to(dtype=hidden_states.dtype) * weight


def _timestep_embedding(
    timesteps: torch.Tensor,
    *,
    embedding_dim: int = 256,
    max_period: int = 10_000,
    scale: float = 1_000.0,
) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise RuntimeError(
            "Qwen Image timestep projection expects a 1D tensor; "
            f"got shape={tuple(timesteps.shape)}."
        )
    half_dim = int(embedding_dim) // 2
    exponent = -math.log(int(max_period)) * torch.arange(
        half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / float(half_dim)
    frequencies = torch.exp(exponent).to(dtype=timesteps.dtype)
    phases = float(scale) * timesteps[:, None].float() * frequencies[None, :]
    embedding = torch.cat((torch.cos(phases), torch.sin(phases)), dim=-1)
    if int(embedding_dim) % 2:
        embedding = nn.functional.pad(embedding, (0, 1))
    return embedding


class _QwenImageTimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(256, int(embedding_dim), bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(int(embedding_dim), int(embedding_dim), bias=True)

    def forward(self, projected_timestep: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(projected_timestep)))


class QwenImageTimestepEmbedding(nn.Module):
    """Qwen timestep projection with exact checkpoint-owned linear names."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = _QwenImageTimestepEmbedder(int(embedding_dim))

    def forward(self, timestep: torch.Tensor, *, hidden_dtype: torch.dtype) -> torch.Tensor:
        projected = _timestep_embedding(timestep)
        return self.timestep_embedder(projected.to(dtype=hidden_dtype))


class QwenImageAttention(nn.Module):
    """Exact native Qwen dual-stream joint-attention projection owner."""

    def __init__(self, *, dim: int, heads: int, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        inner_dim = self.heads * self.head_dim
        if inner_dim != int(dim):
            raise RuntimeError(
                f"Qwen Image attention dimensions are inconsistent: dim={dim} heads={heads} head_dim={head_dim}."
            )

        self.to_q = nn.Linear(int(dim), inner_dim, bias=True)
        self.to_k = nn.Linear(int(dim), inner_dim, bias=True)
        self.to_v = nn.Linear(int(dim), inner_dim, bias=True)
        self.add_q_proj = nn.Linear(int(dim), inner_dim, bias=True)
        self.add_k_proj = nn.Linear(int(dim), inner_dim, bias=True)
        self.add_v_proj = nn.Linear(int(dim), inner_dim, bias=True)
        self.norm_q = QwenImageRMSNorm(self.head_dim, eps=eps)
        self.norm_k = QwenImageRMSNorm(self.head_dim, eps=eps)
        self.norm_added_q = QwenImageRMSNorm(self.head_dim, eps=eps)
        self.norm_added_k = QwenImageRMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList((nn.Linear(inner_dim, int(dim), bias=True), nn.Dropout(0.0)))
        self.to_add_out = nn.Linear(inner_dim, int(dim), bias=True)

    def _heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.unflatten(-1, (self.heads, self.head_dim))

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
        *,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_sequence_length = int(text_hidden_states.shape[1])

        image_query = self.norm_q(self._heads(self.to_q(image_hidden_states)))
        image_key = self.norm_k(self._heads(self.to_k(image_hidden_states)))
        image_value = self._heads(self.to_v(image_hidden_states))
        text_query = self.norm_added_q(self._heads(self.add_q_proj(text_hidden_states)))
        text_key = self.norm_added_k(self._heads(self.add_k_proj(text_hidden_states)))
        text_value = self._heads(self.add_v_proj(text_hidden_states))

        image_frequencies, text_frequencies = rotary_embedding
        image_query = apply_qwen_image_rotary_embedding(image_query, image_frequencies)
        image_key = apply_qwen_image_rotary_embedding(image_key, image_frequencies)
        text_query = apply_qwen_image_rotary_embedding(text_query, text_frequencies)
        text_key = apply_qwen_image_rotary_embedding(text_key, text_frequencies)

        joint_query = torch.cat((text_query, image_query), dim=1).transpose(1, 2).contiguous()
        joint_key = torch.cat((text_key, image_key), dim=1).transpose(1, 2).contiguous()
        joint_value = torch.cat((text_value, image_value), dim=1).transpose(1, 2).contiguous()
        joint_output = attention_function_pre_shaped(
            joint_query,
            joint_key,
            joint_value,
            mask=attention_mask,
            is_causal=False,
        )
        joint_output = joint_output.transpose(1, 2).reshape(
            int(joint_output.shape[0]),
            int(joint_output.shape[2]),
            self.heads * self.head_dim,
        )
        joint_output = joint_output.to(dtype=joint_query.dtype)

        text_output = self.to_add_out(joint_output[:, :text_sequence_length].contiguous())
        image_output = self.to_out[0](joint_output[:, text_sequence_length:].contiguous())
        image_output = self.to_out[1](image_output)
        return image_output, text_output


class _QwenImageGELUProjection(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(dim), int(inner_dim), bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(self.proj(hidden_states), approximate="tanh")


class QwenImageFeedForward(nn.Module):
    """Checkpoint-compatible Qwen feed-forward topology."""

    def __init__(self, dim: int, *, multiplier: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim) * int(multiplier)
        self.net = nn.Sequential(
            _QwenImageGELUProjection(int(dim), inner_dim),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, int(dim), bias=True),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class QwenImageTransformerBlock(nn.Module):
    """One Edit-2511 dual-stream block with zero-conditioned reference tokens."""

    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        zero_cond_t: bool,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(int(dim), 6 * int(dim), bias=True))
        self.img_norm1 = nn.LayerNorm(int(dim), elementwise_affine=False, eps=float(eps))
        self.attn = QwenImageAttention(
            dim=int(dim),
            heads=int(num_attention_heads),
            head_dim=int(attention_head_dim),
            eps=float(eps),
        )
        self.img_norm2 = nn.LayerNorm(int(dim), elementwise_affine=False, eps=float(eps))
        self.img_mlp = QwenImageFeedForward(int(dim))

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(int(dim), 6 * int(dim), bias=True))
        self.txt_norm1 = nn.LayerNorm(int(dim), elementwise_affine=False, eps=float(eps))
        self.txt_norm2 = nn.LayerNorm(int(dim), elementwise_affine=False, eps=float(eps))
        self.txt_mlp = QwenImageFeedForward(int(dim))
        self.zero_cond_t = bool(zero_cond_t)

    @staticmethod
    def _modulate(
        hidden_states: torch.Tensor,
        parameters: torch.Tensor,
        *,
        token_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = parameters.chunk(3, dim=-1)
        if token_index is None:
            return (
                hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1),
                gate.unsqueeze(1),
            )

        if int(shift.shape[0]) % 2:
            raise RuntimeError(
                "Qwen Image zero-conditioned modulation requires a doubled timestep batch; "
                f"got modulation batch={int(shift.shape[0])}."
            )
        batch_size = int(shift.shape[0]) // 2
        if tuple(token_index.shape) != (batch_size, int(hidden_states.shape[1])):
            raise RuntimeError(
                "Qwen Image modulation index shape mismatch: "
                f"got={tuple(token_index.shape)} expected={(batch_size, int(hidden_states.shape[1]))}."
            )
        shift_generated, shift_reference = shift.chunk(2, dim=0)
        scale_generated, scale_reference = scale.chunk(2, dim=0)
        gate_generated, gate_reference = gate.chunk(2, dim=0)
        selector = token_index.unsqueeze(-1)
        shift_value = torch.where(
            selector == 0,
            shift_generated.unsqueeze(1),
            shift_reference.unsqueeze(1),
        )
        scale_value = torch.where(
            selector == 0,
            scale_generated.unsqueeze(1),
            scale_reference.unsqueeze(1),
        )
        gate_value = torch.where(
            selector == 0,
            gate_generated.unsqueeze(1),
            gate_reference.unsqueeze(1),
        )
        return hidden_states * (1 + scale_value) + shift_value, gate_value

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
        *,
        timestep_embedding: torch.Tensor,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        modulate_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_modulation = self.img_mod(timestep_embedding)
        text_timestep_embedding = (
            timestep_embedding.chunk(2, dim=0)[0] if self.zero_cond_t else timestep_embedding
        )
        text_modulation = self.txt_mod(text_timestep_embedding)
        image_modulation_1, image_modulation_2 = image_modulation.chunk(2, dim=-1)
        text_modulation_1, text_modulation_2 = text_modulation.chunk(2, dim=-1)

        image_attention_input, image_gate_1 = self._modulate(
            self.img_norm1(image_hidden_states),
            image_modulation_1,
            token_index=modulate_index,
        )
        text_attention_input, text_gate_1 = self._modulate(
            self.txt_norm1(text_hidden_states),
            text_modulation_1,
            token_index=None,
        )
        image_attention, text_attention = self.attn(
            image_attention_input,
            text_attention_input,
            rotary_embedding=rotary_embedding,
            attention_mask=attention_mask,
        )
        image_hidden_states = image_hidden_states + image_gate_1 * image_attention
        text_hidden_states = text_hidden_states + text_gate_1 * text_attention

        image_mlp_input, image_gate_2 = self._modulate(
            self.img_norm2(image_hidden_states),
            image_modulation_2,
            token_index=modulate_index,
        )
        text_mlp_input, text_gate_2 = self._modulate(
            self.txt_norm2(text_hidden_states),
            text_modulation_2,
            token_index=None,
        )
        image_hidden_states = image_hidden_states + image_gate_2 * self.img_mlp(image_mlp_input)
        text_hidden_states = text_hidden_states + text_gate_2 * self.txt_mlp(text_mlp_input)

        if image_hidden_states.dtype == torch.float16:
            image_hidden_states = image_hidden_states.clamp(-65_504, 65_504)
        if text_hidden_states.dtype == torch.float16:
            text_hidden_states = text_hidden_states.clamp(-65_504, 65_504)
        return text_hidden_states, image_hidden_states


class QwenImageAdaLayerNormContinuous(nn.Module):
    """Final Qwen adaptive layer norm with exact `norm_out.linear.*` keys."""

    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(int(dim), 2 * int(dim), bias=True)
        self.norm = nn.LayerNorm(int(dim), elementwise_affine=False, eps=float(eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        modulation = self.linear(self.silu(conditioning_embedding).to(dtype=hidden_states.dtype))
        scale, shift = modulation.chunk(2, dim=1)
        return self.norm(hidden_states) * (1 + scale[:, None, :]) + shift[:, None, :]


__all__ = [
    "QwenImageAdaLayerNormContinuous",
    "QwenImageAttention",
    "QwenImageFeedForward",
    "QwenImageRMSNorm",
    "QwenImageTimestepEmbedding",
    "QwenImageTransformerBlock",
]

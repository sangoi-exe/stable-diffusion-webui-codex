"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend API-owned WAN video request-key contracts for strict route validation.
Defines the shared top-level/stage allowlists consumed by generation routers for txt2vid/img2vid unknown-key rejection, keeps WAN stage payloads on the
canonical `loras[]` contract (no legacy single-field stage LoRA keys), distinguishes the exact WAN22 lane owners (`wan_single` for 5B,
`wan_high`/`wan_low` for 14B), and exposes the legacy alias mapping used to fail loud on removed WAN request keys before task dispatch.

Symbols (top-level; keep in sync; no ghosts):
- `WanVideoRequestKeys` (dataclass): Canonical WAN video request-key allowlists for txt2vid/img2vid and WAN stage controls.
- `WAN_VIDEO_REQUEST_KEYS` (constant): Singleton request-key map used by WAN video request validators.
- `WAN_VIDEO_LEGACY_REQUEST_KEY_EQUIVALENTS` (constant): Legacy WAN request-key aliases mapped to their canonical replacements.
- `canonical_wan_video_request_key` (function): Returns the canonical WAN request key for a raw input key.
- `legacy_wan_video_request_key_alias_target` (function): Returns the canonical replacement for a removed legacy WAN request key, if any.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import FrozenSet


@dataclass(frozen=True)
class WanVideoRequestKeys:
    """Canonical WAN video request-key allowlists used by generation routers."""

    DEVICE: FrozenSet[str] = frozenset({"device"})
    REVISION: FrozenSet[str] = frozenset({"settings_revision"})
    VIDEO_EXPORT: FrozenSet[str] = frozenset(
        {
            "video_return_frames",
            "video_filename_prefix",
            "video_format",
            "video_pix_fmt",
            "video_crf",
            "video_loop_count",
            "video_pingpong",
            "video_save_metadata",
            "video_save_output",
            "video_trim_to_audio",
        }
    )
    VIDEO_INTERPOLATION: FrozenSet[str] = frozenset({"video_interpolation"})
    WAN_STAGE_CONTAINERS: FrozenSet[str] = frozenset({"wan_single", "wan_high", "wan_low"})
    WAN_SINGLE_ALLOWED: FrozenSet[str] = frozenset(
        {
            "model_sha",
            "model_dir",
            "loras",
            "flow_shift",
        }
    )
    WAN_HIGH_ALLOWED: FrozenSet[str] = frozenset(
        {
            "model_sha",
            "model_dir",
            "loras",
            "flow_shift",
        }
    )
    WAN_LOW_ALLOWED: FrozenSet[str] = frozenset(
        {
            "model_sha",
            "model_dir",
            "prompt",
            "negative_prompt",
            "sampler",
            "scheduler",
            "steps",
            "cfg_scale",
            "seed",
            "lightning",
            "loras",
            "flow_shift",
        }
    )
    WAN_ASSETS: FrozenSet[str] = frozenset(
        {
            "wan_format",
            "wan_metadata_repo",
            "wan_metadata_dir",
            "wan_vae_sha",
            "wan_tenc_sha",
            "wan_vae_path",
            "wan_text_encoder_path",
            "wan_text_encoder_dir",
        }
    )
    GGUF_RUNTIME: FrozenSet[str] = frozenset(
        {
            "gguf_offload",
            "gguf_offload_level",
            "gguf_sdpa_policy",
            "gguf_attention_mode",
            "gguf_attn_chunk",
            "gguf_cache_policy",
            "gguf_cache_limit_mb",
            "gguf_log_mem_interval",
            "gguf_te_device",
        }
    )
    TXT2VID: FrozenSet[str] = frozenset(
        {
            "txt2vid_prompt",
            "txt2vid_neg_prompt",
            "txt2vid_width",
            "txt2vid_height",
            "txt2vid_steps",
            "txt2vid_fps",
            "txt2vid_num_frames",
            "txt2vid_sampler",
            "txt2vid_scheduler",
            "txt2vid_seed",
            "txt2vid_cfg_scale",
            "txt2vid_styles",
        }
    )
    IMG2VID: FrozenSet[str] = frozenset(
        {
            "img2vid_prompt",
            "img2vid_neg_prompt",
            "img2vid_width",
            "img2vid_height",
            "img2vid_steps",
            "img2vid_fps",
            "img2vid_num_frames",
            "img2vid_sampler",
            "img2vid_scheduler",
            "img2vid_seed",
            "img2vid_cfg_scale",
            "img2vid_styles",
            "img2vid_init_image",
            "img2vid_chunk_frames",
            "img2vid_overlap_frames",
            "img2vid_anchor_alpha",
            "img2vid_reset_anchor_to_base",
            "img2vid_chunk_seed_mode",
            "img2vid_chunk_buffer_mode",
            "img2vid_mode",
            "img2vid_window_frames",
            "img2vid_window_stride",
            "img2vid_window_commit_frames",
            "img2vid_image_scale",
            "img2vid_crop_offset_x",
            "img2vid_crop_offset_y",
        }
    )

    @property
    def COMMON(self) -> FrozenSet[str]:
        return (
            self.DEVICE
            | self.REVISION
            | self.VIDEO_EXPORT
            | self.VIDEO_INTERPOLATION
            | self.WAN_STAGE_CONTAINERS
            | self.WAN_ASSETS
            | self.GGUF_RUNTIME
        )

    @property
    def TXT2VID_ALL(self) -> FrozenSet[str]:
        return self.COMMON | self.TXT2VID

    @property
    def IMG2VID_ALL(self) -> FrozenSet[str]:
        return self.COMMON | self.IMG2VID


WAN_VIDEO_REQUEST_KEYS = WanVideoRequestKeys()


WAN_VIDEO_LEGACY_REQUEST_KEY_EQUIVALENTS = MappingProxyType(
    {
        "wan_tokenizer_dir": "wan_metadata_dir",
        "txt2vid_sampling": "txt2vid_sampler",
        "img2vid_sampling": "img2vid_sampler",
    }
)


def canonical_wan_video_request_key(raw_key: str) -> str:
    key = str(raw_key)
    return WAN_VIDEO_LEGACY_REQUEST_KEY_EQUIVALENTS.get(key, key)


def legacy_wan_video_request_key_alias_target(raw_key: str) -> str | None:
    key = str(raw_key)
    return WAN_VIDEO_LEGACY_REQUEST_KEY_EQUIVALENTS.get(key)


__all__ = [
    "WanVideoRequestKeys",
    "WAN_VIDEO_REQUEST_KEYS",
    "WAN_VIDEO_LEGACY_REQUEST_KEY_EQUIVALENTS",
    "canonical_wan_video_request_key",
    "legacy_wan_video_request_key_alias_target",
]

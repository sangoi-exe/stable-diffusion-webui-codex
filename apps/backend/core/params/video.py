"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed video export, interpolation, and SeedVR2 option payloads.
Holds ffmpeg export options, optional interpolation settings, and the dedicated SeedVR2 runtime controls used by video-upscale tasks.

Symbols (top-level; keep in sync; no ghosts):
- `VideoExportOptions` (dataclass): Export parameters for video output (format/codec knobs + metadata/save flags).
- `VideoInterpolationOptions` (dataclass): Optional interpolation settings (enable + model + times).
- `SeedVR2UpscaleOptions` (dataclass): Required SeedVR2 model selection plus curated runtime controls for the dedicated video-upscale task.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class VideoExportOptions:
    filename_prefix: Optional[str] = None
    format: Optional[str] = None
    pix_fmt: Optional[str] = None
    crf: Optional[int] = None
    loop_count: Optional[int] = None
    pingpong: Optional[bool] = None
    save_metadata: Optional[bool] = None
    save_output: Optional[bool] = None
    trim_to_audio: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class VideoInterpolationOptions:
    enabled: bool = False
    model: Optional[str] = None
    times: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True, slots=True)
class SeedVR2UpscaleOptions:
    dit_model: str
    resolution: Optional[int] = None
    max_resolution: Optional[int] = None
    batch_size: Optional[int] = None
    uniform_batch_size: Optional[bool] = None
    temporal_overlap: Optional[int] = None
    prepend_frames: Optional[int] = None
    color_correction: Optional[str] = None
    input_noise_scale: Optional[float] = None
    latent_noise_scale: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


__all__ = ["VideoExportOptions", "VideoInterpolationOptions", "SeedVR2UpscaleOptions"]

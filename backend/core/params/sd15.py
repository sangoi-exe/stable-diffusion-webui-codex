from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import BaseParams, ValidationError
from ..enums import SamplerName, SchedulerName


@dataclass(frozen=True)
class SD15Txt2ImgParams(BaseParams):
    prompt: str
    width: int
    height: int
    steps: int
    sampler: str
    scheduler: str
    seed: int
    guidance_scale: float


@dataclass(frozen=True)
class SD15Img2ImgParams(BaseParams):
    prompt: str
    init_image: object
    width: int
    height: int
    steps: int
    sampler: str
    scheduler: str
    seed: int
    guidance_scale: float
    denoise_strength: float


@dataclass(frozen=True)
class SD15InpaintParams(SD15Img2ImgParams):
    mask: object


@dataclass(frozen=True)
class SD15UpscaleParams(SD15Txt2ImgParams):
    hr_scale: float
    hr_denoise: float
    hr_upscaler: str
    hr_steps: int


from __future__ import annotations

from dataclasses import dataclass

from .base import BaseParams


@dataclass(frozen=True)
class SDXLTxt2ImgParams(BaseParams):
    prompt: str
    width: int
    height: int
    steps: int
    sampler: str
    scheduler: str
    seed: int
    guidance_scale: float


@dataclass(frozen=True)
class SDXLImg2ImgParams(BaseParams):
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
class SDXLInpaintParams(SDXLImg2ImgParams):
    mask: object


@dataclass(frozen=True)
class SDXLUpscaleParams(SDXLTxt2ImgParams):
    hr_scale: float
    hr_denoise: float
    hr_upscaler: str
    hr_steps: int


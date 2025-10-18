from __future__ import annotations

from dataclasses import dataclass

from .base import BaseParams


@dataclass(frozen=True)
class FluxTxt2ImgParams(BaseParams):
    prompt: str
    width: int
    height: int
    steps: int
    sampler: str
    scheduler: str
    seed: int
    guidance_scale: float
    distilled_cfg_scale: float | None


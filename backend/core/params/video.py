from __future__ import annotations

from dataclasses import dataclass

from .base import BaseParams


@dataclass(frozen=True)
class SVDImg2VidParams(BaseParams):
    init_image: object
    width: int
    height: int
    steps: int
    fps: int
    num_frames: int
    sampler: str
    scheduler: str
    seed: int | None
    guidance_scale: float | None
    motion_strength: float | None


@dataclass(frozen=True)
class HunyuanTxt2VidParams(BaseParams):
    prompt: str
    width: int
    height: int
    steps: int
    fps: int
    num_frames: int
    sampler: str
    scheduler: str
    seed: int | None
    guidance_scale: float | None


@dataclass(frozen=True)
class HunyuanImg2VidParams(BaseParams):
    init_image: object
    width: int
    height: int
    steps: int
    fps: int
    num_frames: int
    sampler: str
    scheduler: str
    seed: int | None
    guidance_scale: float | None
    motion_strength: float | None


@dataclass(frozen=True)
class WanTI2V5BParamsT2V(BaseParams):
    prompt: str
    width: int
    height: int
    steps: int
    fps: int
    num_frames: int
    sampler: str
    scheduler: str
    seed: int | None
    guidance_scale: float | None


@dataclass(frozen=True)
class WanTI2V5BParamsI2V(BaseParams):
    init_image: object
    width: int
    height: int
    steps: int
    fps: int
    num_frames: int
    sampler: str
    scheduler: str
    seed: int | None
    guidance_scale: float | None
    motion_strength: float | None


@dataclass(frozen=True)
class WanT2V14BParams(BaseParams):
    prompt: str
    width: int
    height: int
    steps: int
    fps: int
    num_frames: int
    sampler: str
    scheduler: str
    seed: int | None
    guidance_scale: float | None


@dataclass(frozen=True)
class WanI2V14BParams(BaseParams):
    init_image: object
    width: int
    height: int
    steps: int
    fps: int
    num_frames: int
    sampler: str
    scheduler: str
    seed: int | None
    guidance_scale: float | None
    motion_strength: float | None


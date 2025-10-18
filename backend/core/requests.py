"""Typed request/response objects for the modular inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from .engine_interface import TaskType


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    percent: Optional[float] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    eta_seconds: Optional[float] = None
    message: Optional[str] = None
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultEvent:
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


InferenceEvent = Union[ProgressEvent, ResultEvent]


# ---------------------------------------------------------------------------
# Shared request utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseRequest:
    task: TaskType
    prompt: str
    negative_prompt: str = ""
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    guidance_scale: Optional[float] = None
    batch_size: int = 1
    loras: Sequence[str] = field(default_factory=tuple)
    extra_networks: Sequence[str] = field(default_factory=tuple)
    clip_skip: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Txt2ImgRequest(BaseRequest):
    width: int = 512
    height: int = 512
    steps: int = 20
    tiling: bool = False
    highres_fix: Optional[Mapping[str, Any]] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Img2ImgRequest(BaseRequest):
    init_image: Any = None
    mask: Any = None
    denoise_strength: float = 0.5
    width: int = 512
    height: int = 512
    steps: int = 20
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Txt2VidRequest(BaseRequest):
    width: int = 768
    height: int = 432
    num_frames: int = 16
    fps: int = 24
    motion_strength: Optional[float] = None
    extrapolation: Optional[str] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Img2VidRequest(BaseRequest):
    init_image: Any = None
    width: int = 768
    height: int = 432
    num_frames: int = 16
    fps: int = 24
    motion_strength: Optional[float] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


# Convenience tuple for type checkers / consumers that need to accept either
ImageRequest = Union[Txt2ImgRequest, Img2ImgRequest]
VideoRequest = Union[Txt2VidRequest, Img2VidRequest]


"""Registration helpers for default engines."""

from __future__ import annotations

from typing import Optional

from backend.core.registry import EngineRegistry, registry as global_registry

from .sd15.engine import SD15Engine
from .sdxl.engine import SDXLEngine
from .flux.engine import FluxEngine
from .video.svd.engine import SVDEngine
from .video.hunyuan.engine import HunyuanVideoEngine
from .video.wan import WanTI2V5BEngine, WanT2V14BEngine, WanI2V14BEngine


def _resolve_registry(registry: Optional[EngineRegistry]) -> EngineRegistry:
    return registry or global_registry


def register_sd15(*, registry: Optional[EngineRegistry] = None, replace: bool = False) -> None:
    resolved = _resolve_registry(registry)
    resolved.register(
        "sd15",
        SD15Engine,
        aliases=("sd1.5", "stable-diffusion-1.5", "sd15-base"),
        metadata={"family": "stable-diffusion", "version": "1.5"},
        replace=replace,
    )


def register_sdxl(*, registry: Optional[EngineRegistry] = None, replace: bool = False) -> None:
    resolved = _resolve_registry(registry)
    resolved.register(
        "sdxl",
        SDXLEngine,
        aliases=("sd_xl", "stable-diffusion-xl", "sdxl-base"),
        metadata={"family": "stable-diffusion", "version": "xl"},
        replace=replace,
    )


def register_flux(*, registry: Optional[EngineRegistry] = None, replace: bool = False) -> None:
    resolved = _resolve_registry(registry)
    resolved.register(
        "flux",
        FluxEngine,
        aliases=("flux-dev", "flux-schnell"),
        metadata={"family": "flux", "version": "1.x"},
        replace=replace,
    )


def register_svd(*, registry: Optional[EngineRegistry] = None, replace: bool = False) -> None:
    resolved = _resolve_registry(registry)
    resolved.register(
        "svd",
        SVDEngine,
        aliases=("stable-video-diffusion",),
        metadata={"family": "svd", "task": "img2vid"},
        replace=replace,
    )


def register_hunyuan_video(*, registry: Optional[EngineRegistry] = None, replace: bool = False) -> None:
    resolved = _resolve_registry(registry)
    resolved.register(
        "hunyuan_video",
        HunyuanVideoEngine,
        aliases=("hunyuan-video",),
        metadata={"family": "hunyuan", "task": "txt2vid,img2vid"},
        replace=replace,
    )


def register_wan_videos(*, registry: Optional[EngineRegistry] = None, replace: bool = False) -> None:
    resolved = _resolve_registry(registry)
    resolved.register(
        "wan_ti2v_5b",
        WanTI2V5BEngine,
        aliases=("wan-2.2-ti2v-5b", "wan_ti2v"),
        metadata={"family": "wan", "version": "2.2", "params": "5B", "tasks": "txt2vid,img2vid"},
        replace=replace,
    )
    resolved.register(
        "wan_t2v_14b",
        WanT2V14BEngine,
        aliases=("wan-2.2-t2v-14b", "wan_t2v"),
        metadata={"family": "wan", "version": "2.2", "params": "14B", "tasks": "txt2vid"},
        replace=replace,
    )
    resolved.register(
        "wan_i2v_14b",
        WanI2V14BEngine,
        aliases=("wan-2.2-i2v-14b", "wan_i2v"),
        metadata={"family": "wan", "version": "2.2", "params": "14B", "tasks": "img2vid"},
        replace=replace,
    )

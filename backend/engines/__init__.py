"""Inference engine package.

This package hosts concrete engine implementations for each supported model
family (e.g. SD15, SDXL, Flux). Engines are registered via
``register_default_engines`` to avoid side effects at import time; callers can
pass a custom registry for tests or feature-flagged activation.
"""

from __future__ import annotations

from typing import Optional

from backend.core.registry import EngineRegistry

from . import registration
from .base import DiffusionEngine

__all__ = ["DiffusionEngine", "register_default_engines"]


def register_default_engines(registry: Optional[EngineRegistry] = None, *, replace: bool = False) -> None:
    """Register the default set of engines (sd15, sdxl, flux).

    Parameters
    ----------
    registry: EngineRegistry | None
        Optional custom registry. Falls back to the global registry when None.
    replace: bool
        Whether to replace existing registrations. Useful during hot reload.
    """

    registration.register_sd15(registry=registry, replace=replace)
    registration.register_sdxl(registry=registry, replace=replace)
    registration.register_flux(registry=registry, replace=replace)
    # Video engines (stubs for now)
    registration.register_svd(registry=registry, replace=replace)
    registration.register_hunyuan_video(registry=registry, replace=replace)
    registration.register_wan_videos(registry=registry, replace=replace)

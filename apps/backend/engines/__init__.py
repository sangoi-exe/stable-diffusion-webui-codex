"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Engine package facade and default registration entry points.
Exposes `register_default_engines(...)` and lazily resolves optional/large engine classes to avoid heavy imports during startup.
Includes the Anima, FLUX.2, Lens skeleton, LTX2, Qwen Image, and Z-Image L2P engine facades; parked placeholder families such as `netflix_void`
remain importable by name but are not part of default runtime registration.

Symbols (top-level; keep in sync; no ghosts):
- `EngineLoadError` (class): Error raised when an engine fails to load required resources.
- `EngineExecutionError` (class): Error raised when an engine fails during inference execution.
- `EngineRegistrationError` (class): Error raised when default engine registration would leave an invalid partial group.
- `register_default_engines` (function): Registers the canonical engine set into the registry.
- `_ENGINE_EXPORTS` (constant): Lazy export map `{name: (module_path, attr)}` for engine class exports.
- `__getattr__` (function): Lazy import hook for engine class exports.
- `__all__` (constant): Explicit export list for the engine facade.
"""

from __future__ import annotations

# tags: backend, engines, lazy-imports

from importlib import import_module
from typing import TYPE_CHECKING

from apps.backend.core.exceptions import EngineExecutionError, EngineLoadError, EngineNotFoundError, EngineRegistrationError
from apps.backend.core.registry import EngineRegistry

if TYPE_CHECKING:
    # Keep the surface import-light for runtime. For type-checkers, expose names without importing engine modules,
    # which may rely on optional deps and/or heavy import graphs.
    from typing import Any as Chroma
    from typing import Any as Flux
    from typing import Any as Flux2Engine
    from typing import Any as Ltx2Engine
    from typing import Any as NetflixVoidEngine
    from typing import Any as Kontext
    from typing import Any as StableDiffusion
    from typing import Any as StableDiffusion2
    from typing import Any as StableDiffusion3
    from typing import Any as StableDiffusionXL
    from typing import Any as StableDiffusionXLRefiner
    from typing import Any as Wan2214BEngine
    from typing import Any as Wan225BEngine
    from typing import Any as Wan22Animate14BEngine
    from typing import Any as ZImageEngine
    from typing import Any as ZImageL2PEngine
    from typing import Any as AnimaEngine
    from typing import Any as QwenImageEngine
    from typing import Any as LensEngine


def register_default_engines(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    """Register the canonical set of engines into the provided registry."""

    registration = import_module("apps.backend.engines.registration")

    from apps.backend.core.registry import registry as _global_registry

    target = registry or _global_registry

    def _has_descriptor(key: str) -> bool:
        try:
            target.get_descriptor(key)
        except EngineNotFoundError:
            return False
        return True

    def _preflight_group(keys: tuple[str, ...]) -> None:
        present = tuple(key for key in keys if _has_descriptor(key))
        if present and len(present) != len(keys):
            missing = tuple(key for key in keys if key not in present)
            raise EngineRegistrationError(
                "Partial default engine group registration is invalid; "
                f"present={present} missing={missing}."
            )

    def _maybe_register(key: str, fn) -> None:  # type: ignore[no-untyped-def]
        if replace:
            fn(registry=target, replace=True)
            return
        if _has_descriptor(key):
            return
        fn(registry=target, replace=False)

    def _maybe_register_group(keys: tuple[str, ...], fn) -> None:  # type: ignore[no-untyped-def]
        if replace:
            fn(registry=target, replace=True)
            return
        _preflight_group(keys)
        if all(_has_descriptor(key) for key in keys):
            return
        fn(registry=target, replace=False)

    if not replace:
        _preflight_group(("sdxl", "sdxl_refiner"))

    _maybe_register("sd15", registration.register_sd15)
    _maybe_register_group(("sdxl", "sdxl_refiner"), registration.register_sdxl)
    _maybe_register("flux1", registration.register_flux)
    _maybe_register("flux2", registration.register_flux2)
    _maybe_register("ltx2", registration.register_ltx2)
    _maybe_register("flux1_kontext", registration.register_kontext)
    _maybe_register("flux1_chroma", registration.register_chroma)
    _maybe_register("sd20", registration.register_sd20)
    _maybe_register("zimage", registration.register_zimage)
    _maybe_register("zimage_l2p", registration.register_zimage_l2p)
    _maybe_register("anima", registration.register_anima)
    _maybe_register("qwen_image", registration.register_qwen_image)
    # WAN22 GGUF lanes are explicit and variant-specific.
    _maybe_register("wan22_5b", registration.register_wan22_5b)
    _maybe_register("wan22_14b", registration.register_wan22_14b)
    _maybe_register("wan22_14b_animate", registration.register_wan22_14b_animate)


__all__ = [
    "EngineLoadError",
    "EngineExecutionError",
    "EngineRegistrationError",
    "register_default_engines",
    "StableDiffusion",
    "StableDiffusion2",
    "StableDiffusion3",
    "StableDiffusionXL",
    "StableDiffusionXLRefiner",
    "Flux",
    "Flux2Engine",
    "Ltx2Engine",
    "NetflixVoidEngine",
    "Kontext",
    "Chroma",
    "ZImageEngine",
    "ZImageL2PEngine",
    "AnimaEngine",
    "QwenImageEngine",
    "LensEngine",
    "Wan22Animate14BEngine",
    "Wan2214BEngine",
    "Wan225BEngine",
]

_ENGINE_EXPORTS = {
    "StableDiffusion": ("apps.backend.engines.sd.sd15", "StableDiffusion"),
    "StableDiffusion2": ("apps.backend.engines.sd.sd20", "StableDiffusion2"),
    "StableDiffusion3": ("apps.backend.engines.sd.sd35", "StableDiffusion3"),
    "StableDiffusionXL": ("apps.backend.engines.sd.sdxl", "StableDiffusionXL"),
    "StableDiffusionXLRefiner": ("apps.backend.engines.sd.sdxl", "StableDiffusionXLRefiner"),
    "Flux": ("apps.backend.engines.flux.flux", "Flux"),
    "Flux2Engine": ("apps.backend.engines.flux2.flux2", "Flux2Engine"),
    "Ltx2Engine": ("apps.backend.engines.ltx2.ltx2", "Ltx2Engine"),
    "NetflixVoidEngine": ("apps.backend.engines.netflix_void.netflix_void", "NetflixVoidEngine"),
    "Kontext": ("apps.backend.engines.flux.kontext", "Kontext"),
    "Chroma": ("apps.backend.engines.flux.chroma", "Chroma"),
    "ZImageEngine": ("apps.backend.engines.zimage.zimage", "ZImageEngine"),
    "ZImageL2PEngine": ("apps.backend.engines.zimage_l2p.zimage_l2p", "ZImageL2PEngine"),
    "AnimaEngine": ("apps.backend.engines.anima.anima", "AnimaEngine"),
    "QwenImageEngine": ("apps.backend.engines.qwen_image.qwen_image", "QwenImageEngine"),
    "LensEngine": ("apps.backend.engines.lens.lens", "LensEngine"),
    "Wan22Animate14BEngine": ("apps.backend.engines.wan22.wan22_14b_animate", "Wan22Animate14BEngine"),
    "Wan2214BEngine": ("apps.backend.engines.wan22.wan22_14b", "Wan2214BEngine"),
    "Wan225BEngine": ("apps.backend.engines.wan22.wan22_5b", "Wan225BEngine"),
}

def __getattr__(name: str):  # pragma: no cover - runtime dispatch
    if name in _ENGINE_EXPORTS:
        module_name, attr = _ENGINE_EXPORTS[name]
        mod = import_module(module_name)
        value = getattr(mod, attr)
        globals()[name] = value
        return value

    raise AttributeError(name)

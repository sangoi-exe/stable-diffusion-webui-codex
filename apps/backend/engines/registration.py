"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical engine registration functions for the backend.
Defines one `register_<engine>(...)` helper per engine family and wires aliases into the shared `EngineRegistry`.

Symbols (top-level; keep in sync; no ghosts):
- `register_sd15` (function): Registers the SD 1.5 engine and aliases.
- `register_sd20` (function): Registers the SD 2.x engine and aliases.
- `register_sdxl` (function): Registers SDXL base/refiner engines and aliases.
- `register_sd35` (function): Raises `NotImplementedError` for the parked SD 3.5 placeholder.
- `register_flux` (function): Registers the Flux engine.
- `register_flux2` (function): Registers the FLUX.2 Klein engine.
- `register_ltx2` (function): Registers the native backend-only LTX2 video engine.
- `register_netflix_void` (function): Raises `NotImplementedError` for the parked Netflix VOID placeholder.
- `register_svd` (function): Raises `NotImplementedError` for the parked SVD placeholder.
- `register_kontext` (function): Registers the Flux Kontext engine.
- `register_chroma` (function): Registers the Chroma engine.
- `register_hunyuan_video` (function): Raises `NotImplementedError` for the parked Hunyuan Video placeholder.
- `register_wan22_5b` (function): Registers WAN22 GGUF 5B engine and aliases.
- `register_wan22_14b` (function): Registers WAN22 14B engine and aliases.
- `register_wan22_14b_animate` (function): Registers WAN22 Animate 14B engine and aliases.
- `register_wan22_videos` (function): Registers all WAN22 default video engines.
- `register_zimage` (function): Registers the Z-Image engine and aliases.
- `register_zimage_l2p` (function): Registers the Z-Image L2P pixel engine with no aliases.
- `register_anima` (function): Registers the Anima engine.
- `register_qwen_image` (function): Registers the Qwen Image architecture-family engine facade.
- `register_lens` (function): Registers the parked Lens skeleton facade for isolated/manual validation only.
"""

from __future__ import annotations

from typing import Optional

from apps.backend.core.registry import EngineRegistry, register_engine


def _reg(key: str, cls, *, registry: Optional[EngineRegistry], replace: bool, aliases: tuple[str, ...] = ()) -> None:
    if registry is None:
        register_engine(key, cls, aliases=aliases, replace=replace)
    else:
        registry.register(key, cls, aliases=aliases, replace=replace)


def register_sd15(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.sd.sd15 import StableDiffusion
    _reg("sd15", StableDiffusion, registry=registry, replace=replace, aliases=("sd-1.5",))


def register_sd20(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.sd.sd20 import StableDiffusion2
    _reg("sd20", StableDiffusion2, registry=registry, replace=replace, aliases=("sd-2.0", "sd-2.1"))


def register_sdxl(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.sd.sdxl import StableDiffusionXL, StableDiffusionXLRefiner
    _reg("sdxl", StableDiffusionXL, registry=registry, replace=replace, aliases=("sd-xl",))
    _reg("sdxl_refiner", StableDiffusionXLRefiner, registry=registry, replace=replace, aliases=("sd-xl-refiner",))


def register_sd35(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    del registry, replace
    raise NotImplementedError("sd35 not yet implemented")


def register_flux(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.flux.flux import Flux
    _reg("flux1", Flux, registry=registry, replace=replace, aliases=())


def register_flux2(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.flux2.flux2 import Flux2Engine
    _reg("flux2", Flux2Engine, registry=registry, replace=replace, aliases=())


def register_ltx2(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.ltx2.ltx2 import Ltx2Engine
    _reg("ltx2", Ltx2Engine, registry=registry, replace=replace, aliases=())


def register_netflix_void(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    del registry, replace
    raise NotImplementedError("netflix_void not yet implemented")


def register_kontext(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.flux.kontext import Kontext
    _reg(
        "flux1_kontext",
        Kontext,
        registry=registry,
        replace=replace,
        aliases=(),
    )


def register_chroma(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.flux.chroma import Chroma
    _reg("flux1_chroma", Chroma, registry=registry, replace=replace, aliases=())


def register_svd(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:  # optional
    del registry, replace
    raise NotImplementedError("svd not yet implemented")


def register_hunyuan_video(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:  # optional
    del registry, replace
    raise NotImplementedError("hunyuan_video not yet implemented")


def register_wan22_5b(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.wan22.wan22_5b import Wan225BEngine
    _reg("wan22_5b", Wan225BEngine, registry=registry, replace=replace, aliases=("wan22-5b",))


def register_wan22_14b(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.wan22.wan22_14b import Wan2214BEngine
    _reg("wan22_14b", Wan2214BEngine, registry=registry, replace=replace, aliases=("wan22-14b",))


def register_wan22_14b_animate(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.wan22.wan22_14b_animate import Wan22Animate14BEngine
    _reg(
        "wan22_14b_animate",
        Wan22Animate14BEngine,
        registry=registry,
        replace=replace,
        aliases=("wan22-14b-animate", "wan-animate"),
    )


def register_wan22_videos(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    register_wan22_5b(registry=registry, replace=replace)
    register_wan22_14b(registry=registry, replace=replace)
    register_wan22_14b_animate(registry=registry, replace=replace)


def register_zimage(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.zimage.zimage import ZImageEngine
    _reg("zimage", ZImageEngine, registry=registry, replace=replace, aliases=("z-image", "z-image-turbo"))


def register_zimage_l2p(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.zimage_l2p.zimage_l2p import ZImageL2PEngine
    _reg("zimage_l2p", ZImageL2PEngine, registry=registry, replace=replace, aliases=())


def register_anima(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.anima.anima import AnimaEngine
    _reg("anima", AnimaEngine, registry=registry, replace=replace, aliases=())


def register_qwen_image(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.qwen_image.qwen_image import QwenImageEngine
    _reg("qwen_image", QwenImageEngine, registry=registry, replace=replace, aliases=())


def register_lens(*, registry: EngineRegistry | None = None, replace: bool = False) -> None:
    from apps.backend.engines.lens.lens import LensEngine
    _reg("lens", LensEngine, registry=registry, replace=replace, aliases=())

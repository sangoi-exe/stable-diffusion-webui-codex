"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Semantic engine capability surfaces, exact-engine inpaint-mode truth, explicit capability-only exact ids, and parked exact-engine stubs exposed to the UI layer.
Defines `SemanticEngine` tags and an `EngineParamSurface` describing which high-level UI sections and tasks are expected to be used for each engine,
including explicit masked-img2img/inpaint support, vid2vid discoverability, and native IP-Adapter/SUPIR discoverability, with executable defaults and
recommendation hints for the live surface (for example SD15 `ddim`/`ddim`, WAN22 `uni-pc bh2`/`simple`, and LTX2 `euler`/`simple` with no sampler fiction
beyond the live runtime lane).
Includes Qwen Image (`SemanticEngine.QWEN_IMAGE`) as an Edit-2511-only Qwen2.5-VL-conditioned single-image img2img engine,
and Anima (`SemanticEngine.ANIMA`) as a flow-based image engine (txt2img/img2img) requiring sha-selected external assets and exposing
`er sde` in the recommended sampler surface. Z-Image L2P is exposed as a separate no-VAE pixel-space txt2img-only exact engine.
Microsoft Lens is exposed only as a parked exact engine stub until its GPT-OSS/LensTransformer/VAE runtime lands; it is not a semantic engine or runnable capability row.
FLUX.2 exposes the truthful Klein 4B/base-4B slice here: txt2img plus dedicated
image-conditioned img2img with hires enabled only after the real backend continuation path landed; LoRA remains off.
WAN semantic capabilities are bound to explicit WAN22 variant families via primary-family mapping.

Symbols (top-level; keep in sync; no ghosts):
- `SemanticEngine` (enum): UI-facing semantic engine tags used by API/frontend gating.
- `GuidanceAdvancedSurface` (dataclass): Optional per-engine support map for advanced CFG/APG controls (`extras.guidance` keys).
- `EngineParamSurface` (dataclass): Declared parameter surface for an engine (workflow flags including masked img2img/inpaint, vid2vid, IP-Adapter/SUPIR support, and optional sampler/scheduler recommendations).
- `ParkedExactEngineStub` (dataclass): Public placeholder contract for exact engine ids that are intentionally parked/not implemented.
- `ENGINE_SURFACES` (constant): Mapping of semantic engine tag to `EngineParamSurface`.
- `ENGINE_ID_TO_SEMANTIC_ENGINE` (constant): Canonical mapping from API engine ids to semantic engine tags.
- `PARKED_EXACT_ENGINES` (constant): Mapping of exact engine ids that remain public only as parked placeholders.
- `CAPABILITY_ONLY_EXACT_ENGINES` (constant): Exact ids exposed only for capability/taxonomy accounting, never runtime execution.
- `EXACT_ENGINE_INPAINT_MODES` (constant): Mapping of exact engine ids to supported public img2img inpaint modes.
- `ip_adapter_support_error` (function): Return the fail-loud exact-engine/semantic-engine support error for IP-Adapter, or `None` when supported.
- `supports_ip_adapter_engine_id` (function): Return whether the exact engine id is allowed to run IP-Adapter in tranche 1.
- `supir_support_error` (function): Return the fail-loud exact-engine/semantic-engine support error for SUPIR mode, or `None` when supported.
- `inpaint_modes_for_engine_id` (function): Return the exact-engine-owned public img2img inpaint modes.
- `inpaint_mode_support_error` (function): Return the fail-loud exact-engine support error for one public img2img inpaint mode, or `None` when supported.
- `build_ltx2_capability_surface` (function): Build the truthful semantic capability surface for the live LTX2 lane.
- `list_engine_capabilities` (function): Returns engine surfaces keyed by string tag for API responses.
- `semantic_engine_for_engine_id` (function): Resolve a semantic engine tag from an API engine id (fail-loud on unknown ids).
- `primary_family_for_engine_id` (function): Resolve the exact primary `ModelFamily` authority for a runtime engine id (fail-loud on unknown ids).
- `engine_supports_cfg` (function): Return whether the engine family supports classic CFG (`cfg`) via family capabilities.
- `serialize_engine_capabilities` (function): Returns engine capability surfaces as JSON-serializable dicts.
- `serialize_exact_engine_inpaint_modes` (function): Returns exact-engine img2img inpaint modes as JSON-serializable dicts.
- `serialize_family_capabilities` (function): Returns model family capability surfaces as JSON-serializable dicts.
- `serialize_parked_exact_engines` (function): Returns parked exact-engine stubs as JSON-serializable dicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Mapping

from apps.backend.runtime.model_registry.ltx2_execution import (
    LTX2_EXECUTION_SURFACE_KEY,
    Ltx2ExecutionSurface,
    build_ltx2_execution_surface,
)
from apps.backend.runtime.model_registry.specs import ModelFamily


class SemanticEngine(str, Enum):
    """Semantic engine tags exposed to the UI layer.

    These align with `_detect_semantic_engine()` in `run_api` and describe
    high-level workflow families rather than individual checkpoints.
    """

    SD15 = "sd15"
    SDXL = "sdxl"
    FLUX = "flux1"
    FLUX2 = "flux2"
    QWEN_IMAGE = "qwen_image"
    ZIMAGE = "zimage"
    ZIMAGE_L2P = "zimage_l2p"
    ANIMA = "anima"
    CHROMA = "chroma"
    WAN22 = "wan22"
    LTX2 = "ltx2"
    NETFLIX_VOID = "netflix_void"
    HUNYUAN_VIDEO = "hunyuan_video"
    SVD = "svd"


@dataclass(frozen=True)
class GuidanceAdvancedSurface:
    """Per-engine support map for advanced guidance controls exposed in `extras.guidance`."""

    apg_enabled: bool = False
    apg_start_step: bool = False
    apg_eta: bool = False
    apg_momentum: bool = False
    apg_norm_threshold: bool = False
    apg_rescale: bool = False
    guidance_rescale: bool = False
    cfg_trunc_ratio: bool = False
    renorm_cfg: bool = False


@dataclass(frozen=True)
class EngineParamSurface:
    """Declared parameter surface for a semantic engine.

    This describes which high-level workflows and feature sections are intended
    to be used from the Codex UI. It is deliberately narrower than the full
    backend capabilities so the frontend can hide params that have no effect
    for a given engine.
    """

    supports_txt2img: bool
    supports_img2img: bool
    supports_img2img_masking: bool
    supports_txt2vid: bool
    supports_img2vid: bool
    supports_hires: bool
    supports_refiner: bool
    supports_lora: bool
    supports_controlnet: bool
    supports_ip_adapter: bool
    supports_supir_mode: bool = False
    supports_vid2vid: bool = False
    # Optional: recommended sampler/scheduler lists for UI hinting.
    recommended_samplers: tuple[str, ...] | None = None
    recommended_schedulers: tuple[str, ...] | None = None
    # Optional: UI defaults for sampler/scheduler selection.
    default_sampler: str | None = None
    default_scheduler: str | None = None
    # Optional: support map for advanced guidance controls (`extras.guidance` keys).
    guidance_advanced: GuidanceAdvancedSurface | None = None
    # Optional: nested LTX-only execution-profile/default surface.
    ltx_execution_surface: Ltx2ExecutionSurface | None = None


@dataclass(frozen=True)
class ParkedExactEngineStub:
    """Public placeholder contract for an exact engine id that is intentionally parked."""

    status: str
    detail: str


def build_ltx2_capability_surface() -> EngineParamSurface:
    """Build the truthful semantic capability surface for the live LTX2 lane."""

    return EngineParamSurface(
        supports_txt2img=False,
        supports_img2img=False,
        supports_img2img_masking=False,
        supports_txt2vid=True,
        supports_img2vid=True,
        supports_hires=False,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler",),
        recommended_schedulers=("simple",),
        default_sampler="euler",
        default_scheduler="simple",
        ltx_execution_surface=build_ltx2_execution_surface(),
    )


_GUIDANCE_ADVANCED_CLASSIC_CFG = GuidanceAdvancedSurface(
    apg_enabled=True,
    apg_start_step=True,
    apg_eta=True,
    apg_momentum=True,
    apg_norm_threshold=True,
    apg_rescale=True,
    guidance_rescale=True,
    cfg_trunc_ratio=True,
    renorm_cfg=True,
)


ENGINE_SURFACES: Dict[SemanticEngine, EngineParamSurface] = {
    # Classic SD1.x-style image generation.
    SemanticEngine.SD15: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=True,
        supports_controlnet=False,
        supports_ip_adapter=True,
        supports_supir_mode=False,
        default_sampler="ddim",
        default_scheduler="ddim",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # SDXL image workflows (base + hires + refiner).
    SemanticEngine.SDXL: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=True,
        supports_lora=True,
        supports_controlnet=False,
        supports_ip_adapter=True,
        supports_supir_mode=True,
        default_sampler="euler",
        default_scheduler="euler_discrete",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # Flux.1 (flow-based image diffusion).
    SemanticEngine.FLUX: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=False,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=True,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler", "euler a", "dpm++ 2m"),
        recommended_schedulers=("simple", "beta", "normal"),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # FLUX.2 Klein (single-Qwen; txt2img + image-conditioned img2img; hires enabled after dedicated continuation wiring).
    SemanticEngine.FLUX2: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler", "dpm++ 2m"),
        recommended_schedulers=("simple",),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # Qwen Image Edit-2511: single-image edit img2img only.
    SemanticEngine.QWEN_IMAGE: EngineParamSurface(
        supports_txt2img=False,
        supports_img2img=True,
        supports_img2img_masking=False,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=False,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler",),
        recommended_schedulers=("simple",),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # Z-Image (Turbo/Base variants; flow-based; tuned for simple predictor schedule).
    SemanticEngine.ZIMAGE: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=True,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=True,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler", "dpm++ 2m"),
        recommended_schedulers=("simple",),
        default_sampler="euler",
        default_scheduler="simple",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # Z-Image L2P (pixel-space no-VAE 1K checkpoint).
    SemanticEngine.ZIMAGE_L2P: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=False,
        supports_img2img_masking=False,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=False,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler",),
        recommended_schedulers=("simple",),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # Anima (Cosmos Predict2; flow-based; Qwen3-0.6B conditioning; classic CFG).
    SemanticEngine.ANIMA: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=False,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler", "euler a", "dpm++ 2m", "er sde"),
        recommended_schedulers=("simple", "beta", "normal", "exponential"),
        default_sampler="euler",
        default_scheduler="simple",
        guidance_advanced=_GUIDANCE_ADVANCED_CLASSIC_CFG,
    ),
    # Chroma (flow-based image generation).
    SemanticEngine.CHROMA: EngineParamSurface(
        supports_txt2img=True,
        supports_img2img=True,
        supports_img2img_masking=False,
        supports_txt2vid=False,
        supports_img2vid=False,
        supports_hires=True,
        supports_refiner=False,
        supports_lora=False,
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("euler", "dpm++ 2m"),
        recommended_schedulers=("simple", "beta", "normal"),
        default_sampler="euler",
        default_scheduler="simple",
    ),
    # Wan 2.2 dual-stage video (txt2vid/img2vid).
    SemanticEngine.WAN22: EngineParamSurface(
        supports_txt2img=False,
        supports_img2img=False,
        supports_img2img_masking=False,
        supports_txt2vid=True,
        supports_img2vid=True,
        supports_hires=False,
        supports_refiner=False,
        supports_lora=True,  # high/low LoRA slots in WAN22 panel
        supports_controlnet=False,
        supports_ip_adapter=False,
        supports_supir_mode=False,
        recommended_samplers=("uni-pc bh2", "uni-pc", "euler", "euler a"),
        recommended_schedulers=("simple",),
        default_sampler="uni-pc bh2",
        default_scheduler="simple",
    ),
    # LTX2 distilled/core-only video workflows (txt2vid/img2vid).
    SemanticEngine.LTX2: build_ltx2_capability_surface(),
}

ENGINE_ID_TO_SEMANTIC_ENGINE: Dict[str, SemanticEngine] = {
    "sd15": SemanticEngine.SD15,
    "sd20": SemanticEngine.SD15,
    "sdxl": SemanticEngine.SDXL,
    "sdxl_refiner": SemanticEngine.SDXL,
    "flux1": SemanticEngine.FLUX,
    "flux1_kontext": SemanticEngine.FLUX,
    "flux1_fill": SemanticEngine.FLUX,
    "flux2": SemanticEngine.FLUX2,
    "qwen_image": SemanticEngine.QWEN_IMAGE,
    "flux1_chroma": SemanticEngine.CHROMA,
    "zimage": SemanticEngine.ZIMAGE,
    "zimage_l2p": SemanticEngine.ZIMAGE_L2P,
    "anima": SemanticEngine.ANIMA,
    "wan22_5b": SemanticEngine.WAN22,
    "wan22_14b": SemanticEngine.WAN22,
    "wan22_14b_animate": SemanticEngine.WAN22,
    "ltx2": SemanticEngine.LTX2,
}

PARKED_EXACT_ENGINES: Dict[str, ParkedExactEngineStub] = {
    "sd35": ParkedExactEngineStub(
        status="not_implemented",
        detail="Engine 'sd35' is parked until SD3.5 conditioning/keymap support is implemented.",
    ),
    "netflix_void": ParkedExactEngineStub(
        status="not_implemented",
        detail="Engine 'netflix_void' is parked until the native vid2vid runtime is implemented.",
    ),
    "svd": ParkedExactEngineStub(
        status="not_implemented",
        detail="Engine 'svd' is parked; the Stable Video Diffusion runtime is not implemented yet.",
    ),
    "hunyuan_video": ParkedExactEngineStub(
        status="not_implemented",
        detail="Engine 'hunyuan_video' is parked; the Hunyuan Video runtime is not implemented yet.",
    ),
    "lens": ParkedExactEngineStub(
        status="not_implemented",
        detail="Engine 'lens' is parked; Lens txt2img runtime not yet implemented.",
    ),
}

CAPABILITY_ONLY_EXACT_ENGINES: frozenset[str] = frozenset({"flux1_fill"})

_GENERIC_INPAINT_MODES: tuple[str, ...] = ("per_step_blend", "post_sample_blend")

EXACT_ENGINE_INPAINT_MODES: Dict[str, tuple[str, ...]] = {
    "sd15": _GENERIC_INPAINT_MODES,
    "sd20": _GENERIC_INPAINT_MODES,
    "sdxl": (*_GENERIC_INPAINT_MODES, "fooocus_inpaint", "brushnet"),
    "sdxl_refiner": (),
    "flux1": (),
    "flux1_kontext": (),
    "flux1_fill": (),
    "flux2": _GENERIC_INPAINT_MODES,
    "qwen_image": (),
    "flux1_chroma": (),
    "zimage": _GENERIC_INPAINT_MODES,
    "zimage_l2p": (),
    "anima": (),
    "wan22_5b": (),
    "wan22_14b": (),
    "wan22_14b_animate": (),
    "ltx2": (),
    "sd35": (),
    "netflix_void": (),
    "svd": (),
    "hunyuan_video": (),
}

_IP_ADAPTER_EXACT_ENGINE_REJECTS: Dict[str, str] = {
    "sd20": "Engine 'sd20' is unsupported for IP-Adapter in tranche 1.",
    "sd35": "Engine 'sd35' is unsupported for IP-Adapter in tranche 1.",
    "sdxl_refiner": "Engine 'sdxl_refiner' is unsupported for IP-Adapter in tranche 1.",
}

_SUPIR_EXACT_ENGINE_REJECTS: Dict[str, str] = {
    "sd15": "Engine 'sd15' is unsupported for SUPIR mode in tranche 1.",
    "sd20": "Engine 'sd20' is unsupported for SUPIR mode in tranche 1.",
    "sd35": "Engine 'sd35' is unsupported for SUPIR mode in tranche 1.",
    "sdxl_refiner": "Engine 'sdxl_refiner' is unsupported for SUPIR mode in tranche 1.",
}

_ENGINE_ID_PRIMARY_FAMILY: Dict[str, ModelFamily] = {
    "sd15": ModelFamily.SD15,
    "sd20": ModelFamily.SD20,
    "sdxl": ModelFamily.SDXL,
    "sdxl_refiner": ModelFamily.SDXL_REFINER,
    "sd35": ModelFamily.SD35,
    "flux1": ModelFamily.FLUX,
    "flux1_kontext": ModelFamily.FLUX_KONTEXT,
    "flux2": ModelFamily.FLUX2,
    "qwen_image": ModelFamily.QWEN_IMAGE,
    "flux1_chroma": ModelFamily.CHROMA,
    "zimage": ModelFamily.ZIMAGE,
    "zimage_l2p": ModelFamily.ZIMAGE_L2P,
    "anima": ModelFamily.ANIMA,
    "wan22_5b": ModelFamily.WAN22_5B,
    "wan22_14b": ModelFamily.WAN22_14B,
    "wan22_14b_animate": ModelFamily.WAN22_ANIMATE,
    "ltx2": ModelFamily.LTX2,
    "netflix_void": ModelFamily.NETFLIX_VOID,
    "hunyuan_video": ModelFamily.HUNYUAN,
    "svd": ModelFamily.SVD,
}


def list_engine_capabilities() -> Mapping[str, EngineParamSurface]:
    """Return engine capability surfaces keyed by semantic engine tag."""
    return {engine.value: surface for engine, surface in ENGINE_SURFACES.items()}


def semantic_engine_for_engine_id(engine_id: str) -> SemanticEngine:
    normalized = str(engine_id or "").strip()
    if normalized == "":
        raise KeyError("Engine id is empty.")
    if normalized not in ENGINE_ID_TO_SEMANTIC_ENGINE:
        raise KeyError(f"Unknown engine id for semantic mapping: {normalized!r}")
    return ENGINE_ID_TO_SEMANTIC_ENGINE[normalized]


def primary_family_for_engine_id(engine_id: str) -> ModelFamily:
    normalized = str(engine_id or "").strip()
    if normalized == "":
        raise KeyError("Engine id is empty.")
    family = _ENGINE_ID_PRIMARY_FAMILY.get(normalized)
    if family is None:
        raise KeyError(f"No primary family mapping for engine id {normalized!r}.")
    return family


def ip_adapter_support_error(engine_id: str) -> str | None:
    normalized = str(engine_id or "").strip().lower()
    if normalized == "":
        return "IP-Adapter requires a non-empty engine id."
    exact_reject = _IP_ADAPTER_EXACT_ENGINE_REJECTS.get(normalized)
    if exact_reject is not None:
        return exact_reject
    try:
        semantic_engine = semantic_engine_for_engine_id(normalized)
    except KeyError:
        return f"Engine '{normalized}' is unsupported for IP-Adapter in tranche 1."
    if semantic_engine not in {SemanticEngine.SD15, SemanticEngine.SDXL}:
        return (
            f"Engine '{normalized}' is unsupported for IP-Adapter in tranche 1. "
            "Supported semantic engines: sd15, sdxl."
        )
    return None


def supports_ip_adapter_engine_id(engine_id: str) -> bool:
    return ip_adapter_support_error(engine_id) is None


def supir_support_error(engine_id: str) -> str | None:
    normalized = str(engine_id or "").strip().lower()
    if normalized == "":
        return "SUPIR mode requires a non-empty engine id."
    exact_reject = _SUPIR_EXACT_ENGINE_REJECTS.get(normalized)
    if exact_reject is not None:
        return exact_reject
    if normalized == "sdxl":
        return None
    try:
        semantic_engine = semantic_engine_for_engine_id(normalized)
    except KeyError:
        return f"Engine '{normalized}' is unsupported for SUPIR mode in tranche 1."
    if semantic_engine is SemanticEngine.SDXL:
        return f"Engine '{normalized}' is unsupported for SUPIR mode in tranche 1."
    return (
        f"Engine '{normalized}' is unsupported for SUPIR mode in tranche 1. "
        "Supported semantic engine: sdxl (exact engine id 'sdxl' only)."
    )


def inpaint_modes_for_engine_id(engine_id: str) -> tuple[str, ...]:
    normalized = str(engine_id or "").strip().lower()
    if normalized == "":
        raise KeyError("Engine id is empty.")
    if normalized in EXACT_ENGINE_INPAINT_MODES:
        return EXACT_ENGINE_INPAINT_MODES[normalized]
    if normalized in ENGINE_ID_TO_SEMANTIC_ENGINE or normalized in PARKED_EXACT_ENGINES:
        return ()
    raise KeyError(f"Unknown engine id for inpaint mode mapping: {normalized!r}")


def inpaint_mode_support_error(engine_id: str, mode: str) -> str | None:
    normalized_engine = str(engine_id or "").strip().lower()
    normalized_mode = str(mode or "").strip()
    if normalized_engine == "":
        return "Img2img inpaint mode requires a non-empty engine id."
    if normalized_mode == "":
        return "Img2img inpaint mode requires a non-empty mode value."
    try:
        supported_modes = inpaint_modes_for_engine_id(normalized_engine)
    except KeyError:
        return f"Engine '{normalized_engine}' is unsupported for img2img inpaint mode '{normalized_mode}'."
    if normalized_mode in supported_modes:
        return None
    supported_label = ", ".join(supported_modes) if supported_modes else "none"
    return (
        f"Engine '{normalized_engine}' does not support img2img inpaint mode '{normalized_mode}'. "
        f"Supported modes: {supported_label}."
    )

def engine_supports_cfg(engine_id: str) -> bool:
    from apps.backend.runtime.model_registry.family_runtime import get_family_spec

    spec = get_family_spec(primary_family_for_engine_id(engine_id))
    return bool(spec.capabilities.supports_cfg)


def serialize_engine_capabilities() -> Dict[str, Dict[str, object]]:
    """Return capabilities as plain dicts for JSON responses."""
    result: Dict[str, Dict[str, object]] = {}
    for engine, surface in list_engine_capabilities().items():
        payload = asdict(surface)
        if payload.get(LTX2_EXECUTION_SURFACE_KEY) is None:
            payload.pop(LTX2_EXECUTION_SURFACE_KEY, None)
        result[engine] = payload
    return result


def serialize_family_capabilities() -> Dict[str, Dict[str, object]]:
    """Return FamilyCapabilities for all model families as JSON-serializable dicts.

    Returns:
        Dict mapping family name (e.g. "FLUX", "SDXL") to capability dict.
    """
    from apps.backend.runtime.model_registry.family_runtime import FAMILY_RUNTIME_SPECS

    result = {}
    for family, spec in FAMILY_RUNTIME_SPECS.items():
        result[family.value] = spec.capabilities.to_dict()
    return result


def serialize_parked_exact_engines() -> Dict[str, Dict[str, str]]:
    """Return parked exact-engine stubs as JSON-serializable dicts."""

    return {engine_id: asdict(stub) for engine_id, stub in PARKED_EXACT_ENGINES.items()}


def serialize_exact_engine_inpaint_modes() -> Dict[str, list[str]]:
    return {engine_id: list(modes) for engine_id, modes in EXACT_ENGINE_INPAINT_MODES.items()}


__all__ = [
    "SemanticEngine",
    "GuidanceAdvancedSurface",
    "EngineParamSurface",
    "ParkedExactEngineStub",
    "ENGINE_SURFACES",
    "ENGINE_ID_TO_SEMANTIC_ENGINE",
    "PARKED_EXACT_ENGINES",
    "CAPABILITY_ONLY_EXACT_ENGINES",
    "EXACT_ENGINE_INPAINT_MODES",
    "ip_adapter_support_error",
    "supports_ip_adapter_engine_id",
    "supir_support_error",
    "inpaint_modes_for_engine_id",
    "inpaint_mode_support_error",
    "build_ltx2_capability_surface",
    "list_engine_capabilities",
    "semantic_engine_for_engine_id",
    "primary_family_for_engine_id",
    "engine_supports_cfg",
    "serialize_engine_capabilities",
    "serialize_family_capabilities",
    "serialize_parked_exact_engines",
    "serialize_exact_engine_inpaint_modes",
]

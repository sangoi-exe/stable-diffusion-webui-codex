"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model-family → parser-plan dispatch for the Codex model parser.
Maps `ModelFamily` values to the appropriate `build_plan(...)` implementation and provides a single `resolve_plan(...)` entrypoint used by
`parse_state_dict(...)`.
Includes core-only families such as Anima (Cosmos Predict2), FLUX.2 Klein 4B, and Qwen Image Edit-2511 that rely on sha-selected external assets.
Includes LTX2 monolithic-checkpoint planning for strict component seam extraction (`transformer`, `connectors`, `vae`, `audio_vae`, `vocoder`).
WAN22 family variants are dispatched explicitly (`WAN22_5B`/`WAN22_14B`/`WAN22_ANIMATE`).
Z-Image L2P is dispatched separately from latent Z-Image because it is a no-VAE pixel-space checkpoint.

Symbols (top-level; keep in sync; no ghosts):
- `_BUILDERS` (constant): Mapping of supported `ModelFamily` values to plan-builder callables.
- `resolve_plan` (function): Resolves a `ParserPlanBundle` for a signature (raises `UnsupportedFamilyError` when missing).
"""

from __future__ import annotations

from typing import Callable, Dict

from apps.backend.runtime.model_registry.specs import ModelFamily, ModelSignature

from ..errors import UnsupportedFamilyError
from ..specs import ParserPlanBundle
from . import anima, chroma, flux, flux2, ltx2, qwen_image, sd1, sd2, sd3, sdxl, wan22, zimage, zimage_l2p


_BUILDERS: Dict[ModelFamily, Callable[[ModelSignature], ParserPlanBundle]] = {
    ModelFamily.SD15: sd1.build_plan,
    ModelFamily.SD20: sd2.build_plan,
    ModelFamily.SDXL: sdxl.build_plan,
    ModelFamily.SDXL_REFINER: sdxl.build_plan,
    ModelFamily.SD3: sd3.build_plan,
    ModelFamily.SD35: sd3.build_plan,
    ModelFamily.FLUX: flux.build_plan,
    ModelFamily.FLUX_KONTEXT: flux.build_plan,
    ModelFamily.FLUX2: flux2.build_plan,
    ModelFamily.LTX2: ltx2.build_plan,
    ModelFamily.CHROMA: chroma.build_plan,
    ModelFamily.ANIMA: anima.build_plan,
    ModelFamily.QWEN_IMAGE: qwen_image.build_plan,
    ModelFamily.WAN22_5B: wan22.build_plan,
    ModelFamily.WAN22_14B: wan22.build_plan,
    ModelFamily.WAN22_ANIMATE: wan22.build_plan,
    ModelFamily.ZIMAGE: zimage.build_plan,
    ModelFamily.ZIMAGE_L2P: zimage_l2p.build_plan,
}


def resolve_plan(signature: ModelSignature) -> ParserPlanBundle:
    builder = _BUILDERS.get(signature.family)
    if builder is None:
        raise UnsupportedFamilyError(signature.family.value)
    return builder(signature)

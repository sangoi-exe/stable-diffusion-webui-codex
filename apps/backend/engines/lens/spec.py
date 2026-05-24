"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Microsoft Lens parked-engine load specification.
Binds the strict `lens_variant` load option to the JSON/index-only folder validator without introducing default engine loading or runtime tensor materialization.

Symbols (top-level; keep in sync; no ghosts):
- `LensEngineSpec` (dataclass): Resolved Lens skeleton load contract.
- `lens_engine_spec_from_options` (function): Build and validate a Lens engine spec from model_ref plus engine_options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apps.backend.engines.lens.factory import LensFolderMetadata, LensFolderValidationMode, validate_lens_folder
from apps.backend.runtime.families.lens.config import LENS_VARIANT_KEY, require_lens_variant


@dataclass(frozen=True, slots=True)
class LensEngineSpec:
    model_ref: str
    variant: str
    validation_mode: LensFolderValidationMode
    metadata: LensFolderMetadata


def lens_engine_spec_from_options(
    model_ref: object,
    *,
    options: Mapping[str, Any],
    validation_mode: LensFolderValidationMode = "metadata_only",
) -> LensEngineSpec:
    if not isinstance(model_ref, str) or not model_ref.strip():
        raise RuntimeError("Lens model_ref must be a non-empty Diffusers folder path.")
    variant = require_lens_variant(
        options.get(LENS_VARIANT_KEY),
        context=f"Lens engine option {LENS_VARIANT_KEY!r}",
    )
    metadata = validate_lens_folder(model_ref.strip(), variant=variant, validation_mode=validation_mode)
    return LensEngineSpec(
        model_ref=model_ref.strip(),
        variant=variant,
        validation_mode=validation_mode,
        metadata=metadata,
    )


__all__ = ["LensEngineSpec", "lens_engine_spec_from_options"]

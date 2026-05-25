"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Parked Microsoft Lens engine facade for skeleton/bootstrap validation.
Registers manually under `lens`, validates metadata-only Diffusers folders, records Lens bootstrap readiness metadata, exposes the reserved `sample_lens_txt2img(...)` hook, and fails loud until real Lens txt2img runtime is implemented.

Symbols (top-level; keep in sync; no ghosts):
- `LensEngine` (class): Manually-registerable parked Lens engine facade with metadata validation and not-implemented txt2img hook.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping

from apps.backend.core.engine_interface import BaseInferenceEngine, EngineCapabilities
from apps.backend.core.requests import InferenceEvent
from apps.backend.engines.lens.spec import LensEngineSpec, lens_engine_spec_from_options
from apps.backend.runtime.families.lens.bootstrap import LensRuntimeBootstrapStatus, probe_lens_runtime_bootstrap
from apps.backend.runtime.families.lens.config import (
    LENS_ENGINE_ID,
    LENS_NOT_IMPLEMENTED_MESSAGE,
    LENS_SUPPORTED_VARIANTS,
    LENS_VARIANT_KEY,
)
from apps.backend.runtime.families.lens.sampler import sample_lens_txt2img_not_implemented
from apps.backend.runtime.logging import get_backend_logger

logger = get_backend_logger("backend.engines.lens")


class LensEngine(BaseInferenceEngine):
    """Parked Lens facade registered under the canonical `lens` engine id."""

    engine_id = LENS_ENGINE_ID

    def __init__(self) -> None:
        super().__init__()
        self._spec: LensEngineSpec | None = None
        self._bootstrap_status: LensRuntimeBootstrapStatus | None = None

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(),
            model_types=(LENS_ENGINE_ID,),
            devices=("cpu", "cuda"),
            precision=("bf16", "fp16", "fp32"),
            extras={
                "status": "not_implemented",
                "detail": LENS_NOT_IMPLEMENTED_MESSAGE,
                "variants": LENS_SUPPORTED_VARIANTS,
                "validation_modes": ("metadata_only", "runtime_ready"),
            },
        )

    def load(self, model_ref: str, **options: Any) -> None:
        if self._is_loaded:
            self.unload()
        spec = lens_engine_spec_from_options(model_ref, options=options, validation_mode="metadata_only")
        bootstrap_status = probe_lens_runtime_bootstrap(spec.model_ref, variant=spec.variant)
        self._spec = spec
        self._bootstrap_status = bootstrap_status
        self.mark_loaded()
        logger.info(
            "Loaded Lens metadata/bootstrap skeleton: model_ref=%s variant=%s quant_policy=%s runtime_ready=%s indexed_size=%s",
            spec.model_ref,
            spec.variant,
            spec.text_encoder_quant_policy,
            bootstrap_status.full_lens_runtime_ready,
            spec.metadata.total_indexed_weight_size,
        )

    def unload(self) -> None:
        self._spec = None
        self._bootstrap_status = None
        self.mark_unloaded()

    def status(self) -> Mapping[str, Any]:
        data = dict(super().status())
        if self._spec is not None:
            data["model_ref"] = self._spec.model_ref
            data[LENS_VARIANT_KEY] = self._spec.variant
            data["lens_text_encoder_quant_policy"] = self._spec.text_encoder_quant_policy
            data["validation_mode"] = self._spec.validation_mode
            data["indexed_weight_size"] = self._spec.metadata.total_indexed_weight_size
        if self._bootstrap_status is not None:
            data["lens_bootstrap"] = self._bootstrap_status.to_dict()
        return data

    def sample_lens_txt2img(self, *args: Any, **kwargs: Any) -> None:
        self.ensure_loaded()
        sample_lens_txt2img_not_implemented(*args, **kwargs)

    def txt2img(self, request: Any, **kwargs: Any) -> Iterator[InferenceEvent]:
        del request, kwargs
        self.ensure_loaded()
        raise NotImplementedError(LENS_NOT_IMPLEMENTED_MESSAGE)


__all__ = ["LensEngine"]

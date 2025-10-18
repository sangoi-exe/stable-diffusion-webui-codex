"""Flux engine bridged to existing WebUI processing (txt2img)."""

from __future__ import annotations

from typing import Iterator

from backend.core.engine_interface import EngineCapabilities, TaskType
from backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2ImgRequest
from backend.core.param_registry import parse_params
from backend.core.presets import apply_preset_to_request
from backend.core.enums import Mode
from backend.core.exceptions import EngineLoadError

from ..base import DiffusionEngine
from ..util.adapters import build_txt2img_processing

# WebUI modules
from modules import processing as _processing
from modules import sd_models as _sd_models


class FluxEngine(DiffusionEngine):
    engine_id = "flux"

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG,),
            model_types=("flux-dev", "flux-schnell"),
            precision=("fp16", "bf16"),
            extras={"notes": "Bridged to modules.processing"},
        )

    def txt2img(self, request: Txt2ImgRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        mode = Mode(str((request.metadata or {}).get("mode", "Normal"))) if isinstance(request.metadata, dict) else Mode.NORMAL
        request, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.TXT2IMG, request, logger=self._logger)
        _ = parse_params(self.engine_id, TaskType.TXT2IMG, request)
        p = build_txt2img_processing(request)
        yield ProgressEvent(stage="prepare", percent=5.0, message=f"flux txt2img model={self._model_ref}")
        processed = _processing.process_images(p)
        payload = {"images": processed.images + processed.extra_images, "info": processed.js()}
        yield ResultEvent(payload=payload, metadata={"engine": self.engine_id, "task": TaskType.TXT2IMG.value, "model": self._model_ref or ""})

    # ------------------------------------------------------------------
    def load(self, model_ref: str, **options: object) -> None:  # type: ignore[override]
        checkpoint = _sd_models.get_closet_checkpoint_match(model_ref)
        if checkpoint is None:
            raise EngineLoadError(f"flux load failed: checkpoint not found for '{model_ref}'")

        additional_modules = []
        if isinstance(options.get("additional_modules"), (list, tuple)):
            additional_modules = list(options.get("additional_modules") or [])

        _sd_models.model_data.forge_loading_parameters = dict(
            checkpoint_info=checkpoint,
            additional_modules=additional_modules,
            unet_storage_dtype=options.get("unet_storage_dtype"),
        )
        self.mark_loaded()

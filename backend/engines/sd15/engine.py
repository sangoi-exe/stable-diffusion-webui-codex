"""Stub implementation for Stable Diffusion 1.5 engine."""

from __future__ import annotations

from typing import Iterator

from backend.core.engine_interface import EngineCapabilities, TaskType
from backend.core.requests import Img2ImgRequest, InferenceEvent, ProgressEvent, ResultEvent, Txt2ImgRequest
from backend.core.exceptions import EngineLoadError
from backend.core.param_registry import parse_params
from backend.core.presets import apply_preset_to_request
from backend.core.enums import Mode

from ..base import DiffusionEngine
from ..util.adapters import build_txt2img_processing, build_img2img_processing

# WebUI modules
from modules import processing as _processing
from modules import sd_models as _sd_models
from modules import shared as _shared


class SD15Engine(DiffusionEngine):
    engine_id = "sd15"

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(
                TaskType.TXT2IMG,
                TaskType.IMG2IMG,
                TaskType.INPAINT,
                TaskType.UPSCALE,
            ),
            model_types=("sd1.5", "sd-v1", "stable-diffusion"),
            precision=("fp32", "fp16"),
            extras={"notes": "Stub engine awaiting implementation"},
        )

    def txt2img(self, request: Txt2ImgRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        mode = Mode(str((request.metadata or {}).get("mode", "Normal"))) if isinstance(request.metadata, dict) else Mode.NORMAL
        request, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.TXT2IMG, request, logger=self._logger)
        _ = parse_params(self.engine_id, TaskType.TXT2IMG, request)
        p = build_txt2img_processing(request)
        # Model is (re)loaded by processing.manage_model_and_prompt_cache via forge_model_reload()
        yield ProgressEvent(stage="prepare", percent=5.0, message=f"sd15 txt2img model={self._model_ref}")
        processed = _processing.process_images(p)
        payload = {
            "images": processed.images + processed.extra_images,
            "info": processed.js(),
        }
        yield ResultEvent(payload=payload, metadata={"engine": self.engine_id, "task": TaskType.TXT2IMG.value, "model": self._model_ref or ""})

    def img2img(self, request: Img2ImgRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        mode = Mode(str((request.metadata or {}).get("mode", "Normal"))) if isinstance(request.metadata, dict) else Mode.NORMAL
        request, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.IMG2IMG, request, logger=self._logger)
        _ = parse_params(self.engine_id, TaskType.IMG2IMG, request)
        p = build_img2img_processing(request)
        yield ProgressEvent(stage="prepare", percent=5.0, message=f"sd15 img2img model={self._model_ref}")
        processed = _processing.process_images(p)
        payload = {
            "images": processed.images + processed.extra_images,
            "info": processed.js(),
        }
        yield ResultEvent(payload=payload, metadata={"engine": self.engine_id, "task": TaskType.IMG2IMG.value, "model": self._model_ref or ""})

    def inpaint(self, request: Img2ImgRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        mode = Mode(str((request.metadata or {}).get("mode", "Normal"))) if isinstance(request.metadata, dict) else Mode.NORMAL
        request, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.INPAINT, request, logger=self._logger)
        _ = parse_params(self.engine_id, TaskType.INPAINT, request)
        p = build_img2img_processing(request)
        yield ProgressEvent(stage="prepare", percent=5.0, message=f"sd15 inpaint model={self._model_ref}")
        processed = _processing.process_images(p)
        payload = {
            "images": processed.images + processed.extra_images,
            "info": processed.js(),
        }
        yield ResultEvent(payload=payload, metadata={"engine": self.engine_id, "task": TaskType.INPAINT.value, "model": self._model_ref or ""})

    def upscale(self, request: Txt2ImgRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        mode = Mode(str((request.metadata or {}).get("mode", "Normal"))) if isinstance(request.metadata, dict) else Mode.NORMAL
        request, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.UPSCALE, request, logger=self._logger)
        _ = parse_params(self.engine_id, TaskType.UPSCALE, request)
        # Use hires-fix path from txt2img by forcing highres settings
        hr = dict(request.highres_fix or {})
        if not hr:
            hr = {"scale": 2.0, "denoise": 0.2, "upscaler": "Latent", "steps": max(0, (request.steps or 20) // 2)}
        req = Txt2ImgRequest(
            task=request.task,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            sampler=request.sampler,
            scheduler=request.scheduler,
            seed=request.seed,
            guidance_scale=request.guidance_scale,
            batch_size=max(1, request.batch_size),
            loras=request.loras,
            extra_networks=request.extra_networks,
            clip_skip=request.clip_skip,
            metadata=request.metadata,
            width=request.width,
            height=request.height,
            steps=request.steps,
            tiling=False,
            highres_fix=hr,
            extras=request.extras,
        )
        p = build_txt2img_processing(req)
        yield ProgressEvent(stage="prepare", percent=5.0, message=f"sd15 upscale (hires) model={self._model_ref}")
        processed = _processing.process_images(p)
        payload = {
            "images": processed.images + processed.extra_images,
            "info": processed.js(),
        }
        yield ResultEvent(payload=payload, metadata={"engine": self.engine_id, "task": TaskType.UPSCALE.value, "model": self._model_ref or ""})

    # ------------------------------------------------------------------
    def load(self, model_ref: str, **options: object) -> None:  # type: ignore[override]
        # Resolve checkpoint and set Forge loading parameters used by forge_model_reload()
        checkpoint = _sd_models.get_closet_checkpoint_match(model_ref)
        if checkpoint is None:
            raise EngineLoadError(f"sd15 load failed: checkpoint not found for '{model_ref}'")

        additional_modules = []
        if isinstance(options.get("additional_modules"), (list, tuple)):
            additional_modules = list(options.get("additional_modules") or [])

        _sd_models.model_data.forge_loading_parameters = dict(
            checkpoint_info=checkpoint,
            additional_modules=additional_modules,
            unet_storage_dtype=options.get("unet_storage_dtype"),
        )
        self.mark_loaded()

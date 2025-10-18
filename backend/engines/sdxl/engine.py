"""Stable Diffusion XL engine bridged to existing WebUI processing."""

from __future__ import annotations

from typing import Iterator

from backend.core.engine_interface import EngineCapabilities, TaskType
from backend.core.requests import Img2ImgRequest, InferenceEvent, ProgressEvent, ResultEvent, Txt2ImgRequest
from backend.core.param_registry import parse_params
from backend.core.presets import apply_preset_to_request
from backend.core.enums import Mode
from backend.core.exceptions import EngineLoadError

from ..base import DiffusionEngine
from ..util.adapters import build_txt2img_processing, build_img2img_processing

# WebUI modules
from modules import processing as _processing
from modules import sd_models as _sd_models
from modules import shared as _shared
from modules import devices as _devices


class SDXLEngine(DiffusionEngine):
    engine_id = "sdxl"

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(
                TaskType.TXT2IMG,
                TaskType.IMG2IMG,
                TaskType.INPAINT,
                TaskType.UPSCALE,
            ),
            model_types=("sdxl", "stable-diffusion-xl", "sdxl-turbo"),
            precision=("fp32", "fp16"),
            extras={"notes": "Bridged to modules.processing"},
        )

    def txt2img(self, request: Txt2ImgRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.debug(
            "txt2img request: model=%s prompt=%r neg=%r size=%dx%d steps=%s cfg=%s sampler=%s scheduler=%s seed=%s batch=%s",
            self._model_ref,
            (request.prompt or "")[:128],
            (request.negative_prompt or "")[:96],
            request.width,
            request.height,
            request.steps,
            request.guidance_scale,
            request.sampler,
            request.scheduler,
            request.seed,
            request.batch_size,
        )
        self._logger.debug(
            "device=%s dtype_unet=%s dtype_vae=%s", _devices.get_optimal_device_name(), getattr(_devices, "dtype_unet", None), getattr(_devices, "dtype_vae", None)
        )
        mode = Mode(str((request.metadata or {}).get("mode", "Normal"))) if isinstance(request.metadata, dict) else Mode.NORMAL
        request, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.TXT2IMG, request, logger=self._logger)
        _ = parse_params(self.engine_id, TaskType.TXT2IMG, request)
        p = build_txt2img_processing(request)
        yield ProgressEvent(stage="prepare", percent=5.0, message=f"sdxl txt2img model={self._model_ref}")
        try:
            processed = _processing.process_images(p)
        except Exception:
            self._logger.exception("txt2img crashed inside process_images (sdxl)")
            raise
        payload = {"images": processed.images + processed.extra_images, "info": processed.js()}
        yield ResultEvent(payload=payload, metadata={"engine": self.engine_id, "task": TaskType.TXT2IMG.value, "model": self._model_ref or ""})

    def img2img(self, request: Img2ImgRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.debug(
            "img2img request: model=%s prompt=%r neg=%r size=%dx%d steps=%s cfg=%s sampler=%s scheduler=%s seed=%s batch=%s denoise=%s has_init=%s has_mask=%s",
            self._model_ref,
            (request.prompt or "")[:128],
            (request.negative_prompt or "")[:96],
            request.width,
            request.height,
            request.steps,
            request.guidance_scale,
            request.sampler,
            request.scheduler,
            request.seed,
            request.batch_size,
            getattr(request, "denoise_strength", None),
            bool(getattr(request, "init_image", None)),
            bool(getattr(request, "mask", None)),
        )
        self._logger.debug(
            "device=%s dtype_unet=%s dtype_vae=%s", _devices.get_optimal_device_name(), getattr(_devices, "dtype_unet", None), getattr(_devices, "dtype_vae", None)
        )
        mode = Mode(str((request.metadata or {}).get("mode", "Normal"))) if isinstance(request.metadata, dict) else Mode.NORMAL
        request, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.IMG2IMG, request, logger=self._logger)
        _ = parse_params(self.engine_id, TaskType.IMG2IMG, request)
        p = build_img2img_processing(request)
        yield ProgressEvent(stage="prepare", percent=5.0, message=f"sdxl img2img model={self._model_ref}")
        try:
            processed = _processing.process_images(p)
        except Exception:
            self._logger.exception("img2img crashed inside process_images (sdxl)")
            raise
        payload = {"images": processed.images + processed.extra_images, "info": processed.js()}
        yield ResultEvent(payload=payload, metadata={"engine": self.engine_id, "task": TaskType.IMG2IMG.value, "model": self._model_ref or ""})

    def inpaint(self, request: Img2ImgRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.debug(
            "inpaint request: model=%s prompt=%r size=%dx%d steps=%s denoise=%s has_mask=%s",
            self._model_ref,
            (request.prompt or "")[:128],
            request.width,
            request.height,
            request.steps,
            getattr(request, "denoise_strength", None),
            bool(getattr(request, "mask", None)),
        )
        mode = Mode(str((request.metadata or {}).get("mode", "Normal"))) if isinstance(request.metadata, dict) else Mode.NORMAL
        request, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.INPAINT, request, logger=self._logger)
        _ = parse_params(self.engine_id, TaskType.INPAINT, request)
        p = build_img2img_processing(request)
        yield ProgressEvent(stage="prepare", percent=5.0, message=f"sdxl inpaint model={self._model_ref}")
        try:
            processed = _processing.process_images(p)
        except Exception:
            self._logger.exception("inpaint crashed inside process_images (sdxl)")
            raise
        payload = {"images": processed.images + processed.extra_images, "info": processed.js()}
        yield ResultEvent(payload=payload, metadata={"engine": self.engine_id, "task": TaskType.INPAINT.value, "model": self._model_ref or ""})

    def upscale(self, request: Txt2ImgRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.debug(
            "upscale request: model=%s w=%s h=%s steps=%s sampler=%s scheduler=%s hr=%s",
            self._model_ref,
            request.width,
            request.height,
            request.steps,
            request.sampler,
            request.scheduler,
            bool(request.highres_fix),
        )
        mode = Mode(str((request.metadata or {}).get("mode", "Normal"))) if isinstance(request.metadata, dict) else Mode.NORMAL
        request, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.UPSCALE, request, logger=self._logger)
        _ = parse_params(self.engine_id, TaskType.UPSCALE, request)
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
        yield ProgressEvent(stage="prepare", percent=5.0, message=f"sdxl upscale (hires) model={self._model_ref}")
        try:
            processed = _processing.process_images(p)
        except Exception:
            self._logger.exception("upscale (hires) crashed inside process_images (sdxl)")
            raise
        payload = {"images": processed.images + processed.extra_images, "info": processed.js()}
        yield ResultEvent(payload=payload, metadata={"engine": self.engine_id, "task": TaskType.UPSCALE.value, "model": self._model_ref or ""})

    # ------------------------------------------------------------------
    def load(self, model_ref: str, **options: object) -> None:  # type: ignore[override]
        self._logger.info("loading sdxl checkpoint=%s options=%s", model_ref, options or {})
        checkpoint = _sd_models.get_closet_checkpoint_match(model_ref)
        if checkpoint is None:
            raise EngineLoadError(f"sdxl load failed: checkpoint not found for '{model_ref}'")

        additional_modules = []
        if isinstance(options.get("additional_modules"), (list, tuple)):
            additional_modules = list(options.get("additional_modules") or [])

        _sd_models.model_data.forge_loading_parameters = dict(
            checkpoint_info=checkpoint,
            additional_modules=additional_modules,
            unet_storage_dtype=options.get("unet_storage_dtype"),
        )
        self.mark_loaded()
        self._logger.info("sdxl loaded: ckpt=%s add_modules=%s", getattr(checkpoint, "name", checkpoint), additional_modules)

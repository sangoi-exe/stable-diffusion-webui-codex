from __future__ import annotations

from typing import Callable, Dict, Tuple, Union

from .engine_interface import TaskType
from .enums import EngineKey
from .requests import Txt2ImgRequest, Img2ImgRequest, Txt2VidRequest, Img2VidRequest
from .params import (
    SD15Txt2ImgParams, SD15Img2ImgParams, SD15InpaintParams, SD15UpscaleParams,
    SDXLTxt2ImgParams, SDXLImg2ImgParams, SDXLInpaintParams, SDXLUpscaleParams,
    FluxTxt2ImgParams,
    SVDImg2VidParams,
    HunyuanTxt2VidParams, HunyuanImg2VidParams,
    WanTI2V5BParamsT2V, WanTI2V5BParamsI2V,
    WanT2V14BParams, WanI2V14BParams,
)
from .params.base import ValidationError


Parser = Callable[[object], object]


def _sd15_txt2img(req: Txt2ImgRequest) -> SD15Txt2ImgParams:
    if not req.prompt:
        raise ValidationError("sd15/txt2img: 'prompt' is required")
    return SD15Txt2ImgParams(
        prompt=req.prompt,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed or -1),
        guidance_scale=float(req.guidance_scale or 7.0),
    )


def _sd15_img2img(req: Img2ImgRequest) -> SD15Img2ImgParams:
    if req.init_image is None:
        raise ValidationError("sd15/img2img: 'init_image' is required")
    return SD15Img2ImgParams(
        prompt=req.prompt,
        init_image=req.init_image,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed or -1),
        guidance_scale=float(req.guidance_scale or 7.0),
        denoise_strength=float(req.denoise_strength),
    )


def _sd15_inpaint(req: Img2ImgRequest) -> SD15InpaintParams:
    if req.init_image is None or req.mask is None:
        raise ValidationError("sd15/inpaint: 'init_image' and 'mask' are required")
    base = _sd15_img2img(req)
    return SD15InpaintParams(**{**base.as_dict(), "mask": req.mask})


def _sd15_upscale(req: Txt2ImgRequest) -> SD15UpscaleParams:
    hr = req.highres_fix or {}
    return SD15UpscaleParams(
        prompt=req.prompt,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed or -1),
        guidance_scale=float(req.guidance_scale or 7.0),
        hr_scale=float(hr.get("scale", 2.0)),
        hr_denoise=float(hr.get("denoise", 0.2)),
        hr_upscaler=str(hr.get("upscaler", "Latent")),
        hr_steps=int(hr.get("steps", 0)),
    )


def _sdxl_txt2img(req: Txt2ImgRequest) -> SDXLTxt2ImgParams:
    return SDXLTxt2ImgParams(
        prompt=req.prompt,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed or -1),
        guidance_scale=float(req.guidance_scale or 7.0),
    )


def _sdxl_img2img(req: Img2ImgRequest) -> SDXLImg2ImgParams:
    if req.init_image is None:
        raise ValidationError("sdxl/img2img: 'init_image' is required")
    return SDXLImg2ImgParams(
        prompt=req.prompt,
        init_image=req.init_image,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed or -1),
        guidance_scale=float(req.guidance_scale or 7.0),
        denoise_strength=float(req.denoise_strength),
    )


def _sdxl_inpaint(req: Img2ImgRequest) -> SDXLInpaintParams:
    if req.init_image is None or req.mask is None:
        raise ValidationError("sdxl/inpaint: 'init_image' and 'mask' are required")
    base = _sdxl_img2img(req)
    return SDXLInpaintParams(**{**base.as_dict(), "mask": req.mask})


def _sdxl_upscale(req: Txt2ImgRequest) -> SDXLUpscaleParams:
    hr = req.highres_fix or {}
    return SDXLUpscaleParams(
        prompt=req.prompt,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed or -1),
        guidance_scale=float(req.guidance_scale or 7.0),
        hr_scale=float(hr.get("scale", 2.0)),
        hr_denoise=float(hr.get("denoise", 0.2)),
        hr_upscaler=str(hr.get("upscaler", "Latent")),
        hr_steps=int(hr.get("steps", 0)),
    )


def _flux_txt2img(req: Txt2ImgRequest) -> FluxTxt2ImgParams:
    return FluxTxt2ImgParams(
        prompt=req.prompt,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed or -1),
        guidance_scale=float(req.guidance_scale or 1.0),
        distilled_cfg_scale=float(req.metadata.get("distilled_cfg_scale")) if isinstance(req.metadata, dict) and req.metadata.get("distilled_cfg_scale") is not None else None,
    )


def _svd_img2vid(req: Img2VidRequest) -> SVDImg2VidParams:
    if req.init_image is None:
        raise ValidationError("svd/img2vid: 'init_image' is required")
    return SVDImg2VidParams(
        init_image=req.init_image,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        fps=int(req.fps or 24),
        num_frames=int(req.num_frames or 16),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed) if req.seed is not None else None,
        guidance_scale=float(req.guidance_scale) if req.guidance_scale is not None else None,
        motion_strength=float(req.motion_strength) if req.motion_strength is not None else None,
    )


def _hunyuan_txt2vid(req: Txt2VidRequest) -> HunyuanTxt2VidParams:
    return HunyuanTxt2VidParams(
        prompt=req.prompt,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        fps=int(req.fps or 24),
        num_frames=int(req.num_frames or 16),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed) if req.seed is not None else None,
        guidance_scale=float(req.guidance_scale) if req.guidance_scale is not None else None,
    )


def _hunyuan_img2vid(req: Img2VidRequest) -> HunyuanImg2VidParams:
    if req.init_image is None:
        raise ValidationError("hunyuan/img2vid: 'init_image' is required")
    return HunyuanImg2VidParams(
        init_image=req.init_image,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        fps=int(req.fps or 24),
        num_frames=int(req.num_frames or 16),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed) if req.seed is not None else None,
        guidance_scale=float(req.guidance_scale) if req.guidance_scale is not None else None,
        motion_strength=float(req.motion_strength) if req.motion_strength is not None else None,
    )


def _wan_ti2v_t2v(req: Txt2VidRequest) -> WanTI2V5BParamsT2V:
    return WanTI2V5BParamsT2V(
        prompt=req.prompt,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        fps=int(req.fps or 24),
        num_frames=int(req.num_frames or 16),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed) if req.seed is not None else None,
        guidance_scale=float(req.guidance_scale) if req.guidance_scale is not None else None,
    )


def _wan_ti2v_i2v(req: Img2VidRequest) -> WanTI2V5BParamsI2V:
    if req.init_image is None:
        raise ValidationError("wan_ti2v_5b/img2vid: 'init_image' is required")
    return WanTI2V5BParamsI2V(
        init_image=req.init_image,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        fps=int(req.fps or 24),
        num_frames=int(req.num_frames or 16),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed) if req.seed is not None else None,
        guidance_scale=float(req.guidance_scale) if req.guidance_scale is not None else None,
        motion_strength=float(req.motion_strength) if req.motion_strength is not None else None,
    )


def _wan_t2v_14b(req: Txt2VidRequest) -> WanT2V14BParams:
    return WanT2V14BParams(
        prompt=req.prompt,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        fps=int(req.fps or 24),
        num_frames=int(req.num_frames or 16),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed) if req.seed is not None else None,
        guidance_scale=float(req.guidance_scale) if req.guidance_scale is not None else None,
    )


def _wan_i2v_14b(req: Img2VidRequest) -> WanI2V14BParams:
    if req.init_image is None:
        raise ValidationError("wan_i2v_14b/img2vid: 'init_image' is required")
    return WanI2V14BParams(
        init_image=req.init_image,
        width=int(req.width),
        height=int(req.height),
        steps=int(req.steps or 0),
        fps=int(req.fps or 24),
        num_frames=int(req.num_frames or 16),
        sampler=str(req.sampler or "Automatic"),
        scheduler=str(req.scheduler or "Automatic"),
        seed=int(req.seed) if req.seed is not None else None,
        guidance_scale=float(req.guidance_scale) if req.guidance_scale is not None else None,
        motion_strength=float(req.motion_strength) if req.motion_strength is not None else None,
    )


_PARSERS: Dict[Tuple[str, TaskType], Parser] = {
    ("sd15", TaskType.TXT2IMG): _sd15_txt2img,
    ("sd15", TaskType.IMG2IMG): _sd15_img2img,
    ("sd15", TaskType.INPAINT): _sd15_inpaint,
    ("sd15", TaskType.UPSCALE): _sd15_upscale,
    ("sdxl", TaskType.TXT2IMG): _sdxl_txt2img,
    ("sdxl", TaskType.IMG2IMG): _sdxl_img2img,
    ("sdxl", TaskType.INPAINT): _sdxl_inpaint,
    ("sdxl", TaskType.UPSCALE): _sdxl_upscale,
    ("flux", TaskType.TXT2IMG): _flux_txt2img,
    ("svd", TaskType.IMG2VID): _svd_img2vid,
    ("hunyuan_video", TaskType.TXT2VID): _hunyuan_txt2vid,
    ("hunyuan_video", TaskType.IMG2VID): _hunyuan_img2vid,
    ("wan_ti2v_5b", TaskType.TXT2VID): _wan_ti2v_t2v,
    ("wan_ti2v_5b", TaskType.IMG2VID): _wan_ti2v_i2v,
    ("wan_t2v_14b", TaskType.TXT2VID): _wan_t2v_14b,
    ("wan_i2v_14b", TaskType.IMG2VID): _wan_i2v_14b,
}


def parse_params(engine_key: Union[str, EngineKey], task: TaskType, request: object) -> object:
    key = engine_key.value if isinstance(engine_key, EngineKey) else str(engine_key)
    parser = _PARSERS.get((key, task))
    if parser is None:
        raise ValidationError(f"No parameter interface for engine='{key}', task='{task.value}'")
    return parser(request)  # type: ignore[misc]

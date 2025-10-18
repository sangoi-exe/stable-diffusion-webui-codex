from __future__ import annotations

from typing import Any, Mapping, Sequence
import logging

from modules import processing as _processing
from modules import shared as _shared

from backend.core.requests import Img2ImgRequest, Txt2ImgRequest

_log = logging.getLogger(__name__)


def build_txt2img_processing(req: Txt2ImgRequest) -> _processing.StableDiffusionProcessingTxt2Img:
    _log.debug(
        "build_txt2img_processing: size=%dx%d steps=%s sampler=%s scheduler=%s cfg=%s seed=%s hr=%s",
        req.width,
        req.height,
        req.steps,
        req.sampler,
        req.scheduler,
        req.guidance_scale,
        req.seed,
        bool(req.highres_fix),
    )
    opts = _shared.opts
    # Respect explicit enable flag inside highres_fix; a non-empty dict must not
    # implicitly enable hires. This fixes HR showing as enabled with scale=1.0.
    _hr = req.highres_fix if isinstance(req.highres_fix, dict) else {}
    _hr_enable = bool(_hr.get("enable", False))
    _hr_denoise = float(_hr.get("denoise", 0.5)) if _hr_enable else 0.0
    _hr_scale = float(_hr.get("scale", 2.0)) if _hr_enable else 1.0
    _hr_upscaler = _hr.get("upscaler", "Latent") if _hr_enable else None
    _hr_steps = int(_hr.get("steps", 0)) if _hr_enable else 0
    _hr_resize_x = int(_hr.get("resize_x", 0)) if _hr_enable else 0
    _hr_resize_y = int(_hr.get("resize_y", 0)) if _hr_enable else 0
    _hr_prompt = _hr.get("hr_prompt", "") if _hr_enable else ""
    _hr_neg_prompt = _hr.get("hr_negative_prompt", "") if _hr_enable else ""
    _hr_cfg = float(_hr.get("hr_cfg", 1.0)) if _hr_enable else 1.0
    _hr_distilled_cfg = float(_hr.get("hr_distilled_cfg", 3.5)) if _hr_enable else 3.5

    p = _processing.StableDiffusionProcessingTxt2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=req.prompt,
        styles=[],
        negative_prompt=req.negative_prompt,
        batch_size=req.batch_size or 1,
        n_iter=1,
        cfg_scale=req.guidance_scale or 7.0,
        distilled_cfg_scale=3.5,
        width=req.width,
        height=req.height,
        enable_hr=_hr_enable,
        denoising_strength=_hr_denoise,
        hr_scale=_hr_scale,
        hr_upscaler=_hr_upscaler,
        hr_second_pass_steps=_hr_steps,
        hr_resize_x=_hr_resize_x,
        hr_resize_y=_hr_resize_y,
        hr_checkpoint_name=None,
        hr_additional_modules=["Use same choices"],
        hr_sampler_name=None,
        hr_scheduler=None,
        hr_prompt=_hr_prompt,
        hr_negative_prompt=_hr_neg_prompt,
        hr_cfg=_hr_cfg,
        hr_distilled_cfg=_hr_distilled_cfg,
        override_settings={},
    )

    p.scripts = None
    p.script_args = {}
    p.steps = req.steps or 20
    p.sampler_name = req.sampler or None
    p.scheduler = req.scheduler or None
    p.seed = -1 if req.seed is None else int(req.seed)
    p.user = "engine"
    _log.debug(
        "processing_txt2img: enable_hr=%s hr_scale=%s hr_upscaler=%s hr_steps=%s hr_resize=(%s,%s)",
        bool(p.enable_hr),
        getattr(p, "hr_scale", None),
        getattr(p, "hr_upscaler", None),
        getattr(p, "hr_second_pass_steps", None),
        getattr(p, "hr_resize_x", None),
        getattr(p, "hr_resize_y", None),
    )
    return p


def build_img2img_processing(req: Img2ImgRequest) -> _processing.StableDiffusionProcessingImg2Img:
    _log.debug(
        "build_img2img_processing: size=%sx%s steps=%s sampler=%s scheduler=%s cfg=%s denoise=%s has_init=%s has_mask=%s",
        req.width,
        req.height,
        req.steps,
        req.sampler,
        req.scheduler,
        req.guidance_scale,
        getattr(req, "denoise_strength", None),
        bool(getattr(req, "init_image", None)),
        bool(getattr(req, "mask", None)),
    )
    opts = _shared.opts
    width = req.width
    height = req.height
    if getattr(req, "init_image", None) is not None:
        try:
            w, h = req.init_image.size  # type: ignore[attr-defined]
            width = width or w
            height = height or h
        except Exception:
            pass

    p = _processing.StableDiffusionProcessingImg2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        styles=[],
        batch_size=req.batch_size or 1,
        n_iter=1,
        cfg_scale=req.guidance_scale or 7.0,
        distilled_cfg_scale=3.5,
        width=width,
        height=height,
        init_images=[req.init_image] if getattr(req, "init_image", None) is not None else None,
        mask=req.mask if getattr(req, "mask", None) is not None else None,
        denoising_strength=float(req.denoise_strength),
        inpaint_full_res=True,
        inpaint_full_res_padding=0,
        image_cfg_scale=None,
        resize_mode=0,
        override_settings={},
    )

    p.scripts = None
    p.script_args = {}
    p.steps = req.steps or 20
    p.sampler_name = req.sampler or None
    p.scheduler = req.scheduler or None
    p.seed = -1 if req.seed is None else int(req.seed)
    p.user = "engine"
    _log.debug(
        "processing_img2img: denoise=%s init_images=%s mask=%s",
        getattr(p, "denoising_strength", None),
        bool(getattr(p, "init_images", None)),
        bool(getattr(p, "mask", None)),
    )
    return p

from __future__ import annotations

from typing import Any, Mapping, Sequence

from modules import processing as _processing
from modules import shared as _shared

from backend.core.requests import Img2ImgRequest, Txt2ImgRequest


def build_txt2img_processing(req: Txt2ImgRequest) -> _processing.StableDiffusionProcessingTxt2Img:
    opts = _shared.opts
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
        enable_hr=bool(req.highres_fix),
        denoising_strength=float(req.highres_fix.get("denoise", 0.5)) if isinstance(req.highres_fix, dict) else 0.0,
        hr_scale=float(req.highres_fix.get("scale", 2.0)) if isinstance(req.highres_fix, dict) else 1.0,
        hr_upscaler=req.highres_fix.get("upscaler", "Latent") if isinstance(req.highres_fix, dict) else None,
        hr_second_pass_steps=int(req.highres_fix.get("steps", 0)) if isinstance(req.highres_fix, dict) else 0,
        hr_resize_x=int(req.highres_fix.get("resize_x", 0)) if isinstance(req.highres_fix, dict) else 0,
        hr_resize_y=int(req.highres_fix.get("resize_y", 0)) if isinstance(req.highres_fix, dict) else 0,
        hr_checkpoint_name=None,
        hr_additional_modules=["Use same choices"],
        hr_sampler_name=None,
        hr_scheduler=None,
        hr_prompt=req.metadata.get("hr_prompt", "") if req.metadata else "",
        hr_negative_prompt=req.metadata.get("hr_negative_prompt", "") if req.metadata else "",
        hr_cfg=float(req.metadata.get("hr_cfg", 1.0)) if req.metadata else 1.0,
        hr_distilled_cfg=float(req.metadata.get("hr_distilled_cfg", 3.5)) if req.metadata else 3.5,
        override_settings={},
    )

    p.scripts = None
    p.script_args = {}
    p.steps = req.steps or 20
    p.sampler_name = req.sampler or None
    p.scheduler = req.scheduler or None
    p.seed = -1 if req.seed is None else int(req.seed)
    p.user = "engine"
    return p


def build_img2img_processing(req: Img2ImgRequest) -> _processing.StableDiffusionProcessingImg2Img:
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
    return p


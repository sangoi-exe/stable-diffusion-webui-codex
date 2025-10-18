import os
from contextlib import closing
from pathlib import Path

from PIL import Image, ImageOps, ImageFilter, ImageEnhance, UnidentifiedImageError
import gradio as gr

from modules import images
from modules.infotext_utils import create_override_settings_dict, parse_generation_parameters
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
from modules.sd_models import get_closet_checkpoint_match
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.scripts
from modules_forge import main_thread

# New modular pipeline imports (Codex)
from backend.core.engine_interface import TaskType
from backend.core.orchestrator import InferenceOrchestrator
from backend.core.requests import Img2ImgRequest
from backend.core.requests import ProgressEvent, ResultEvent


def process_batch(proc, input, output_dir, inpaint_mask_dir, args, to_scale=False, scale_by=1.0, use_png_info=False, png_info_props=None, png_info_dir=None):
    output_dir = output_dir.strip()
    processing.fix_seed(proc)

    if isinstance(input, str):
        batch_images = list(shared.walk_files(input, allowed_extensions=(".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".avif")))
    else:
        batch_images = [os.path.abspath(x.name) for x in input]

    is_inpaint_batch = False
    if inpaint_mask_dir:
        inpaint_masks = shared.listfiles(inpaint_mask_dir)
        is_inpaint_batch = bool(inpaint_masks)

        if is_inpaint_batch:
            print(f"\nInpaint batch is enabled. {len(inpaint_masks)} masks found.")

    print(f"Will process {len(batch_images)} images, creating {proc.n_iter * proc.batch_size} new images for each.")

    state.job_count = len(batch_images) * proc.n_iter

    # extract "default" params to use in case getting png info fails
    prompt = proc.prompt
    negative_prompt = proc.negative_prompt
    seed = proc.seed
    cfg_scale = proc.cfg_scale
    sampler_name = proc.sampler_name
    steps = proc.steps
    override_settings = proc.override_settings
    sd_model_checkpoint_override = get_closet_checkpoint_match(override_settings.get("sd_model_checkpoint", None))
    batch_results = None
    discard_further_results = False
    for i, image in enumerate(batch_images):
        state.job = f"{i+1} out of {len(batch_images)}"
        if state.skipped:
            state.skipped = False

        if state.interrupted or state.stopping_generation:
            break

        try:
            img = images.read(image)
        except UnidentifiedImageError as e:
            print(e)
            continue
        # Use the EXIF orientation of photos taken by smartphones.
        img = ImageOps.exif_transpose(img)

        if to_scale:
            proc.width = int(img.width * scale_by)
            proc.height = int(img.height * scale_by)

        proc.init_images = [img] * proc.batch_size

        image_path = Path(image)
        if is_inpaint_batch:
            # try to find corresponding mask for an image using simple filename matching
            if len(inpaint_masks) == 1:
                mask_image_path = inpaint_masks[0]
            else:
                # try to find corresponding mask for an image using simple filename matching
                mask_image_dir = Path(inpaint_mask_dir)
                masks_found = list(mask_image_dir.glob(f"{image_path.stem}.*"))

                if len(masks_found) == 0:
                    print(f"Warning: mask is not found for {image_path} in {mask_image_dir}. Skipping it.")
                    continue

                # it should contain only 1 matching mask
                # otherwise user has many masks with the same name but different extensions
                mask_image_path = masks_found[0]

            mask_image = images.read(mask_image_path)
            proc.image_mask = mask_image

        if use_png_info:
            try:
                info_img = img
                if png_info_dir:
                    info_img_path = os.path.join(png_info_dir, os.path.basename(image))
                    info_img = images.read(info_img_path)
                geninfo, _ = images.read_info_from_image(info_img)
                parsed_parameters = parse_generation_parameters(geninfo)
                parsed_parameters = {k: v for k, v in parsed_parameters.items() if k in (png_info_props or {})}
            except Exception:
                parsed_parameters = {}

            proc.prompt = prompt + (" " + parsed_parameters["Prompt"] if "Prompt" in parsed_parameters else "")
            proc.negative_prompt = negative_prompt + (" " + parsed_parameters["Negative prompt"] if "Negative prompt" in parsed_parameters else "")
            proc.seed = int(parsed_parameters.get("Seed", seed))
            proc.cfg_scale = float(parsed_parameters.get("CFG scale", cfg_scale))
            proc.sampler_name = parsed_parameters.get("Sampler", sampler_name)
            proc.steps = int(parsed_parameters.get("Steps", steps))

            model_info = get_closet_checkpoint_match(parsed_parameters.get("Model hash", None))
            if model_info is not None:
                proc.override_settings['sd_model_checkpoint'] = model_info.name
            elif sd_model_checkpoint_override:
                proc.override_settings['sd_model_checkpoint'] = sd_model_checkpoint_override
            else:
                proc.override_settings.pop("sd_model_checkpoint", None)

        if output_dir:
            proc.outpath_samples = output_dir
            proc.override_settings['save_to_dirs'] = False

        if opts.img2img_batch_use_original_name:
            filename_pattern = f'{image_path.stem}-[generation_number]' if proc.n_iter > 1 or proc.batch_size > 1 else f'{image_path.stem}'
            proc.override_settings['samples_filename_pattern'] = filename_pattern

        result = modules.scripts.scripts_img2img.run(proc, *args)

        if result is None:
            result = process_images(proc)

        if not discard_further_results and result:
            if batch_results:
                batch_results.images.extend(result.images)
                batch_results.infotexts.extend(result.infotexts)
            else:
                batch_results = result

            if 0 <= shared.opts.img2img_batch_show_results_limit < len(batch_results.images):
                discard_further_results = True
                batch_results.images = batch_results.images[:int(shared.opts.img2img_batch_show_results_limit)]
                batch_results.infotexts = batch_results.infotexts[:int(shared.opts.img2img_batch_show_results_limit)]

    return batch_results


def img2img_function(id_task: str, request: gr.Request, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, sketch_fg, init_img_with_mask, init_img_with_mask_fg, inpaint_color_sketch, inpaint_color_sketch_fg, init_img_inpaint, init_mask_inpaint, mask_blur: int, mask_alpha: float, inpainting_fill: int, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, image_cfg_scale: float, denoising_strength: float, selected_scale_tab: int, height: int, width: int, scale_by: float, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, img2img_batch_use_png_info: bool, img2img_batch_png_info_props: list, img2img_batch_png_info_dir: str, img2img_batch_source_type: str, img2img_batch_upload: list, *args):

    override_settings = create_override_settings_dict(override_settings_texts)

    is_batch = mode == 5

    height, width = int(height), int(width)

    image = None
    mask = None

    if mode == 0:  # img2img
        image = init_img
        mask = None
    elif mode == 1:  # img2img sketch
        mask = None
        image = Image.alpha_composite(sketch, sketch_fg)
    elif mode == 2:  # inpaint
        image = init_img_with_mask
        mask = init_img_with_mask_fg.getchannel('A').convert('L')
        mask = Image.merge('RGBA', (mask, mask, mask, Image.new('L', mask.size, 255)))
    elif mode == 3:  # inpaint sketch
        image = Image.alpha_composite(inpaint_color_sketch, inpaint_color_sketch_fg)
        mask = inpaint_color_sketch_fg.getchannel('A').convert('L')
        short_side = min(mask.size)
        dilation_size = int(0.015 * short_side) * 2 + 1
        mask = mask.filter(ImageFilter.MaxFilter(dilation_size))
        mask = Image.merge('RGBA', (mask, mask, mask, Image.new('L', mask.size, 255)))
    elif mode == 4:  # inpaint upload mask
        image = init_img_inpaint
        mask = init_mask_inpaint

    if mask and isinstance(mask, Image.Image):
        mask = mask.point(lambda v: 255 if v > 128 else 0)

    image = images.fix_image(image)
    mask = images.fix_image(mask)

    if selected_scale_tab == 1 and not is_batch:
        assert image, "Can't scale by because no image is selected"

        width = int(image.width * scale_by)
        width -= width % 8
        height = int(image.height * scale_by)
        height -= height % 8

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    proc = StableDiffusionProcessingImg2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=prompt_styles,
        batch_size=batch_size,
        n_iter=n_iter,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        override_settings=override_settings,
        distilled_cfg_scale=distilled_cfg_scale
    )

    proc.scripts = modules.scripts.scripts_img2img
    proc.script_args = args

    proc.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

    with closing(proc):
        if is_batch:
            if img2img_batch_source_type == "upload":
                assert isinstance(img2img_batch_upload, list) and img2img_batch_upload
                output_dir = ""
                inpaint_mask_dir = ""
                png_info_dir = img2img_batch_png_info_dir if not shared.cmd_opts.hide_ui_dir_config else ""
                processed = process_batch(proc, img2img_batch_upload, output_dir, inpaint_mask_dir, args, to_scale=selected_scale_tab == 1, scale_by=scale_by, use_png_info=img2img_batch_use_png_info, png_info_props=img2img_batch_png_info_props, png_info_dir=png_info_dir)
            else: # "from dir"
                assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
                processed = process_batch(proc, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, args, to_scale=selected_scale_tab == 1, scale_by=scale_by, use_png_info=img2img_batch_use_png_info, png_info_props=img2img_batch_png_info_props, png_info_dir=img2img_batch_png_info_dir)

            if processed is None:
                processed = Processed(proc, [], proc.seed, "")
        else:
            processed = modules.scripts.scripts_img2img.run(proc, *args)
            if processed is None:
                processed = process_images(proc)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images + processed.extra_images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def img2img(id_task: str, request: gr.Request, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, sketch_fg, init_img_with_mask, init_img_with_mask_fg, inpaint_color_sketch, inpaint_color_sketch_fg, init_img_inpaint, init_mask_inpaint, mask_blur: int, mask_alpha: float, inpainting_fill: int, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, image_cfg_scale: float, denoising_strength: float, selected_scale_tab: int, height: int, width: int, scale_by: float, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, img2img_batch_use_png_info: bool, img2img_batch_png_info_props: list, img2img_batch_png_info_dir: str, img2img_batch_source_type: str, img2img_batch_upload: list, *args):
    return main_thread.run_and_wait_result(img2img_function, id_task, request, mode, prompt, negative_prompt, prompt_styles, init_img, sketch, sketch_fg, init_img_with_mask, init_img_with_mask_fg, inpaint_color_sketch, inpaint_color_sketch_fg, init_img_inpaint, init_mask_inpaint, mask_blur, mask_alpha, inpainting_fill, n_iter, batch_size, cfg_scale, distilled_cfg_scale, image_cfg_scale, denoising_strength, selected_scale_tab, height, width, scale_by, resize_mode, inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, override_settings_texts, img2img_batch_use_png_info, img2img_batch_png_info_props, img2img_batch_png_info_dir, img2img_batch_source_type, img2img_batch_upload, *args)


# Strict JSON helpers
def _require(payload: dict, key: str):
    if key not in payload:
        raise ValueError(f"Missing required field in payload: {key}")
    return payload[key]


def _as_int(payload: dict, key: str) -> int:
    v = _require(payload, key)
    if isinstance(v, bool):
        raise ValueError(f"Field {key} must be int, got bool")
    try:
        return int(v)
    except Exception:
        raise ValueError(f"Field {key} must be int, got {v!r}")


def _as_float(payload: dict, key: str) -> float:
    v = _require(payload, key)
    try:
        return float(v)
    except Exception:
        raise ValueError(f"Field {key} must be float, got {v!r}")


def _as_bool(payload: dict, key: str) -> bool:
    v = _require(payload, key)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "1", "yes", "on"): return True
        if t in ("false", "0", "no", "off"): return False
    raise ValueError(f"Field {key} must be bool, got {v!r}")


def _as_list(payload: dict, key: str):
    v = _require(payload, key)
    if isinstance(v, list):
        return v
    raise ValueError(f"Field {key} must be list, got {type(v).__name__}")


def img2img_from_json(id_task: str,
                      request: gr.Request,
                      _prompt_passthrough,
                      _neg_passthrough,
                      _styles_dropdown,
                      init_img,
                      sketch,
                      sketch_fg,
                      init_img_with_mask,
                      init_img_with_mask_fg,
                      inpaint_color_sketch,
                      inpaint_color_sketch_fg,
                      init_img_inpaint,
                      init_mask_inpaint,
                      img2img_batch_input_dir: str,
                      img2img_batch_output_dir: str,
                      img2img_batch_inpaint_mask_dir: str,
                      img2img_batch_use_png_info: bool,
                      img2img_batch_png_info_props: list,
                      img2img_batch_png_info_dir: str,
                      img2img_batch_source_type: str,
                      img2img_batch_upload: list,
                      *rest):
    if not rest:
        raise ValueError("Missing JSON payload in img2img_from_json")
    # split custom script args vs payload (ignore positional script_args; compute from payload)
    payload = rest[-1]
    if isinstance(payload, str):
        try:
            import json as _json
            payload = _json.loads(payload)
        except Exception:
            raise ValueError("Invalid strict JSON payload: could not parse string")
    script_args = tuple()
    if not isinstance(payload, dict) or payload.get("__strict_version") != 1:
        raise ValueError("Invalid or missing __strict_version in payload; frontend must send strict JSON")

    if not isinstance(payload, dict) or payload.get("__strict_version") != 1:
        raise ValueError("Invalid or missing __strict_version in payload; frontend must send strict JSON")

    prompt = _require(payload, 'img2img_prompt') or ''
    negative_prompt = _require(payload, 'img2img_neg_prompt') or ''
    prompt_styles = _as_list(payload, 'img2img_styles')
    n_iter = _as_int(payload, 'img2img_batch_count')
    batch_size = _as_int(payload, 'img2img_batch_size')
    cfg_scale = _as_float(payload, 'img2img_cfg_scale')
    distilled_cfg_scale = _as_float(payload, 'img2img_distilled_cfg_scale')
    image_cfg_scale = _as_float(payload, 'img2img_image_cfg_scale')
    denoising_strength = _as_float(payload, 'img2img_denoising_strength')
    selected_scale_tab = _as_int(payload, 'img2img_selected_scale_tab')
    height = _as_int(payload, 'img2img_height')
    width = _as_int(payload, 'img2img_width')
    scale_by = _as_float(payload, 'img2img_scale_by')
    resize_mode = _as_int(payload, 'img2img_resize_mode')

    # Inpaint-related (required if on inpaint tabs; otherwise ignored)
    inpaint_full_res = bool(payload.get('img2img_inpaint_full_res', False))
    inpaint_full_res_padding = int(payload.get('img2img_inpaint_full_res_padding', 0))
    inpainting_mask_invert = int(payload.get('img2img_inpainting_mask_invert', 0))
    mask_blur = int(payload.get('img2img_mask_blur', 0))
    mask_alpha = float(payload.get('img2img_mask_alpha', 0.0))
    inpainting_fill = int(payload.get('img2img_inpainting_fill', 0))

    override_settings_texts = []

    # Early required keys for img2img core
    required_keys = [
        'img2img_prompt', 'img2img_neg_prompt', 'img2img_styles',
        'img2img_batch_count', 'img2img_batch_size',
        'img2img_cfg_scale', 'img2img_distilled_cfg_scale',
        'img2img_height', 'img2img_width',
        'img2img_steps', 'img2img_sampling', 'img2img_scheduler', 'img2img_seed',
    ]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        present = sorted([k for k in payload.keys() if not k.startswith('__')])[:16]
        raise ValueError(f"Strict JSON is missing required fields: {missing} | present_keys(sample)={present}")

    # Build Codex Img2ImgRequest (no legacy processing path)
    # Choose the best available source image/mask from the provided UI inputs
    base_image = None
    mask_image = None
    try:
        if init_img is not None:
            base_image = init_img
        elif sketch is not None:
            try:
                base_image = Image.alpha_composite(sketch, sketch_fg) if sketch_fg is not None else sketch
            except Exception:
                base_image = sketch
        elif init_img_with_mask is not None:
            base_image = init_img_with_mask
            if init_img_with_mask_fg is not None and isinstance(init_img_with_mask_fg, Image.Image):
                try:
                    m = init_img_with_mask_fg.getchannel('A').convert('L')
                    mask_image = m
                except Exception:
                    mask_image = None
        elif inpaint_color_sketch is not None:
            try:
                base_image = Image.alpha_composite(inpaint_color_sketch, inpaint_color_sketch_fg) if inpaint_color_sketch_fg is not None else inpaint_color_sketch
                if inpaint_color_sketch_fg is not None and isinstance(inpaint_color_sketch_fg, Image.Image):
                    mask_image = inpaint_color_sketch_fg.getchannel('A').convert('L')
            except Exception:
                base_image = inpaint_color_sketch
        elif init_img_inpaint is not None:
            base_image = init_img_inpaint
            if init_mask_inpaint is not None:
                mask_image = init_mask_inpaint
    except Exception:
        # Keep going; orchestrator will error with full context if inputs are invalid
        pass

    # Fall back to first non-null among upload lists if base_image is still None
    if base_image is None and isinstance(img2img_batch_upload, list) and img2img_batch_upload:
        try:
            p = img2img_batch_upload[0]
            if hasattr(p, 'name'):
                base_image = images.read(p.name)
        except Exception:
            base_image = None

    # Compose request
    req = Img2ImgRequest(
        task=TaskType.IMG2IMG,
        prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=base_image,
        mask=mask_image,
        denoise_strength=denoising_strength,
        width=width,
        height=height,
        steps=int(payload.get('img2img_steps', 20)),
        guidance_scale=cfg_scale,
        sampler=str(payload.get('img2img_sampling')),
        scheduler=str(payload.get('img2img_scheduler')),
        seed=int(payload.get('img2img_seed', -1)),
        batch_size=batch_size,
        metadata={
            "mode": getattr(shared.opts, 'codex_mode', 'Normal'),
            "styles": prompt_styles,
            "distilled_cfg_scale": distilled_cfg_scale,
            "image_cfg_scale": image_cfg_scale,
            "resize_mode": resize_mode,
            "scale_by": scale_by,
            "selected_scale_tab": selected_scale_tab,
            "inpaint_full_res": inpaint_full_res,
            "inpaint_full_res_padding": inpaint_full_res_padding,
            "inpainting_mask_invert": inpainting_mask_invert,
            "mask_blur": mask_blur,
            "mask_alpha": mask_alpha,
            "inpainting_fill": inpainting_fill,
        },
        extras={},
    )

    # Resolve engine/model from Quicksettings
    engine_key = getattr(shared.opts, 'codex_engine', 'sd15')
    model_ref = getattr(shared.opts, 'sd_model_checkpoint', None)

    orch = InferenceOrchestrator()
    images_out = []
    info_js = "{}"
    for ev in orch.run(TaskType.IMG2IMG, str(engine_key), req, model_ref=model_ref):
        if isinstance(ev, ResultEvent):
            payload = ev.payload or {}
            images_out = payload.get("images", [])
            info_js = payload.get("info", "{}")

    if opts.do_not_show_images:
        images_out = []

    return images_out, info_js, plaintext_to_html(""), plaintext_to_html("", classname="comments")

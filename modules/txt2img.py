import json
from contextlib import closing

import modules.scripts
from modules import processing, infotext_utils
from modules.infotext_utils import create_override_settings_dict, parse_generation_parameters
from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html
from PIL import Image
import gradio as gr
from modules_forge import main_thread


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


def _txt2img_from_payload(id_task: str, request: gr.Request, payload: dict, *script_args):
    if not isinstance(payload, dict) or payload.get("__strict_version") != 1:
        raise ValueError("Invalid or missing __strict_version in payload; frontend must send strict JSON")

    # Early validation to report all missing required fields at once
    required_keys = [
        'txt2img_prompt', 'txt2img_neg_prompt', 'txt2img_styles',
        'txt2img_batch_count', 'txt2img_batch_size',
        'txt2img_cfg_scale', 'txt2img_distilled_cfg_scale',
        'txt2img_height', 'txt2img_width', 'txt2img_hr_enable',
        'txt2img_steps', 'txt2img_sampling', 'txt2img_scheduler', 'txt2img_seed',
    ]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        present = sorted([k for k in payload.keys() if not k.startswith('__')])[:16]
        raise ValueError(f"Strict JSON is missing required fields: {missing} | present_keys(sample)={present}")

    prompt = _require(payload, 'txt2img_prompt') or ''
    negative_prompt = _require(payload, 'txt2img_neg_prompt') or ''
    prompt_styles = _as_list(payload, 'txt2img_styles')
    n_iter = _as_int(payload, 'txt2img_batch_count')
    batch_size = _as_int(payload, 'txt2img_batch_size')
    cfg_scale = _as_float(payload, 'txt2img_cfg_scale')
    distilled_cfg_scale = _as_float(payload, 'txt2img_distilled_cfg_scale')
    height = _as_int(payload, 'txt2img_height')
    width = _as_int(payload, 'txt2img_width')
    enable_hr = _as_bool(payload, 'txt2img_hr_enable')

    # Sampler core (required)
    steps = _as_int(payload, 'txt2img_steps')
    sampler_name = _require(payload, 'txt2img_sampling')
    scheduler_name = _require(payload, 'txt2img_scheduler')
    seed = _as_int(payload, 'txt2img_seed')

    # Hires fields required only if enabled
    if enable_hr:
        denoising_strength = _as_float(payload, 'txt2img_denoising_strength')
        hr_scale = _as_float(payload, 'txt2img_hr_scale')
        hr_upscaler = _require(payload, 'txt2img_hr_upscaler')
        hr_second_pass_steps = _as_int(payload, 'txt2img_hires_steps')
        hr_resize_x = _as_int(payload, 'txt2img_hr_resize_x')
        hr_resize_y = _as_int(payload, 'txt2img_hr_resize_y')
        hr_checkpoint_name = payload.get('hr_checkpoint_name') or _require(payload, 'hr_checkpoint')
        hr_additional_modules = payload.get('hr_additional_modules') or _as_list(payload, 'hr_vae_te')
        hr_sampler_name = payload.get('hr_sampler_name') or _require(payload, 'hr_sampler')
        hr_scheduler = payload.get('hr_scheduler') or _require(payload, 'hr_scheduler')
        hr_prompt = payload.get('txt2img_hr_prompt') or ''
        hr_negative_prompt = payload.get('txt2img_hr_neg_prompt') or ''
        hr_cfg = _as_float(payload, 'txt2img_hr_cfg')
        hr_distilled_cfg = _as_float(payload, 'txt2img_hr_distilled_cfg')
    else:
        denoising_strength = 0.0
        hr_scale = 1.0
        hr_upscaler = 'Use same upscaler'
        hr_second_pass_steps = 0
        hr_resize_x = 0
        hr_resize_y = 0
        hr_checkpoint_name = 'Use same checkpoint'
        hr_additional_modules = ['Use same choices']
        hr_sampler_name = 'Use same sampler'
        hr_scheduler = 'Use same scheduler'
        hr_prompt = ''
        hr_negative_prompt = ''
        hr_cfg = 7.0
        hr_distilled_cfg = 3.5

    override_settings_texts = []

    # Build script args from named payload (ignores positional script_args to avoid mismatches)
    script_args_payload = modules.scripts.build_script_args_from_payload(modules.scripts.scripts_txt2img, payload)

    proc = txt2img_create_processing(
        id_task,
        request,
        prompt,
        negative_prompt,
        prompt_styles,
        n_iter,
        batch_size,
        cfg_scale,
        distilled_cfg_scale,
        height,
        width,
        enable_hr,
        denoising_strength,
        hr_scale,
        hr_upscaler,
        hr_second_pass_steps,
        hr_resize_x,
        hr_resize_y,
        hr_checkpoint_name,
        hr_additional_modules,
        hr_sampler_name,
        hr_scheduler,
        hr_prompt,
        hr_negative_prompt,
        hr_cfg,
        hr_distilled_cfg,
        override_settings_texts,
        *script_args_payload,
    )

    # Apply sampler core onto processing object
    proc.steps = steps
    proc.sampler_name = sampler_name
    proc.scheduler = scheduler_name
    proc.seed = seed

    with closing(proc):
        processed = modules.scripts.scripts_txt2img.run(proc, *proc.script_args)
        if processed is None:
            processed = processing.process_images(proc)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images + processed.extra_images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def txt2img_from_json(id_task: str, request: gr.Request, payload, *script_args):
    # ignore positional script_args; use payload to build named args
    return _txt2img_from_payload(id_task, request, payload)


def txt2img_upscale_from_json(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, payload, *script_args):
    assert len(gallery) > 0, 'No image to upscale'
    if gallery_index < 0 or gallery_index >= len(gallery):
        return gallery, generation_info, f'Bad image index: {gallery_index}', ''

    # Build processing object with enable_hr forced on
    proc = None
    def _build():
        nonlocal proc
        proc = txt2img_create_processing(id_task, request, *_txt2img_params_from_payload(payload, force_enable_hr=True, script_args=script_args))

    # Helper to reuse the parser
    def _txt2img_params_from_payload(payload: dict, force_enable_hr: bool, script_args=()):
        if not isinstance(payload, dict) or payload.get("__strict_version") != 1:
            raise ValueError("Invalid or missing __strict_version in payload; frontend must send strict JSON")
        prompt = _require(payload, 'txt2img_prompt') or ''
        negative_prompt = _require(payload, 'txt2img_neg_prompt') or ''
        prompt_styles = _as_list(payload, 'txt2img_styles')
        n_iter = _as_int(payload, 'txt2img_batch_count')
        batch_size = _as_int(payload, 'txt2img_batch_size')
        cfg_scale = _as_float(payload, 'txt2img_cfg_scale')
        distilled_cfg_scale = _as_float(payload, 'txt2img_distilled_cfg_scale')
        height = _as_int(payload, 'txt2img_height')
        width = _as_int(payload, 'txt2img_width')
        enable_hr = True if force_enable_hr else _as_bool(payload, 'txt2img_hr_enable')
        if enable_hr:
            denoising_strength = _as_float(payload, 'txt2img_denoising_strength')
            hr_scale = _as_float(payload, 'txt2img_hr_scale')
            hr_upscaler = _require(payload, 'txt2img_hr_upscaler')
            hr_second_pass_steps = _as_int(payload, 'txt2img_hires_steps')
            hr_resize_x = _as_int(payload, 'txt2img_hr_resize_x')
            hr_resize_y = _as_int(payload, 'txt2img_hr_resize_y')
            hr_checkpoint_name = payload.get('hr_checkpoint_name') or _require(payload, 'hr_checkpoint')
            hr_additional_modules = payload.get('hr_additional_modules') or _as_list(payload, 'hr_vae_te')
            hr_sampler_name = payload.get('hr_sampler_name') or _require(payload, 'hr_sampler')
            hr_scheduler = payload.get('hr_scheduler') or _require(payload, 'hr_scheduler')
            hr_prompt = payload.get('txt2img_hr_prompt') or ''
            hr_negative_prompt = payload.get('txt2img_hr_neg_prompt') or ''
            hr_cfg = _as_float(payload, 'txt2img_hr_cfg')
            hr_distilled_cfg = _as_float(payload, 'txt2img_hr_distilled_cfg')
        else:
            denoising_strength = 0.0
            hr_scale = 1.0
            hr_upscaler = 'Use same upscaler'
            hr_second_pass_steps = 0
            hr_resize_x = 0
            hr_resize_y = 0
            hr_checkpoint_name = 'Use same checkpoint'
            hr_additional_modules = ['Use same choices']
            hr_sampler_name = 'Use same sampler'
            hr_scheduler = 'Use same scheduler'
            hr_prompt = ''
            hr_negative_prompt = ''
            hr_cfg = 7.0
            hr_distilled_cfg = 3.5
        override_settings_texts = []
        # Build script args from payload (ignore provided script_args)
        script_args_payload = modules.scripts.build_script_args_from_payload(modules.scripts.scripts_txt2img, payload)
        return (
            prompt,
            negative_prompt,
            prompt_styles,
            n_iter,
            batch_size,
            cfg_scale,
            distilled_cfg_scale,
            height,
            width,
            enable_hr,
            denoising_strength,
            hr_scale,
            hr_upscaler,
            hr_second_pass_steps,
            hr_resize_x,
            hr_resize_y,
            hr_checkpoint_name,
            hr_additional_modules,
            hr_sampler_name,
            hr_scheduler,
            hr_prompt,
            hr_negative_prompt,
            hr_cfg,
            hr_distilled_cfg,
            override_settings_texts,
            *script_args_payload,
        )

    # Build p
    _build()
    proc.batch_size = 1
    proc.n_iter = 1
    proc.txt2img_upscale = True

    image_info = gallery[gallery_index]
    proc.firstpass_image = infotext_utils.image_from_url_text(image_info)

    parameters = parse_generation_parameters(json.loads(generation_info).get('infotexts')[gallery_index], [])
    proc.seed = parameters.get('Seed', -1)
    proc.subseed = parameters.get('Variation seed', -1)

    # Update processing width/height to source image dims
    proc.width = gallery[gallery_index][0].size[0]
    proc.height = gallery[gallery_index][0].size[1]
    proc.extra_generation_params['Original Size'] = f"{proc.width}x{proc.height}"

    proc.override_settings['save_images_before_highres_fix'] = False

    with closing(proc):
        processed = modules.scripts.scripts_txt2img.run(proc, *proc.script_args)
        if processed is None:
            processed = processing.process_images(proc)

    shared.total_tqdm.clear()

    new_gallery = []
    geninfo = json.loads(generation_info)
    insert = getattr(shared.opts, 'hires_button_gallery_insert', False)
    for i, image in enumerate(gallery):
        if insert or i != gallery_index:
            image[0].already_saved_as = image[0].filename.rsplit('?', 1)[0]
            new_gallery.append(image)
        if i == gallery_index:
            new_gallery.extend(processed.images)
    new_index = gallery_index + (1 if insert else 0)
    if insert:
        geninfo["infotexts"].insert(new_index, processed.info)
    else:
        geninfo["infotexts"][gallery_index] = processed.info

    return new_gallery, json.dumps(geninfo), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def txt2img_create_processing(id_task: str, request: gr.Request, prompt: str, negative_prompt: str, prompt_styles, n_iter: int, batch_size: int, cfg_scale: float, distilled_cfg_scale: float, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_checkpoint_name: str, hr_additional_modules: list, hr_sampler_name: str, hr_scheduler: str, hr_prompt: str, hr_negative_prompt, hr_cfg: float, hr_distilled_cfg: float, override_settings_texts, *args, force_enable_hr=False):
    override_settings = create_override_settings_dict(override_settings_texts)

    if force_enable_hr:
        enable_hr = True

    proc = processing.StableDiffusionProcessingTxt2Img(
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
        n_iter=n_iter,
        cfg_scale=cfg_scale,
        distilled_cfg_scale=distilled_cfg_scale,
        width=width,
        height=height,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_checkpoint_name=None if hr_checkpoint_name == 'Use same checkpoint' else hr_checkpoint_name,
        hr_additional_modules=hr_additional_modules,
        hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
        hr_scheduler=None if hr_scheduler == 'Use same scheduler' else hr_scheduler,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        hr_cfg=hr_cfg,
        hr_distilled_cfg=hr_distilled_cfg,
        override_settings=override_settings,
    )

    proc.scripts = modules.scripts.scripts_txt2img
    proc.script_args = args

    proc.user = request.username

    if shared.opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    return proc


def txt2img_upscale_function(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
    assert len(gallery) > 0, 'No image to upscale'

    if gallery_index < 0 or gallery_index >= len(gallery):
        return gallery, generation_info, f'Bad image index: {gallery_index}', ''

    geninfo = json.loads(generation_info)

    #   catch situation where user tries to hires-fix the grid: probably a mistake, results can be bad aspect ratio - just don't do it
    first_image_index = geninfo.get('index_of_first_image', 0)
    #   catch if user tries to upscale a control image, this function will fail later trying to get infotext that doesn't exist
    count_images = len(geninfo.get('infotexts'))        #   note: we have batch_size in geninfo, but not batch_count
    if len(gallery) > 1 and (gallery_index < first_image_index or gallery_index >= count_images):
        return gallery, generation_info, 'Unable to upscale grid or control images.', ''

    proc = txt2img_create_processing(id_task, request, *args, force_enable_hr=True)
    proc.batch_size = 1
    proc.n_iter = 1
    # txt2img_upscale attribute that signifies this is called by txt2img_upscale
    proc.txt2img_upscale = True

    image_info = gallery[gallery_index]
    proc.firstpass_image = infotext_utils.image_from_url_text(image_info)

    parameters = parse_generation_parameters(geninfo.get('infotexts')[gallery_index], [])
    proc.seed = parameters.get('Seed', -1)
    proc.subseed = parameters.get('Variation seed', -1)

    #   update processing width/height based on actual dimensions of source image
    proc.width = gallery[gallery_index][0].size[0]
    proc.height = gallery[gallery_index][0].size[1]
    proc.extra_generation_params['Original Size'] = f'{args[8]}x{args[7]}'

    proc.override_settings['save_images_before_highres_fix'] = False

    with closing(proc):
        processed = modules.scripts.scripts_txt2img.run(proc, *proc.script_args)

        if processed is None:
            processed = processing.process_images(proc)

    shared.total_tqdm.clear()

    insert = getattr(shared.opts, 'hires_button_gallery_insert', False)
    new_gallery = []
    for i, image in enumerate(gallery):
        if insert or i != gallery_index:
            image[0].already_saved_as = image[0].filename.rsplit('?', 1)[0]
            new_gallery.append(image)
        if i == gallery_index:
            new_gallery.extend(processed.images)
        
    new_index = gallery_index
    if insert:
        new_index += 1
        geninfo["infotexts"].insert(new_index, processed.info)
    else:
        geninfo["infotexts"][gallery_index] = processed.info

    return new_gallery, json.dumps(geninfo), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def txt2img_function(id_task: str, request: gr.Request, *args):
    proc = txt2img_create_processing(id_task, request, *args)

    with closing(proc):
        processed = modules.scripts.scripts_txt2img.run(proc, *proc.script_args)

        if processed is None:
            processed = processing.process_images(proc)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images + processed.extra_images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def txt2img_upscale(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
    return main_thread.run_and_wait_result(txt2img_upscale_function, id_task, request, gallery, gallery_index, generation_info, *args)


def txt2img(id_task: str, request: gr.Request, *args):
    return main_thread.run_and_wait_result(txt2img_function, id_task, request, *args)

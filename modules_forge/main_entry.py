from __future__ import annotations

import os
import torch
import gradio as gr

from gradio.context import Context
from modules import shared, ui_common, sd_models, processing, infotext_utils, paths, sd_vae
from backend import memory_management, stream
from backend.args import dynamic_args


total_vram = int(memory_management.total_vram)

ui_forge_preset: gr.Radio | None = None

ui_checkpoint: gr.Dropdown | None = None
ui_vae: gr.Dropdown | None = None
ui_text_encoders: gr.Dropdown | None = None
ui_clip_skip: gr.Slider | None = None

ui_forge_unet_storage_dtype_options: gr.Dropdown | None = None
ui_forge_async_loading: gr.Radio | None = None
ui_forge_pin_shared_memory: gr.Radio | None = None
ui_forge_inference_memory: gr.Slider | None = None

forge_unet_storage_dtype_options = {
    'Automatic': (None, False),
    'Automatic (fp16 LoRA)': (None, True),
    'bnb-nf4': ('nf4', False),
    'bnb-nf4 (fp16 LoRA)': ('nf4', True),
    'float8-e4m3fn': (torch.float8_e4m3fn, False),
    'float8-e4m3fn (fp16 LoRA)': (torch.float8_e4m3fn, True),
    'bnb-fp4': ('fp4', False),
    'bnb-fp4 (fp16 LoRA)': ('fp4', True),
    'float8-e5m2': (torch.float8_e5m2, False),
    'float8-e5m2 (fp16 LoRA)': (torch.float8_e5m2, True),
}

module_list: dict[str, str] = {}
vae_module_list: dict[str, str] = {}
text_encoder_module_list: dict[str, str] = {}


def bind_to_opts(comp, k, save=False, callback=None):
    def on_change(v):
        shared.opts.set(k, v)
        if save:
            shared.opts.save(shared.config_filename)
        if callback is not None:
            callback()
        return

    comp.change(on_change, inputs=[comp], queue=False, show_progress=False)
    return


def _find_files_with_extensions(root: str, exts: tuple[str, ...]) -> dict[str, str]:
    result = {}
    if not isinstance(root, str) or not os.path.isdir(root):
        return result
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in exts:
                result[fname] = os.path.join(dirpath, fname)
    return result


def refresh_models():
    ckpt_list = sd_models.checkpoint_tiles(use_short=True)

    # Discover VAE/Text Encoder modules by scanning known paths
    exts = ('.safetensors', '.ckpt', '.pt')
    vae_dirs = [
        os.path.abspath(os.path.join(paths.models_path, "VAE")),
    ]
    text_encoder_dirs = [
        os.path.abspath(os.path.join(paths.models_path, "text_encoder")),
    ]
    if isinstance(shared.cmd_opts.vae_dir, str):
        vae_dirs.append(os.path.abspath(shared.cmd_opts.vae_dir))
    if isinstance(shared.cmd_opts.text_encoder_dir, str):
        text_encoder_dirs.append(os.path.abspath(shared.cmd_opts.text_encoder_dir))

    module_list.clear()
    vae_module_list.clear()
    text_encoder_module_list.clear()

    for p in vae_dirs:
        vae_module_list.update(_find_files_with_extensions(p, exts))
    for p in text_encoder_dirs:
        text_encoder_module_list.update(_find_files_with_extensions(p, exts))

    # Union for compatibility APIs
    module_list.clear()
    module_list.update(vae_module_list)
    module_list.update(text_encoder_module_list)

    return ckpt_list, sorted(vae_module_list.keys()), sorted(text_encoder_module_list.keys())


def _resolve_module_path(name: str | None) -> str | None:
    if not isinstance(name, str):
        return None
    if name in ('Automatic', 'Built in', 'None', ''):
        return None
    if name in module_list:
        return module_list[name]
    base = os.path.basename(name)
    if base in module_list:
        return module_list[base]
    if os.path.isfile(name):
        return os.path.abspath(name)
    return None


def make_checkpoint_manager_ui():
    global ui_checkpoint, ui_vae, ui_text_encoders, ui_clip_skip
    global ui_forge_unet_storage_dtype_options, ui_forge_async_loading, ui_forge_pin_shared_memory, ui_forge_inference_memory, ui_forge_preset

    if shared.opts.sd_model_checkpoint in [None, 'None', 'none', '']:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set('sd_model_checkpoint', next(iter(sd_models.checkpoints_list.values())).name)

    ui_forge_preset = gr.Radio(
        label="UI",
        value=lambda: getattr(shared.opts, 'forge_preset', 'all'),
        choices=['sd', 'xl', 'flux', 'all'],
        elem_id="forge_ui_preset",
        interactive=True,
    )

    ckpt_list, _vae_choices, text_encoder_choices = refresh_models()
    if not ckpt_list:
        ckpt_list = ["(no checkpoints found)"]

    def _compose_vae_choices():
        # Preserve compatibility aliases
        baseline = ['Automatic', 'Built in', 'None']
        dynamic = [c for c in sorted(vae_module_list.keys()) if c not in baseline]
        return baseline + dynamic

    def _current_vae_value():
        stored = getattr(shared.opts, 'forge_selected_vae', 'Automatic')
        if stored in (None, '', 'Automatic', 'Built in'):
            return 'Automatic'
        if stored == 'None':
            return 'None'
        base = os.path.basename(stored)
        return base if base in vae_module_list else 'Automatic'

    def _current_text_encoder_values():
        selected = []
        for path in getattr(shared.opts, 'forge_additional_modules', []) or []:
            base = os.path.basename(path)
            if base in text_encoder_module_list:
                selected.append(base)
        return selected

    def _migrate_legacy_selection():
        current_modules = getattr(shared.opts, 'forge_additional_modules', []) or []
        selected_vae = getattr(shared.opts, 'forge_selected_vae', 'Automatic')
        if selected_vae not in (None, '', 'Automatic', 'Built in'):
            return
        for path in list(current_modules):
            base = os.path.basename(path)
            if base in vae_module_list:
                shared.opts.set('forge_selected_vae', path)
                current_modules.remove(path)
                shared.opts.set('forge_additional_modules', current_modules)
                try:
                    shared.opts.save(shared.config_filename)
                except Exception:
                    pass
                break

    _migrate_legacy_selection()

    ui_checkpoint = gr.Dropdown(
        value=lambda: shared.opts.sd_model_checkpoint or "(no checkpoints found)",
        label="Checkpoint",
        elem_classes=['model_selection'],
        choices=ckpt_list
    )

    ui_vae = gr.Dropdown(
        value=_current_vae_value,
        label="VAE",
        elem_classes=['model_selection'],
        choices=_compose_vae_choices()
    )

    ui_text_encoders = gr.Dropdown(
        value=_current_text_encoder_values,
        multiselect=True,
        label="Text Encoder(s)",
        render=False,
        choices=sorted(text_encoder_module_list.keys()),
        elem_classes=['model_selection'],
    )

    def gr_refresh():
        ckpts, vaes, text_encs = refresh_models()
        return (
            gr.update(choices=ckpts),
            gr.update(choices=_compose_vae_choices(), value=_current_vae_value()),
            gr.update(choices=text_encs, value=_current_text_encoder_values()),
        )

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id=f"forge_refresh_checkpoint", tooltip="Refresh")
    refresh_button.click(
        fn=gr_refresh,
        inputs=[],
        outputs=[ui_checkpoint, ui_vae, ui_text_encoders],
        show_progress=False,
        queue=False,
    )
    Context.root_block.load(
        fn=gr_refresh,
        inputs=[],
        outputs=[ui_checkpoint, ui_vae, ui_text_encoders],
        show_progress=False,
        queue=False,
    )

    ui_text_encoders.render()

    ui_forge_unet_storage_dtype_options = gr.Dropdown(label="Diffusion in Low Bits", value=lambda: getattr(shared.opts, 'forge_unet_storage_dtype', 'Automatic'), choices=list(forge_unet_storage_dtype_options.keys()))
    bind_to_opts(ui_forge_unet_storage_dtype_options, 'forge_unet_storage_dtype', save=True, callback=refresh_model_loading_parameters)

    ui_forge_async_loading = gr.Radio(label="Swap Method", value=lambda: getattr(shared.opts, 'forge_async_loading', 'Queue'), choices=['Queue', 'Async'])
    ui_forge_pin_shared_memory = gr.Radio(label="Swap Location", value=lambda: getattr(shared.opts, 'forge_pin_shared_memory', 'CPU'), choices=['CPU', 'Shared'])
    ui_forge_inference_memory = gr.Slider(label="GPU Weights (MB)", value=lambda: total_vram - getattr(shared.opts, 'forge_inference_memory', total_vram-1024), minimum=0, maximum=int(memory_management.total_vram), step=1)

    mem_comps = [ui_forge_inference_memory, ui_forge_async_loading, ui_forge_pin_shared_memory]
    ui_forge_inference_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_async_loading.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_pin_shared_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    Context.root_block.load(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)

    ui_clip_skip = gr.Slider(label="Clip skip", value=lambda: getattr(shared.opts, 'CLIP_stop_at_last_layers', 1), **{"minimum": 1, "maximum": 12, "step": 1})
    bind_to_opts(ui_clip_skip, 'CLIP_stop_at_last_layers', save=True)

    ui_checkpoint.change(checkpoint_change, inputs=[ui_checkpoint], show_progress=False)
    ui_vae.change(vae_change, inputs=[ui_vae], queue=False, show_progress=False)
    ui_text_encoders.change(modules_change, inputs=[ui_text_encoders], queue=False, show_progress=False)

    # Ensure initial model loading params exist to avoid KeyError during first gen
    refresh_model_loading_parameters()
    return


def ui_refresh_memory_management_settings(model_memory, async_loading, pin_shared_memory):
    refresh_memory_management_settings(async_loading=async_loading, model_memory=model_memory, pin_shared_memory=pin_shared_memory)


def refresh_memory_management_settings(async_loading=None, inference_memory=None, pin_shared_memory=None, model_memory=None):
    async_loading = async_loading if async_loading is not None else getattr(shared.opts, 'forge_async_loading', 'Queue')
    inference_memory = inference_memory if inference_memory is not None else getattr(shared.opts, 'forge_inference_memory', total_vram-1024)
    pin_shared_memory = pin_shared_memory if pin_shared_memory is not None else getattr(shared.opts, 'forge_pin_shared_memory', 'CPU')

    if model_memory is None:
        model_memory = total_vram - inference_memory
    else:
        inference_memory = total_vram - model_memory

    shared.opts.set('forge_async_loading', async_loading)
    shared.opts.set('forge_inference_memory', inference_memory)
    shared.opts.set('forge_pin_shared_memory', pin_shared_memory)

    stream.stream_activated = async_loading == 'Async'
    memory_management.current_inference_memory = inference_memory * 1024 * 1024
    memory_management.PIN_SHARED_MEMORY = pin_shared_memory == 'Shared'

    processing.need_global_unload = True
    return


def refresh_model_loading_parameters():
    from modules.sd_models import select_checkpoint, model_data

    checkpoint_info = select_checkpoint()
    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(getattr(shared.opts, 'forge_unet_storage_dtype', 'Automatic'), (None, False))
    dynamic_args['online_lora'] = lora_fp16

    selected_vae = getattr(shared.opts, 'forge_selected_vae', 'Automatic')
    additional_modules: list[str] = []
    if isinstance(selected_vae, str) and selected_vae not in ('Automatic', 'Built in', 'None', ''):
        additional_modules.append(selected_vae)

    for module in getattr(shared.opts, 'forge_additional_modules', []) or []:
        if isinstance(module, str) and module:
            additional_modules.append(module)

    model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=additional_modules,
        unet_storage_dtype=unet_storage_dtype,
    )
    processing.need_global_unload = True
    return


def checkpoint_change(ckpt_name: str, save=True, refresh=True):
    new_ckpt = sd_models.get_closet_checkpoint_match(ckpt_name)
    cur_ckpt = sd_models.get_closet_checkpoint_match(getattr(shared.opts, 'sd_model_checkpoint', ''))
    if new_ckpt == cur_ckpt:
        return False
    shared.opts.set('sd_model_checkpoint', ckpt_name)
    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def vae_change(vae_value, save=True, refresh=True) -> bool:
    selected = vae_value
    if isinstance(vae_value, list):
        selected = vae_value[0] if vae_value else 'Automatic'

    if selected in (None, '', 'Automatic', 'Built in'):
        target = 'Automatic'
    elif selected == 'None':
        target = 'None'
    else:
        resolved = _resolve_module_path(selected)
        target = resolved if resolved is not None else 'Automatic'

    if target == getattr(shared.opts, 'forge_selected_vae', 'Automatic'):
        return False

    shared.opts.set('forge_selected_vae', target)
    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def modules_change(module_values: list, save=True, refresh=True) -> bool:
    selected_vae_override: str | None = None
    resolved_modules: list[str] = []
    for value in module_values or []:
        if isinstance(value, str) and value in ('Use same choices', 'Use same checkpoint'):
            continue
        path = _resolve_module_path(value)
        if path is not None:
            base = os.path.basename(path)
            if base in vae_module_list:
                if selected_vae_override is None:
                    selected_vae_override = path
                continue
            resolved_modules.append(path)
    # Deduplicate while preserving order
    modules = []
    seen = set()
    for path in resolved_modules:
        if path not in seen:
            modules.append(path)
            seen.add(path)

    current = list(getattr(shared.opts, 'forge_additional_modules', []))
    changed = False

    if selected_vae_override is not None:
        changed = vae_change(selected_vae_override, save=save, refresh=False) or changed

    if sorted(modules) != sorted(current):
        shared.opts.set('forge_additional_modules', modules)
        if save:
            shared.opts.save(shared.config_filename)
        changed = True

    if not changed:
        return False

    if refresh:
        refresh_model_loading_parameters()
    return True
    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def get_a1111_ui_component(tab, label):
    fields = infotext_utils.paste_fields[tab]['fields']
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    # Wire preset changes to downstream defaults and persist option
    try:
        ui_txt2img_width = get_a1111_ui_component('txt2img', 'Size-1')
        ui_txt2img_height = get_a1111_ui_component('txt2img', 'Size-2')
        ui_txt2img_cfg = get_a1111_ui_component('txt2img', 'CFG scale')
        ui_txt2img_distilled_cfg = get_a1111_ui_component('txt2img', 'Distilled CFG Scale')
        ui_txt2img_hr_distilled_cfg = get_a1111_ui_component('txt2img', 'Hires Distilled CFG Scale')
        ui_txt2img_sampler = get_a1111_ui_component('txt2img', 'sampler_name')
        ui_txt2img_scheduler = get_a1111_ui_component('txt2img', 'scheduler')

        ui_img2img_width = get_a1111_ui_component('img2img', 'Size-1')
        ui_img2img_height = get_a1111_ui_component('img2img', 'Size-2')
        ui_img2img_cfg = get_a1111_ui_component('img2img', 'CFG scale')
        ui_img2img_distilled_cfg = get_a1111_ui_component('img2img', 'Distilled CFG Scale')
        ui_img2img_sampler = get_a1111_ui_component('img2img', 'sampler_name')
        ui_img2img_scheduler = get_a1111_ui_component('img2img', 'scheduler')

        outputs = [
            ui_vae,
            ui_clip_skip,
            ui_forge_unet_storage_dtype_options,
            ui_forge_async_loading,
            ui_forge_pin_shared_memory,
            ui_forge_inference_memory,
            ui_txt2img_width,
            ui_img2img_width,
            ui_txt2img_height,
            ui_img2img_height,
            ui_txt2img_cfg,
            ui_img2img_cfg,
            ui_txt2img_distilled_cfg,
            ui_txt2img_hr_distilled_cfg,
            ui_img2img_distilled_cfg,
            ui_txt2img_sampler,
            ui_img2img_sampler,
            ui_txt2img_scheduler,
            ui_img2img_scheduler,
        ]
        if ui_forge_preset is not None:
            def on_preset_change(v):
                shared.opts.set('forge_preset', v)
                try:
                    shared.opts.save(shared.config_filename)
                except Exception:
                    pass

                # Choose per-preset defaults
                if v == 'sd':
                    t2i_wh = (512, 512)
                    i2i_wh = (512, 512)
                    t2i_cfg, t2i_dcfg = (7.0, 3.5)
                    i2i_cfg, i2i_dcfg = (7.0, 3.5)
                    t2i_sampler = getattr(shared.opts, 'sd_t2i_sampler', 'Euler a')
                    t2i_scheduler = getattr(shared.opts, 'sd_t2i_scheduler', 'Automatic')
                    i2i_sampler = getattr(shared.opts, 'sd_i2i_sampler', 'Euler a')
                    i2i_scheduler = getattr(shared.opts, 'sd_i2i_scheduler', 'Automatic')
                elif v == 'xl':
                    t2i_wh = (1024, 1024)
                    i2i_wh = (1024, 1024)
                    t2i_cfg, t2i_dcfg = (7.0, 3.5)
                    i2i_cfg, i2i_dcfg = (7.0, 3.5)
                    t2i_sampler = getattr(shared.opts, 'xl_t2i_sampler', 'DPM++ 2M SDE')
                    t2i_scheduler = getattr(shared.opts, 'xl_t2i_scheduler', 'Karras')
                    i2i_sampler = getattr(shared.opts, 'xl_i2i_sampler', 'DPM++ 2M SDE')
                    i2i_scheduler = getattr(shared.opts, 'xl_i2i_scheduler', 'Karras')
                elif v == 'flux':
                    t2i_wh = (896, 1152)
                    i2i_wh = (1024, 1024)
                    t2i_cfg, t2i_dcfg = (1.0, 3.5)
                    i2i_cfg, i2i_dcfg = (1.0, 3.5)
                    t2i_sampler = getattr(shared.opts, 'flux_t2i_sampler', 'Euler')
                    t2i_scheduler = getattr(shared.opts, 'flux_t2i_scheduler', 'Simple')
                    i2i_sampler = getattr(shared.opts, 'flux_i2i_sampler', 'Euler')
                    i2i_scheduler = getattr(shared.opts, 'flux_i2i_scheduler', 'Simple')
                else:
                    # 'all' preset leaves values unchanged
                    return [gr.update() for _ in outputs]

                (t2i_w, t2i_h) = t2i_wh
                (i2i_w, i2i_h) = i2i_wh
                flux_interactive = (v == 'flux')
                return [
                    gr.update(),  # ui_vae
                    gr.update(),  # ui_clip_skip
                    gr.update(),  # ui_forge_unet_storage_dtype_options
                    gr.update(),  # ui_forge_async_loading
                    gr.update(),  # ui_forge_pin_shared_memory
                    gr.update(),  # ui_forge_inference_memory
                    gr.update(value=t2i_w),   # ui_txt2img_width
                    gr.update(value=i2i_w),   # ui_img2img_width
                    gr.update(value=t2i_h),   # ui_txt2img_height
                    gr.update(value=i2i_h),   # ui_img2img_height
                    gr.update(value=t2i_cfg), # ui_txt2img_cfg
                    gr.update(value=i2i_cfg), # ui_img2img_cfg
                    gr.update(value=t2i_dcfg, interactive=flux_interactive), # ui_txt2img_distilled_cfg
                    gr.update(interactive=flux_interactive), # ui_txt2img_hr_distilled_cfg
                    gr.update(value=i2i_dcfg, interactive=flux_interactive), # ui_img2img_distilled_cfg
                    gr.update(value=t2i_sampler),    # ui_txt2img_sampler
                    gr.update(value=i2i_sampler),    # ui_img2img_sampler
                    gr.update(value=t2i_scheduler),  # ui_txt2img_scheduler
                    gr.update(value=i2i_scheduler),  # ui_img2img_scheduler
                ]
            ui_forge_preset.change(on_preset_change, inputs=[ui_forge_preset], outputs=outputs, queue=False, show_progress=False)
    except Exception:
        pass
    refresh_model_loading_parameters()
    return

import os

import torch

from modules import shared
from modules.shared import cmd_opts


def initialize():
    """Initializes fields inside the shared module in a controlled manner.

    Should be called early because some other modules you can import mingt need these fields to be already set.
    """

    os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)

    from modules import options, shared_options
    shared.options_templates = shared_options.options_templates
    shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
    shared.restricted_opts = shared_options.restricted_opts
    try:
        shared.opts.load(shared.config_filename)
    except FileNotFoundError:
        pass

    # Clamp critical numeric options to safe ranges before UI builds
    sanitize_options(shared.opts)

    # Clamp critical numeric options to safe ranges before UI builds
    sanitize_options(shared.opts)

    from modules import devices
    shared.device = devices.device
    shared.weight_load_location = None if cmd_opts.lowram else "cpu"

    from modules import shared_state
    shared.state = shared_state.State()

    from modules import styles
    shared.prompt_styles = styles.StyleDatabase(shared.styles_filename)

    from modules import interrogate
    shared.interrogator = interrogate.InterrogateModels("interrogate")

    from modules import shared_total_tqdm
    shared.total_tqdm = shared_total_tqdm.TotalTQDM()

    from modules import memmon, devices
    shared.mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, shared.opts)
    shared.mem_mon.start()


def sanitize_options(opts):
    """Ensure persisted options remain within component constraints."""

    clamp_int = lambda key, lo, hi: _clamp_option(opts, key, lo, hi, cast=int)
    clamp_float = lambda key, lo, hi: _clamp_option(opts, key, lo, hi, cast=float)

    clamp_int('img2img_batch_size', 1, 64)
    clamp_int('img2img_batch_count', 1, 256)
    clamp_int('samples_save', 0, 1)

    clamp_int('sd_batch_count', 1, 256)
    clamp_int('sd_batch_size', 1, 64)
    clamp_int('CLIP_stop_at_last_layers', 1, 12)

    clamp_int('sd_sampling_steps', 1, 150)
    clamp_float('eta_noise_seed_delta', 0, 32)

    clamp_int('SD_upscale_override_upscale', 64, 2048)
    clamp_int('Hires_upscale_minimal', 64, 2048)

    clamp_int('img2img_background_color', 0, 0xFFFFFF)
    clamp_float('realesrgan_visibility', 0.0, 1.0)

    clamp_int('img2img_width', 64, 2048)
    clamp_int('img2img_height', 64, 2048)
    clamp_int('sd_width', 64, 2048)
    clamp_int('sd_height', 64, 2048)

    clamp_int('firstpass_width', 0, 4096)
    clamp_int('firstpass_height', 0, 4096)
    clamp_int('Hires_resize_x', 0, 4096)
    clamp_int('Hires_resize_y', 0, 4096)
    clamp_float('Hires_upscale', 1.0, 4.0)


def _clamp_option(opts, key, lo, hi, cast=int):
    if key not in opts.data:
        return
    try:
        val = cast(opts.data[key])
    except Exception:
        val = cast(shared.opts.data_labels[key].default)
    if lo is not None:
        val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    opts.data[key] = val

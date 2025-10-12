from __future__ import annotations

import gradio as gr
from modules import sd_models, sd_vae


# Minimal checkpoint/VAE selector row to replace Forge's main_entry wiring
ui_checkpoint = gr.Dropdown(choices=sd_models.checkpoint_tiles(use_short=True), label="Checkpoint", elem_id="ui_checkpoint")
ui_vae = gr.Dropdown(choices=["Built in"] + list(sd_vae.vae_dict), label="VAE", elem_id="ui_vae")

# Stub module registry for compatibility with callers that expect Forge modules
module_list: dict[str, str] = {}


def refresh_models():
    """Compatibility shim: return (state, modules) like Forge.

    Returns a tuple (changed_state, modules_iterable). We don't manage extra modules,
    so this returns (None, []).
    """
    return None, []


def make_checkpoint_manager_ui():
    with gr.Row():
        global ui_checkpoint, ui_vae
        ui_checkpoint.render()
        ui_vae.render()


def forge_main_entry():
    # No-op shim; kept for compatibility with callers expecting this function
    pass

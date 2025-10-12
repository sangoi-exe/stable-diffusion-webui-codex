from __future__ import annotations

import gradio as gr
from modules import sd_models, sd_vae, shared


# Minimal checkpoint/VAE selector row to replace Forge's main_entry wiring
ui_checkpoint = gr.Dropdown(choices=sd_models.checkpoint_tiles(use_short=True), label="Checkpoint", elem_id="ui_checkpoint")
ui_vae = gr.Dropdown(choices=["Built in"] + list(sd_vae.vae_dict), label="VAE", elem_id="ui_vae")


def make_checkpoint_manager_ui():
    with gr.Row():
        global ui_checkpoint, ui_vae
        ui_checkpoint.render()
        ui_vae.render()


def forge_main_entry():
    # No-op shim; kept for compatibility with callers expecting this function
    pass


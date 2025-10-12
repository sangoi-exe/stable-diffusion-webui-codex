from __future__ import annotations

import gradio as gr


class NullCanvas:
    def __init__(
        self,
        no_upload=False,
        no_scribbles=False,
        contrast_scribbles=False,
        height=512,
        scribble_color='#000000',
        scribble_color_fixed=False,
        scribble_width=4,
        scribble_width_fixed=False,
        scribble_alpha=100,
        scribble_alpha_fixed=False,
        scribble_softness=0,
        scribble_softness_fixed=False,
        visible=True,
        numpy=False,
        initial_image=None,
        elem_id=None,
        elem_classes=None,
    ):
        # Minimal UI: single image input behaves as background; no foreground layer
        self.background = gr.Image(
            label="Image", show_label=False, source="upload", interactive=True, type="pil", image_mode="RGBA",
            elem_id=elem_id or "img_background", height=height
        )
        # keep attribute for compatibility, but unused
        self.foreground = gr.Image(visible=False, label="Foreground", type="pil", image_mode="RGBA")
ForgeCanvas = NullCanvas  # type: ignore
canvas_head = ''

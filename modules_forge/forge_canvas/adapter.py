from __future__ import annotations

try:
    from .canvas import ForgeCanvas as RealForgeCanvas, canvas_head as real_canvas_head
except Exception:
    RealForgeCanvas, real_canvas_head = None, ''

import gradio as gr
from modules.shared import opts


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


def _enabled():
    try:
        return bool(getattr(opts, 'forge_canvas_enable', False)) and RealForgeCanvas is not None
    except Exception:
        return False


if _enabled():
    ForgeCanvas = RealForgeCanvas  # type: ignore
    canvas_head = real_canvas_head
else:
    ForgeCanvas = NullCanvas  # type: ignore
    canvas_head = ''


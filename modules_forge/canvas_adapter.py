from __future__ import annotations

import gradio as gr


class GradioCanvas:
    """Minimal replacement for ForgeCanvas using built-in Gradio components.

    Exposes a compatible interface: `.background`, `.foreground`, and `.block`.
    - `.background`: primary image input/editor (ImageEditor if available, else Image)
    - `.foreground`: hidden placeholder to satisfy call sites expecting it
    - `.block`: parent container/block to attach callbacks if needed
    """

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
        with gr.Column(visible=visible) as container:
            # Prefer the built-in ImageEditor (Gradio 5) for drawing/overlays
            try:
                editor = gr.ImageEditor(
                    label=None,
                    show_label=False,
                    height=height,
                    elem_id=elem_id or "img_background",
                    elem_classes=elem_classes,
                )
                # Hidden processed background (PIL) derived from editor's composite
                processed_bg = gr.Image(
                    visible=False, label="Background", type="pil", image_mode="RGBA"
                )
                # When user edits/applies, map EditorValue -> composite image
                editor.change(lambda ev: (ev or {}).get('composite') if isinstance(ev, dict) else None,
                              inputs=editor, outputs=processed_bg)
                editor.input(lambda ev: (ev or {}).get('composite') if isinstance(ev, dict) else None,
                             inputs=editor, outputs=processed_bg)
                # Expose processed image as background for downstream pipeline
                self.background = processed_bg
                self.editor = editor
            except Exception:
                self.background = gr.Image(
                    label=None, show_label=False, source="upload", interactive=True, type="pil", image_mode="RGBA",
                    elem_id=elem_id or "img_background", height=height, elem_classes=elem_classes
                )
                self.editor = self.background

            # Hidden placeholder to keep API parity with sites expecting a foreground image
            self.foreground = gr.Image(visible=False, label="Foreground", type="pil", image_mode="RGBA")

        # Expose the parent block as `.block` for extensions that attach callbacks
        self.block = container


ForgeCanvas = GradioCanvas
canvas_head = ''

from functools import wraps

import gradio as gr
from modules import gradio_extensions  # noqa: F401


class FormComponent:
    webui_do_not_create_gradio_pyi_thank_you = True

    def get_expected_parent(self):
        return gr.components.Form


gr.Dropdown.get_expected_parent = FormComponent.get_expected_parent


class ToolButton(gr.Button, FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    @wraps(gr.Button.__init__)
    def __init__(self, value="", *args, elem_classes=None, **kwargs):
        elem_classes = elem_classes or []
        super().__init__(*args, elem_classes=["tool", *elem_classes], value=value, **kwargs)

    def get_block_name(self):
        return "button"


class ResizeHandleRow(gr.Row):
    """Same as gr.Row but fits inside gradio forms"""
    webui_do_not_create_gradio_pyi_thank_you = True

    @wraps(gr.Row.__init__)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.elem_classes.append("resize-handle-row")

    def get_block_name(self):
        return "row"


class FormRow(gr.Row, FormComponent):
    """Same as gr.Row but fits inside gradio forms"""

    def get_block_name(self):
        return "row"


class FormColumn(gr.Column, FormComponent):
    """Same as gr.Column but fits inside gradio forms"""

    def get_block_name(self):
        return "column"


class FormGroup(gr.Group, FormComponent):
    """Same as gr.Group but fits inside gradio forms"""

    def get_block_name(self):
        return "group"


class FormHTML(gr.HTML, FormComponent):
    """Same as gr.HTML but fits inside gradio forms"""

    def get_block_name(self):
        return "html"


class FormColorPicker(gr.ColorPicker, FormComponent):
    """Same as gr.ColorPicker but fits inside gradio forms"""

    def get_block_name(self):
        return "colorpicker"


class DropdownMulti(gr.Dropdown, FormComponent):
    """Same as gr.Dropdown but always multiselect"""

    @wraps(gr.Dropdown.__init__)
    def __init__(self, **kwargs):
        kwargs['multiselect'] = True
        super().__init__(**kwargs)

    def get_block_name(self):
        return "dropdown"


class DropdownEditable(gr.Dropdown, FormComponent):
    """Same as gr.Dropdown but allows editing value"""

    @wraps(gr.Dropdown.__init__)
    def __init__(self, **kwargs):
        kwargs['allow_custom_value'] = True
        super().__init__(**kwargs)

    def get_block_name(self):
        return "dropdown"


class InputAccordionImpl(gr.Checkbox):
    """An input-driven Accordion that returns True if open, False if closed.

    Implementation notes for Gradio 5:
    - Avoid any DOM-coupled JS. We bind the hidden Checkbox directly to the Accordion's `open` via Python updates.
    - Keeps the same public surface: `.accordion` attribute and `.extra()` helper.
    """

    webui_do_not_create_gradio_pyi_thank_you = True

    global_index = 0

    @wraps(gr.Checkbox.__init__)
    def __init__(self, value=None, setup=False, **kwargs):
        if not setup:
            super().__init__(value=value, **kwargs)
            return

        self.accordion_id = kwargs.get('elem_id')
        if self.accordion_id is None:
            self.accordion_id = f"input-accordion-{InputAccordionImpl.global_index}"
            InputAccordionImpl.global_index += 1

        # Build hidden checkbox first (input carrier)
        kwargs_checkbox = {
            **kwargs,
            "elem_id": f"{self.accordion_id}-checkbox",
            "visible": False,
            "label": None,
        }
        super().__init__(value=value, **kwargs_checkbox)

        # Build the visible accordion
        kwargs_accordion = {
            **kwargs,
            "elem_id": self.accordion_id,
            "label": kwargs.get('label', 'Accordion'),
            "elem_classes": ['input-accordion'],
            "open": value,
        }
        self.accordion = gr.Accordion(**kwargs_accordion)

        # Keep accordion open state in sync with checkbox value
        # Pure-Python update; no JS dependency.
        self.change(
            fn=lambda checked: gr.Accordion.update(open=bool(checked)),
            inputs=[self],
            outputs=[self.accordion],
            show_progress=False,
            queue=False,
        )

    def extra(self):
        """Place content into the label area of the accordion."""
        return gr.Column(elem_id=self.accordion_id + '-extra', elem_classes='input-accordion-extra', min_width=0)

    def __enter__(self):
        self.accordion.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accordion.__exit__(exc_type, exc_val, exc_tb)

    def get_block_name(self):
        return "checkbox"


def InputAccordion(value=None, **kwargs):
    return InputAccordionImpl(value=value, setup=True, **kwargs)

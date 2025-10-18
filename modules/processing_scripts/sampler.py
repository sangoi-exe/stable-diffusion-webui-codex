import gradio as gr

from backend.diffusion_engine.flux_config import FLUX_DROPDOWN_KEYS, build_flux_option_info
from backend.core.engine_interface import TaskType
from backend.core.sampler_policy import allowed_samplers, allowed_schedulers
from modules import scripts, sd_samplers, sd_schedulers, shared
from modules.infotext_utils import PasteField
from modules.ui_components import FormRow, FormGroup


class ScriptSampler(scripts.ScriptBuiltinUI):
    section = "sampler"

    def __init__(self):
        self.steps = None
        self.sampler_name = None
        self.scheduler = None

    def title(self):
        return "Sampler"

    def ui(self, is_img2img):
        engine_key = getattr(shared.opts, 'codex_engine', 'sd15')
        task = TaskType.IMG2IMG if is_img2img else TaskType.TXT2IMG
        # Use backend policy for UI lists; fall back to sd_* if needed
        sampler_names = allowed_samplers(str(engine_key), task)
        scheduler_names = allowed_schedulers(str(engine_key), task)

        preset = getattr(shared.opts, 'forge_preset', 'all')
        def _defaults(tab: str):
            if preset == 'xl':
                return (
                    getattr(shared.opts, 'xl_t2i_sampler' if tab == 'txt2img' else 'xl_i2i_sampler', sampler_names[0]),
                    getattr(shared.opts, 'xl_t2i_scheduler' if tab == 'txt2img' else 'xl_i2i_scheduler', scheduler_names[0]),
                )
            if preset == 'flux':
                return (
                    getattr(shared.opts, 'flux_t2i_sampler' if tab == 'txt2img' else 'flux_i2i_sampler', sampler_names[0]),
                    getattr(shared.opts, 'flux_t2i_scheduler' if tab == 'txt2img' else 'flux_i2i_scheduler', scheduler_names[0]),
                )
            # default to 'sd'
            return (
                getattr(shared.opts, 'sd_t2i_sampler' if tab == 'txt2img' else 'sd_i2i_sampler', sampler_names[0]),
                getattr(shared.opts, 'sd_t2i_scheduler' if tab == 'txt2img' else 'sd_i2i_scheduler', scheduler_names[0]),
            )

        default_sampler, default_scheduler = _defaults(self.tabname)

        if shared.opts.samplers_in_dropdown:
            with FormRow(elem_id=f"sampler_selection_{self.tabname}"):
                self.sampler_name = gr.Dropdown(label='Sampling method', elem_id=f"{self.tabname}_sampling", choices=sampler_names, value=default_sampler)
                self.scheduler = gr.Dropdown(label='Schedule type', elem_id=f"{self.tabname}_scheduler", choices=scheduler_names, value=default_scheduler)
                self.steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{self.tabname}_steps", label="Sampling steps", value=20)
        else:
            with FormGroup(elem_id=f"sampler_selection_{self.tabname}"):
                self.steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{self.tabname}_steps", label="Sampling steps", value=20)
                self.sampler_name = gr.Radio(label='Sampling method', elem_id=f"{self.tabname}_sampling", choices=sampler_names, value=default_sampler)
                self.scheduler = gr.Dropdown(label='Schedule type', elem_id=f"{self.tabname}_scheduler", choices=scheduler_names, value=default_scheduler)

        self.infotext_fields = [
            PasteField(self.steps, "Steps", api="steps"),
            PasteField(self.sampler_name, sd_samplers.get_sampler_from_infotext, api="sampler_name"),
            PasteField(self.scheduler, sd_samplers.get_scheduler_from_infotext, api="scheduler"),
        ]

        shared.options_templates.update(shared.options_section(('ui_sd', "UI defaults 'sd'", "ui"), {
            "sd_t2i_sampler":     shared.OptionInfo('Euler a',      "txt2img sampler",      gr.Dropdown, {"choices": sampler_names}),
            "sd_t2i_scheduler":   shared.OptionInfo('Automatic',    "txt2img scheduler",    gr.Dropdown, {"choices": scheduler_names}),
            "sd_i2i_sampler":     shared.OptionInfo('Euler a',      "img2img sampler",      gr.Dropdown, {"choices": sampler_names}),
            "sd_i2i_scheduler":   shared.OptionInfo('Automatic',    "img2img scheduler",    gr.Dropdown, {"choices": scheduler_names}),
        }))
        shared.options_templates.update(shared.options_section(('ui_xl', "UI defaults 'xl'", "ui"), {
            "xl_t2i_sampler":     shared.OptionInfo('DPM++ 2M SDE', "txt2img sampler",      gr.Dropdown, {"choices": sampler_names}),
            "xl_t2i_scheduler":   shared.OptionInfo('Karras',       "txt2img scheduler",    gr.Dropdown, {"choices": scheduler_names}),
            "xl_i2i_sampler":     shared.OptionInfo('DPM++ 2M SDE', "img2img sampler",      gr.Dropdown, {"choices": sampler_names}),
            "xl_i2i_scheduler":   shared.OptionInfo('Karras',       "img2img scheduler",    gr.Dropdown, {"choices": scheduler_names}),
        }))
        shared.options_templates.update(
            shared.options_section(
                ('ui_flux', "UI defaults 'flux'", "ui"),
                build_flux_option_info(
                    shared.OptionInfo,
                    gr,
                    keys=FLUX_DROPDOWN_KEYS,
                    dynamic_choices={
                        "flux_t2i_sampler": sampler_names,
                        "flux_i2i_sampler": sampler_names,
                        "flux_t2i_scheduler": scheduler_names,
                        "flux_i2i_scheduler": scheduler_names,
                    },
                ),
            )
        )

        # Bind engine change to filter choices
        try:
            from modules_forge.main_entry import ui_codex_engine as _engine_dd

            def _update(engine: str):
                ek = engine or getattr(shared.opts, 'codex_engine', 'sd15')
                sn = allowed_samplers(str(ek), task)
                sc = allowed_schedulers(str(ek), task)
                return (
                    gr.update(choices=sn, value=sn[0] if sn else None),
                    gr.update(choices=sc, value=sc[0] if sc else None),
                )

            if _engine_dd is not None and self.sampler_name is not None and self.scheduler is not None:
                _engine_dd.change(_update, inputs=[_engine_dd], outputs=[self.sampler_name, self.scheduler], queue=False, show_progress=False)
        except Exception:
            # If engine dropdown is not available here, UI will still honor policy at render time.
            pass

        return self.steps, self.sampler_name, self.scheduler

    def setup(self, p, steps, sampler_name, scheduler):
        p.steps = steps
        p.sampler_name = sampler_name
        p.scheduler = scheduler

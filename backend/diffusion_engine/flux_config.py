"""Flux configuration schema shared between UI presets and backend engines.

This module centralises metadata about Flux-specific UI toggles so both the
frontend and backend can reference a single source of truth. The schema keeps
track of default values, control types, and slider/dropdown ranges. Consumers
can turn the metadata into Gradio ``OptionInfo`` objects to wire presets or
validate runtime payloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence


@dataclass(frozen=True)
class FluxFieldSpec:
    """Describes a Flux configuration toggle exposed in the UI."""

    key: str
    label: str
    control: str
    scope: str
    default: Optional[Any] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[Sequence[Any]] = None
    default_factory: Optional[Callable[[Mapping[str, Any]], Any]] = None
    minimum_factory: Optional[Callable[[Mapping[str, Any]], float]] = None
    maximum_factory: Optional[Callable[[Mapping[str, Any]], float]] = None
    notes: Optional[str] = None

    def resolve_default(self, context: Mapping[str, Any]) -> Any:
        if self.default_factory is not None:
            return self.default_factory(context)
        return self.default


def _total_vram_default(context: Mapping[str, Any]) -> int:
    total_vram = context.get("total_vram")
    if total_vram is None:
        raise ValueError("total_vram is required to compute the Flux GPU weight default")
    return max(0, total_vram - 1024)


FLUX_FIELD_SPECS: Sequence[FluxFieldSpec] = (
    FluxFieldSpec(
        key="flux_t2i_width",
        label="txt2img width",
        control="slider",
        scope="txt2img",
        default=896,
        minimum=64,
        maximum=2048,
        step=8,
    ),
    FluxFieldSpec(
        key="flux_t2i_height",
        label="txt2img height",
        control="slider",
        scope="txt2img",
        default=1152,
        minimum=64,
        maximum=2048,
        step=8,
    ),
    FluxFieldSpec(
        key="flux_t2i_cfg",
        label="txt2img CFG",
        control="slider",
        scope="txt2img",
        default=1.0,
        minimum=1.0,
        maximum=30.0,
        step=0.1,
    ),
    FluxFieldSpec(
        key="flux_t2i_hr_cfg",
        label="txt2img HiRes CFG",
        control="slider",
        scope="txt2img",
        default=1.0,
        minimum=1.0,
        maximum=30.0,
        step=0.1,
    ),
    FluxFieldSpec(
        key="flux_t2i_d_cfg",
        label="txt2img Distilled CFG",
        control="slider",
        scope="txt2img",
        default=3.5,
        minimum=0.0,
        maximum=30.0,
        step=0.1,
    ),
    FluxFieldSpec(
        key="flux_t2i_hr_d_cfg",
        label="txt2img Distilled HiRes CFG",
        control="slider",
        scope="txt2img",
        default=3.5,
        minimum=0.0,
        maximum=30.0,
        step=0.1,
    ),
    FluxFieldSpec(
        key="flux_i2i_width",
        label="img2img width",
        control="slider",
        scope="img2img",
        default=1024,
        minimum=64,
        maximum=2048,
        step=8,
    ),
    FluxFieldSpec(
        key="flux_i2i_height",
        label="img2img height",
        control="slider",
        scope="img2img",
        default=1024,
        minimum=64,
        maximum=2048,
        step=8,
    ),
    FluxFieldSpec(
        key="flux_i2i_cfg",
        label="img2img CFG",
        control="slider",
        scope="img2img",
        default=1.0,
        minimum=1.0,
        maximum=30.0,
        step=0.1,
    ),
    FluxFieldSpec(
        key="flux_i2i_d_cfg",
        label="img2img Distilled CFG",
        control="slider",
        scope="img2img",
        default=3.5,
        minimum=0.0,
        maximum=30.0,
        step=0.1,
    ),
    FluxFieldSpec(
        key="flux_GPU_MB",
        label="GPU Weights (MB)",
        control="slider",
        scope="shared",
        minimum=0,
        maximum_factory=lambda ctx: ctx.get("total_vram"),
        step=1,
        default_factory=_total_vram_default,
        notes="Default is computed as total_vram - 1024 to retain 1 GB headroom.",
    ),
    FluxFieldSpec(
        key="flux_t2i_sampler",
        label="txt2img sampler",
        control="dropdown",
        scope="txt2img",
        default="Euler",
    ),
    FluxFieldSpec(
        key="flux_t2i_scheduler",
        label="txt2img scheduler",
        control="dropdown",
        scope="txt2img",
        default="Simple",
    ),
    FluxFieldSpec(
        key="flux_i2i_sampler",
        label="img2img sampler",
        control="dropdown",
        scope="img2img",
        default="Euler",
    ),
    FluxFieldSpec(
        key="flux_i2i_scheduler",
        label="img2img scheduler",
        control="dropdown",
        scope="img2img",
        default="Simple",
    ),
)

FLUX_SLIDER_KEYS: Sequence[str] = tuple(spec.key for spec in FLUX_FIELD_SPECS if spec.control == "slider")
FLUX_DROPDOWN_KEYS: Sequence[str] = tuple(spec.key for spec in FLUX_FIELD_SPECS if spec.control == "dropdown")


def iter_flux_field_specs(keys: Optional[Iterable[str]] = None) -> Iterable[FluxFieldSpec]:
    if keys is None:
        yield from FLUX_FIELD_SPECS
        return

    requested = set(keys)
    for spec in FLUX_FIELD_SPECS:
        if spec.key in requested:
            yield spec
            requested.remove(spec.key)
    if requested:
        missing = ", ".join(sorted(requested))
        raise KeyError(f"Flux config keys not defined in schema: {missing}")


def build_flux_option_info(
    option_info_factory: Callable[[Any, str, Any, Optional[Dict[str, Any]]], Any],
    gr_module: Any,
    *,
    keys: Optional[Iterable[str]] = None,
    dynamic_choices: Optional[Mapping[str, Sequence[Any]]] = None,
    context: Optional[Mapping[str, Any]] = None,
    skip_unresolved: bool = False,
) -> MutableMapping[str, Any]:
    """Generate ``OptionInfo`` mappings for Flux presets.

    Parameters
    ----------
    option_info_factory:
        Callable compatible with ``shared.OptionInfo`` to build option metadata.
    gr_module:
        The imported ``gradio`` module so we can reference slider/dropdown types.
    keys:
        Optional iterable restricting which schema entries are emitted.
    dynamic_choices:
        Optional mapping providing dropdown choices keyed by field name.
    context:
        Supplemental context used by ``default_factory`` implementations. If not
        provided, defaults to an empty mapping.
    skip_unresolved:
        When ``True``, fields lacking dropdown choices are ignored instead of
        raising ``ValueError``.
    """

    context = context or {}
    dynamic_choices = dynamic_choices or {}
    option_info: Dict[str, Any] = {}

    for spec in iter_flux_field_specs(keys):
        component_kwargs: Dict[str, Any]
        component_type: Any

        if spec.control == "slider":
            component_type = gr_module.Slider
            component_kwargs = {}
            minimum_value = spec.minimum
            maximum_value = spec.maximum
            if spec.minimum_factory is not None:
                minimum_value = spec.minimum_factory(context)
            if spec.maximum_factory is not None:
                maximum_value = spec.maximum_factory(context)
            if minimum_value is not None:
                component_kwargs["minimum"] = minimum_value
            if maximum_value is not None:
                component_kwargs["maximum"] = maximum_value
            if spec.step is not None:
                component_kwargs["step"] = spec.step
        elif spec.control == "dropdown":
            component_type = gr_module.Dropdown
            component_kwargs = {
                "choices": dynamic_choices.get(spec.key, spec.choices),
            }
            if component_kwargs["choices"] is None:
                if skip_unresolved:
                    continue
                raise ValueError(f"Dropdown choices for '{spec.key}' were not provided")
        else:
            raise ValueError(f"Unsupported control type '{spec.control}' for key '{spec.key}'")

        default_value = spec.resolve_default(context)
        option_info[spec.key] = option_info_factory(default_value, spec.label, component_type, component_kwargs)

    return option_info

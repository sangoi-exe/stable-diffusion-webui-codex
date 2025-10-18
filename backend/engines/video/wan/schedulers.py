from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

try:
    # Diffusers schedulers (classes may not exist on older versions)
    from diffusers import (
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        PNDMScheduler,
    )  # type: ignore
except Exception:  # pragma: no cover
    EulerAncestralDiscreteScheduler = None  # type: ignore
    EulerDiscreteScheduler = None  # type: ignore
    DDIMScheduler = None  # type: ignore
    DPMSolverMultistepScheduler = None  # type: ignore
    PNDMScheduler = None  # type: ignore


@dataclass(frozen=True)
class MappingOutcome:
    sampler_in: str
    scheduler_in: str
    sampler_effective: str
    scheduler_effective: str
    warnings: List[str]


SUPPORTED_SAMPLERS: Mapping[str, Tuple[str, Mapping[str, Any]] ] = {
    # sampler_name -> (diffusers_class_name, default_overrides)
    "Euler a": ("EulerAncestralDiscreteScheduler", {}),
    "Euler": ("EulerDiscreteScheduler", {}),
    "DDIM": ("DDIMScheduler", {}),
    "DPM++ 2M": ("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 2}),
    "DPM++ 2M SDE": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "solver_order": 2}),
    "PLMS": ("PNDMScheduler", {"skip_prk_steps": True}),
}


def allowed_samplers_for_engine(engine_key: str) -> List[str]:
    # Narrow WAN to a stable subset; expand later per validation
    if engine_key == "wan_ti2v_5b":
        return [
            "Euler a",
            "Euler",
            "DDIM",
            "DPM++ 2M",
            "DPM++ 2M SDE",
            "PLMS",
        ]
    # Default: expose known ones
    return list(SUPPORTED_SAMPLERS.keys())


def _class_by_name(name: str):
    return {
        "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "DDIMScheduler": DDIMScheduler,
        "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
        "PNDMScheduler": PNDMScheduler,
    }.get(name)


def _apply_schedule_overrides(config: Dict[str, Any], scheduler_name: str) -> Tuple[Dict[str, Any], List[str]]:
    # Maps UI scheduler to diffusers config flags
    cfg = dict(config)
    warns: List[str] = []

    if scheduler_name == "Karras":
        # Many schedulers support the flag; harmless if ignored
        cfg["use_karras_sigmas"] = True
    elif scheduler_name == "Exponential":
        cfg["use_exponential_sigmas"] = True
    elif scheduler_name == "Simple":
        cfg["timestep_spacing"] = "trailing"
    elif scheduler_name == "Automatic":
        pass
    else:
        warns.append(f"Unknown scheduler '{scheduler_name}', using defaults")

    return cfg, warns


def apply_sampler_scheduler(pipe: Any, sampler_name: str, scheduler_name: str) -> MappingOutcome:
    """Replace pipe.scheduler based on UI sampler/scheduler names.

    Returns MappingOutcome with effective choices and potential warnings.
    """

    warnings: List[str] = []
    sampler_in = sampler_name or "Automatic"
    scheduler_in = scheduler_name or "Automatic"

    chosen_sampler = sampler_in
    if sampler_in.lower() == "automatic":
        chosen_sampler = "Euler a"

    spec = SUPPORTED_SAMPLERS.get(chosen_sampler)
    if spec is None:
        # Fallback to Euler a
        warnings.append(f"Unsupported sampler '{chosen_sampler}' for this engine; using 'Euler a'")
        chosen_sampler = "Euler a"
        spec = SUPPORTED_SAMPLERS[chosen_sampler]

    class_name, overrides = spec
    klass = _class_by_name(class_name)
    if klass is None:
        warnings.append(
            f"Diffusers scheduler '{class_name}' not available; keeping existing scheduler"
        )
        return MappingOutcome(sampler_in, scheduler_in, pipe.scheduler.__class__.__name__, scheduler_in or "Automatic", warnings)

    base_cfg: Dict[str, Any] = dict(getattr(pipe.scheduler, "config", {}))
    cfg = {**base_cfg, **overrides}
    cfg, schedule_warns = _apply_schedule_overrides(cfg, scheduler_in)
    warnings.extend(schedule_warns)

    try:
        pipe.scheduler = klass.from_config(cfg)
        effective_sched = pipe.scheduler.__class__.__name__
    except Exception as exc:
        warnings.append(f"Failed to apply scheduler '{class_name}': {exc}; keeping previous scheduler")
        effective_sched = pipe.scheduler.__class__.__name__

    return MappingOutcome(sampler_in, scheduler_in, chosen_sampler, effective_sched, warnings)


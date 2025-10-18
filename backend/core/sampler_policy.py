from __future__ import annotations

from typing import List

from .engine_interface import TaskType


_COMMON_SAMPLERS: List[str] = [
    "Euler a",
    "Euler",
    "DDIM",
    "DPM++ 2M",
    "DPM++ 2M SDE",
    "PLMS",
]

_COMMON_SCHEDULERS: List[str] = [
    "Automatic",
    "Karras",
    "Simple",
    "Exponential",
]


def allowed_samplers(engine_key: str, task: TaskType) -> List[str]:
    key = (engine_key or "").strip().lower()
    # Conservative, compatible subset across engines. Expand per-engine later.
    if key in {"wan_ti2v_5b", "wan_t2v_14b", "wan_i2v_14b", "hunyuan_video", "svd"}:
        return list(_COMMON_SAMPLERS)
    if key in {"sd15", "sdxl", "flux"}:
        return list(_COMMON_SAMPLERS)
    # Unknown engine: return common set
    return list(_COMMON_SAMPLERS)


def allowed_schedulers(engine_key: str, task: TaskType) -> List[str]:
    # For WAN/video engines we constrain aggressively; for SD/Flux we allow same common set for UI consistency.
    # Backends may still ignore/override with explicit logs.
    return list(_COMMON_SCHEDULERS)


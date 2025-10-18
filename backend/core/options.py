from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .enums import Precision


@dataclass(frozen=True)
class EngineOptions:
    unet_storage_dtype: Optional[Precision] = None
    additional_modules: List[str] = field(default_factory=list)

    @staticmethod
    def from_mapping(m: object) -> "EngineOptions":
        if not isinstance(m, dict):
            return EngineOptions()
        dtype = m.get("unet_storage_dtype")
        precise = None
        try:
            if isinstance(dtype, str):
                precise = Precision(dtype)
        except Exception:
            precise = None
        modules = []
        if isinstance(m.get("additional_modules"), (list, tuple)):
            modules = [str(x) for x in m["additional_modules"]]
        return EngineOptions(unet_storage_dtype=precise, additional_modules=modules)


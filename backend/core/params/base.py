from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class BaseParams:
    """Base class for engine-specific, task-specific parameter interfaces.

    Instances should be constructed by parser functions that validate and
    normalize incoming request fields.
    """

    def as_dict(self) -> Mapping[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}  # type: ignore[attr-defined]

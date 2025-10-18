"""Wan 2.2 I2V 14B (Image → Video) engine stub."""

from __future__ import annotations

from typing import Iterator

from backend.core.engine_interface import EngineCapabilities, TaskType
from backend.core.requests import InferenceEvent, Img2VidRequest
from backend.core.param_registry import parse_params
from backend.core.exceptions import UnsupportedTaskError, EngineLoadError

from ...base import DiffusionEngine


class WanI2V14BEngine(DiffusionEngine):
    engine_id = "wan_i2v_14b"

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.IMG2VID,),
            model_types=("wan-2.2-i2v-14b",),
            precision=("fp16", "bf16"),
            extras={"notes": "Stub engine; integration pending with WAN 2.2 I2V 14B"},
        )

    def img2vid(self, request: Img2VidRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        _ = parse_params(self.engine_id, TaskType.IMG2VID, request)
        self._not_implemented(TaskType.IMG2VID, "WAN 2.2 I2V 14B integration pending")
        yield from ()

    def load(self, model_ref: str, **options: object) -> None:  # type: ignore[override]
        if not model_ref:
            raise EngineLoadError("wan_i2v_14b load failed: empty model_ref")
        self.mark_loaded()

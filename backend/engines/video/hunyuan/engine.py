"""Hunyuan Video engine stub."""

from __future__ import annotations

from typing import Iterator

from backend.core.engine_interface import EngineCapabilities, TaskType
from backend.core.requests import InferenceEvent, Txt2VidRequest, Img2VidRequest
from backend.core.param_registry import parse_params
from backend.core.exceptions import UnsupportedTaskError, EngineLoadError

from ...base import DiffusionEngine


class HunyuanVideoEngine(DiffusionEngine):
    engine_id = "hunyuan_video"

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2VID, TaskType.IMG2VID),
            model_types=("hunyuan-video",),
            precision=("fp16", "bf16"),
            extras={"notes": "Stub engine; txt2vid & img2vid"},
        )

    def txt2vid(self, request: Txt2VidRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        _ = parse_params(self.engine_id, TaskType.TXT2VID, request)
        self._not_implemented(TaskType.TXT2VID, "pending video engines")
        yield from ()

    def img2vid(self, request: Img2VidRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        _ = parse_params(self.engine_id, TaskType.IMG2VID, request)
        self._not_implemented(TaskType.IMG2VID, "pending video engines")
        yield from ()

    def load(self, model_ref: str, **options: object) -> None:  # type: ignore[override]
        if not model_ref:
            raise EngineLoadError("hunyuan_video load failed: empty model_ref")
        self.mark_loaded()

"""Core orchestration primitives for the modular inference pipeline.

This package is intentionally lightweight: it only declares interfaces,
request/response dataclasses, registry utilities, and the orchestrator shell.
Concrete engines live under ``backend.engines`` and will be wired in
subsequent phases behind feature flags.
"""

from .engine_interface import BaseInferenceEngine, EngineCapabilities, TaskType
from .requests import (
    Img2ImgRequest,
    Img2VidRequest,
    InferenceEvent,
    ProgressEvent,
    ResultEvent,
    Txt2ImgRequest,
    Txt2VidRequest,
)
from .registry import EngineRegistry, EngineDescriptor, registry
from .orchestrator import InferenceOrchestrator

__all__ = [
    "BaseInferenceEngine",
    "EngineCapabilities",
    "EngineDescriptor",
    "EngineRegistry",
    "Img2ImgRequest",
    "Img2VidRequest",
    "InferenceEvent",
    "InferenceOrchestrator",
    "ProgressEvent",
    "ResultEvent",
    "TaskType",
    "Txt2ImgRequest",
    "Txt2VidRequest",
    "registry",
]

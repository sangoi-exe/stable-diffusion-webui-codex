"""Custom exception hierarchy for the modular inference pipeline."""

from __future__ import annotations


class EngineError(RuntimeError):
    """Base error for engine-related failures."""


class EngineRegistrationError(EngineError):
    """Raised when attempting to register an engine with conflicting keys."""


class EngineNotFoundError(EngineError):
    """Raised when resolving an unknown engine key or alias."""


class UnsupportedTaskError(EngineError):
    """Raised when the requested task is not implemented by the engine."""


class EngineLoadError(EngineError):
    """Raised when an engine fails to load required weights or resources."""


class EngineExecutionError(EngineError):
    """Raised when an engine encounters a runtime failure during inference."""

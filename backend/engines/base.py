"""Shared helpers for diffusion-style inference engines."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from backend.core.engine_interface import BaseInferenceEngine, TaskType
from backend.core.exceptions import UnsupportedTaskError


class DiffusionEngine(BaseInferenceEngine):
    """Base helper for diffusion engines.

    Provides common load/unload bookkeeping and logging. Concrete engines are
    expected to override the task methods they support.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model_ref: Optional[str] = None
        self._load_options: Dict[str, Any] = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        self._logger.info(
            "Loading engine %s (model=%s, options=%s)",
            self.engine_id,
            model_ref,
            options or {},
        )
        self._model_ref = model_ref
        self._load_options = dict(options)
        self.mark_loaded()

    def unload(self) -> None:  # type: ignore[override]
        self._logger.info("Unloading engine %s (model=%s)", self.engine_id, self._model_ref)
        self._model_ref = None
        self._load_options.clear()
        self.mark_unloaded()

    # ------------------------------------------------------------------
    def status(self) -> Mapping[str, Any]:  # type: ignore[override]
        base = super().status()
        return {
            **base,
            "model_ref": self._model_ref,
            "options": dict(self._load_options),
        }

    # ------------------------------------------------------------------
    def _not_implemented(self, task: TaskType, detail: str = "") -> None:
        suffix = f" detail={detail}" if detail else ""
        raise UnsupportedTaskError(
            "Engine '%s' does not implement task '%s' yet.%s model=%s options=%s"
            % (
                self.engine_id,
                task.value,
                suffix,
                self._model_ref,
                self._load_options,
            )
        )

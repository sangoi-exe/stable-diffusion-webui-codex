"""High-level orchestrator to route requests to engines and stream events."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

from .engine_interface import BaseInferenceEngine, TaskType
from .exceptions import EngineExecutionError, EngineNotFoundError, EngineLoadError, UnsupportedTaskError
from .registry import EngineRegistry, registry as global_registry
from .requests import InferenceEvent, ProgressEvent


logger = logging.getLogger(__name__)


class InferenceOrchestrator:
    """Routes typed requests to registered engines.

    This orchestrator is intentionally thin. It resolves engine instances via the
    registry, ensures they are loaded for the requested model, and streams
    events back to the caller. It does not persist results or mutate UI state.
    """

    def __init__(
        self,
        registry: Optional[EngineRegistry] = None,
        *,
        enable_cache: bool = True,
    ) -> None:
        self._registry = registry or global_registry
        self._enable_cache = enable_cache
        self._engine_cache: MutableMapping[str, BaseInferenceEngine] = {}

    # ------------------------------------------------------------------
    def run(
        self,
        task: TaskType,
        engine_key: str,
        request: object,
        *,
        model_ref: Optional[str] = None,
        engine_options: Optional[Mapping[str, object]] = None,
    ) -> Iterator[InferenceEvent]:
        start = time.perf_counter()
        engine = self._resolve_engine(engine_key, engine_options or {})

        logger.info(
            "Orchestrator dispatch: task=%s engine=%s model=%s", task.value, engine_key, model_ref or "default"
        )

        capabilities = engine.capabilities()
        if not capabilities.supports(task):
            raise UnsupportedTaskError(
                f"Engine '{engine_key}' does not support task '{task.value}'. Supported: {capabilities.tasks}"
            )

        if model_ref is not None:
            needs_load = False
            if not engine._is_loaded:  # noqa: SLF001 (intentional internal check)
                needs_load = True
            else:
                try:
                    cur_model = engine.status().get("model_ref")
                    needs_load = cur_model != model_ref
                except Exception:
                    needs_load = True

            if needs_load:
                try:
                    engine.load(model_ref, **(engine_options or {}))
                    engine.mark_loaded()
                except Exception as exc:  # noqa: BLE001
                    raise EngineLoadError(
                        f"Failed to load engine '{engine_key}' for model '{model_ref}': {exc}"
                    ) from exc

        handler = getattr(engine, task.value, None)
        if handler is None:
            raise UnsupportedTaskError(f"Engine '{engine_key}' is missing handler for task '{task.value}'")

        try:
            yield ProgressEvent(stage="start", percent=0.0, message="Starting inference")
            for event in handler(request):
                yield event
        except UnsupportedTaskError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Engine '%s' failed during '%s'", engine_key, task.value)
            raise EngineExecutionError(str(exc)) from exc
        finally:
            elapsed = time.perf_counter() - start
            yield ProgressEvent(stage="end", percent=100.0, message="Inference complete", data={"elapsed": elapsed})

    # ------------------------------------------------------------------
    def _resolve_engine(self, engine_key: str, engine_options: Mapping[str, object]) -> BaseInferenceEngine:
        normalized_key = engine_key.strip().lower()
        if self._enable_cache and normalized_key in self._engine_cache:
            return self._engine_cache[normalized_key]

        try:
            engine = self._registry.create(normalized_key, **engine_options)
        except EngineNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EngineExecutionError(f"Failed to create engine '{engine_key}': {exc}") from exc

        if self._enable_cache:
            self._engine_cache[normalized_key] = engine

        return engine

    # ------------------------------------------------------------------
    def evict(self, engine_key: str) -> None:
        normalized_key = engine_key.strip().lower()
        engine = self._engine_cache.pop(normalized_key, None)
        if engine is None:
            return
        with contextlib.suppress(Exception):
            engine.unload()
            engine.mark_unloaded()
        logger.info("Evicted engine '%s'", normalized_key)

    def clear_cache(self) -> None:
        for key in list(self._engine_cache.keys()):
            self.evict(key)

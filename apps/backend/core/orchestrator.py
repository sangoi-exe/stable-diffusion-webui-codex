"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend inference orchestrator (engine routing + caching + event streaming).
Resolves engines from the registry, canonicalizes aliases to descriptor keys before cache/fingerprint ownership, loads/unloads per request,
fingerprints load-affecting options, purges VRAM on model swaps, and yields typed progress events back to API callers.
On load/execution failures, performs a best-effort purge to release VRAM/RAM so the backend can recover without restart.

Symbols (top-level; keep in sync; no ghosts):
- `InferenceOrchestrator` (class): Routes typed requests to engines; caches loaded engines with option fingerprinting, reloads when overrides
  change (incl. `vae_source`/`tenc_source`/`zimage_variant`/`qwen_image_variant`/`lens_variant`/`dtype`), and manages VRAM hygiene across generations (contains nested helpers for option freezing and cache purges).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import gc
import logging
import threading
import time
from typing import Iterator, Mapping, MutableMapping, Optional

from .engine_interface import BaseInferenceEngine, TaskType
from .exceptions import EngineExecutionError, EngineNotFoundError, EngineLoadError, UnsupportedTaskError
from .registry import EngineRegistry, registry as global_registry
from .requests import InferenceEvent, ProgressEvent
from .strict_values import parse_bool_value
from apps.backend.runtime.load_authority import (
    LoadAuthorityStage,
    LoadAuthorityViolationError,
    coordinator_load_permit,
    require_load_authority,
)
from apps.backend.runtime.diagnostics.error_summary import summarize_exception_for_console
from apps.backend.runtime.diagnostics.exception_hook import dump_exception
from apps.backend.runtime.families.qwen_image.config import QWEN_IMAGE_VARIANT_KEY
from apps.backend.runtime.families.lens.config import LENS_VARIANT_KEY


logger = get_backend_logger(__name__)


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
        self._engine_options_fingerprint: MutableMapping[str, object] = {}
        self._last_generation_signature: object | None = None
        self._run_lock = threading.Lock()

    @staticmethod
    def _freeze_engine_options(value: object) -> object:
        """Return a comparable, stable structure for engine option fingerprints."""
        if isinstance(value, dict):
            return tuple((str(k), InferenceOrchestrator._freeze_engine_options(v)) for k, v in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(InferenceOrchestrator._freeze_engine_options(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted((InferenceOrchestrator._freeze_engine_options(v) for v in value), key=repr))
        return value

    @staticmethod
    def _reload_fingerprint(engine_options: Mapping[str, object]) -> object:
        """Fingerprint options that change loaded weights/runtime wiring.

        The orchestrator caches engine instances, so changing options like
        text encoder overrides or VAE overrides must trigger a reload even when
        model_ref is unchanged.
        """

        te_override = engine_options.get("text_encoder_override")
        vae_path = engine_options.get("vae_path")
        vae_source = engine_options.get("vae_source")
        tenc_path = engine_options.get("tenc_path")
        tenc_source = engine_options.get("tenc_source")
        zimage_variant = engine_options.get("zimage_variant")
        qwen_image_variant = engine_options.get(QWEN_IMAGE_VARIANT_KEY)
        lens_variant = engine_options.get(LENS_VARIANT_KEY)
        dtype_raw = engine_options.get("dtype")
        if dtype_raw is None:
            dtype_value = None
        else:
            dtype_normalized = str(dtype_raw).strip().lower()
            dtype_value = None if dtype_normalized in {"", "auto"} else dtype_normalized

        # Normalize streaming option key to a single boolean or None.
        streaming_val: bool | None = None
        if "core_streaming_enabled" in engine_options:
            core_streaming_value = engine_options.get("core_streaming_enabled")
            streaming_val = (
                None
                if core_streaming_value is None
                else parse_bool_value(
                    core_streaming_value,
                    field="engine_options.core_streaming_enabled",
                )
            )

        relevant = {
            "text_encoder_override": te_override,
            "vae_path": vae_path,
            "vae_source": vae_source,
            "tenc_path": tenc_path,
            "tenc_source": tenc_source,
            "zimage_variant": zimage_variant,
            QWEN_IMAGE_VARIANT_KEY: qwen_image_variant,
            LENS_VARIANT_KEY: lens_variant,
            "core_streaming_enabled": streaming_val,
            "dtype": dtype_value,
        }
        return InferenceOrchestrator._freeze_engine_options(relevant)

    @staticmethod
    def _component_device(component: object) -> object | None:
        if component is None:
            return None

        load_device = getattr(component, "load_device", None)
        if load_device is not None:
            return load_device

        device = getattr(component, "device", None)
        if device is not None:
            return device

        if hasattr(component, "parameters"):
            try:
                first_param = next(component.parameters())
                param_device = getattr(first_param, "device", None)
                if param_device is not None:
                    return param_device
            except Exception:
                pass

        nested_model = getattr(component, "model", None)
        if nested_model is not None and nested_model is not component:
            nested_device = InferenceOrchestrator._component_device(nested_model)
            if nested_device is not None:
                return nested_device

        return None

    @staticmethod
    def _engine_primary_component_device(engine: BaseInferenceEngine) -> object | None:
        codex_objects = getattr(engine, "codex_objects", None)
        if codex_objects is None:
            return None

        denoiser = getattr(codex_objects, "denoiser", None)
        legacy_unet = getattr(codex_objects, "unet", None)
        for component in (denoiser, legacy_unet):
            resolved = InferenceOrchestrator._component_device(component)
            if resolved is not None:
                return resolved
        return None

    def _generation_signature(
        self,
        engine_key: str,
        model_ref: str,
        engine_options: Mapping[str, object],
    ) -> object:
        canonical_key = self._canonical_engine_key(engine_key)
        relevant = {
            "engine_key": canonical_key,
            "model_ref": model_ref,
            "reload_fingerprint": self._reload_fingerprint(engine_options),
        }
        return InferenceOrchestrator._freeze_engine_options(relevant)

    def _canonical_engine_key(self, engine_key: str) -> str:
        """Return the registry descriptor key for an exact key or alias."""

        return self._registry.get_descriptor(engine_key.strip().lower()).key

    @staticmethod
    def _guarded_engine_load(engine: BaseInferenceEngine, model_ref: str, engine_opts: Mapping[str, object]) -> None:
        require_load_authority(
            "core.orchestrator.engine.load",
            allowed_stages=(LoadAuthorityStage.LOAD, LoadAuthorityStage.RELOAD),
        )
        engine.load(model_ref, **engine_opts)

    @staticmethod
    def _guarded_engine_unload(engine: BaseInferenceEngine) -> None:
        require_load_authority(
            "core.orchestrator.engine.unload",
            allowed_stages=(
                LoadAuthorityStage.LOAD,
                LoadAuthorityStage.MATERIALIZE,
                LoadAuthorityStage.UNLOAD,
                LoadAuthorityStage.RELOAD,
                LoadAuthorityStage.CLEANUP,
            ),
        )
        engine.unload()

    @staticmethod
    def _engine_residency_targets(engine: BaseInferenceEngine) -> list[tuple[str, object]]:
        codex_objects = getattr(engine, "codex_objects", None)
        if codex_objects is None:
            return []

        targets: list[tuple[str, object]] = []
        for label in ("denoiser", "unet", "vae", "clipvision"):
            candidate = getattr(codex_objects, label, None)
            if candidate is not None:
                targets.append((label, candidate))

        text_encoders = getattr(codex_objects, "text_encoders", None)
        if isinstance(text_encoders, Mapping):
            for name, candidate in text_encoders.items():
                if candidate is None:
                    continue
                targets.append((f"text_encoder:{name}", candidate))

        return targets

    @staticmethod
    def _verify_engine_unloaded(engine: BaseInferenceEngine, *, source: str) -> None:
        loaded_flag = getattr(engine, "_is_loaded", None)
        if not isinstance(loaded_flag, bool):
            raise RuntimeError(
                f"Post-unload residency verification failed at {source}: invalid engine._is_loaded={loaded_flag!r}."
            )

        status_loaded = loaded_flag
        status_payload = engine.status()
        if isinstance(status_payload, Mapping):
            raw_loaded = status_payload.get("loaded", loaded_flag)
            if not isinstance(raw_loaded, bool):
                raise RuntimeError(
                    f"Post-unload residency verification failed at {source}: "
                    f"engine.status()['loaded'] is not bool ({raw_loaded!r})."
                )
            status_loaded = raw_loaded

        if loaded_flag or status_loaded:
            raise RuntimeError(
                f"Post-unload residency verification failed at {source}: "
                f"engine still reports loaded (_is_loaded={loaded_flag}, status.loaded={status_loaded})."
            )

    def _unload_engine_with_residency_verification(
        self,
        engine: BaseInferenceEngine,
        *,
        source: str,
    ) -> None:
        targets = self._engine_residency_targets(engine)
        tracked_targets: list[tuple[str, object]] = []

        if targets:
            from apps.backend.runtime.memory import memory_management as _mem

            for label, target in targets:
                try:
                    was_loaded = _mem.manager.is_model_loaded(target)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"Failed to evaluate pre-unload residency target '{label}' at {source}: {exc}"
                    ) from exc
                if was_loaded:
                    tracked_targets.append((label, target))

        self._guarded_engine_unload(engine)
        engine.mark_unloaded()
        self._verify_engine_unloaded(engine, source=source)

        if tracked_targets:
            from apps.backend.runtime.memory import memory_management as _mem

            lingering: list[str] = []
            for label, target in tracked_targets:
                try:
                    if _mem.manager.is_model_loaded(target):
                        lingering.append(label)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"Failed to evaluate post-unload residency target '{label}' at {source}: {exc}"
                    ) from exc

            if lingering:
                raise RuntimeError(
                    f"Post-unload residency verification failed at {source}: "
                    f"lingering memory-manager residency for {', '.join(lingering)}."
                )

    def _purge_vram(self, *, reason: str, clear_engine_cache: bool = False) -> None:
        def _is_cleanup_oom(detail: str) -> bool:
            text = str(detail or "").strip().lower()
            return ("out of memory" in text) and ("cuda" in text)

        with coordinator_load_permit(
            owner="core.orchestrator._purge_vram",
            stage=LoadAuthorityStage.CLEANUP,
        ):
            purge_failures: list[str] = []
            for cached_engine in list(self._engine_cache.values()):
                if not clear_engine_cache and not getattr(cached_engine, "_is_loaded", False):
                    continue
                try:
                    self._unload_engine_with_residency_verification(
                        cached_engine,
                        source="core.orchestrator._purge_vram",
                    )
                except Exception as exc:  # noqa: BLE001
                    purge_failures.append(f"cached_engine_unload:{exc}")

            if clear_engine_cache:
                self._engine_cache.clear()
                self._engine_options_fingerprint.clear()
                self._last_generation_signature = None

            self._engine_options_fingerprint.clear()

            try:
                from apps.backend.runtime.memory import memory_management as _mem

                _mem.manager.unload_all_models()
                _mem.manager.soft_empty_cache(force=True)
            except Exception as exc:  # noqa: BLE001
                purge_failures.append(f"memory_manager:{exc}")

            try:
                gc.collect()
            except Exception:  # pragma: no cover
                pass
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:  # pragma: no cover
                pass

            if purge_failures:
                reason_lc = str(reason or "").strip().lower()
                if "engine execution failure" in reason_lc and all(_is_cleanup_oom(item) for item in purge_failures):
                    detail = "; ".join(purge_failures[:3])
                    if len(purge_failures) > 3:
                        detail = f"{detail}; ... (+{len(purge_failures) - 3} more)"
                    logger.warning(
                        "VRAM purge encountered non-fatal CUDA OOM while unwinding execution failure (%s): %s",
                        reason,
                        detail,
                    )
                    return

                detail = "; ".join(purge_failures[:3])
                if len(purge_failures) > 3:
                    detail = f"{detail}; ... (+{len(purge_failures) - 3} more)"
                raise RuntimeError(f"VRAM purge failed ({reason}): {detail}")

            logger.info("VRAM purge complete (%s).", reason)

    def _maybe_purge_vram_for_generation(
        self,
        *,
        engine_key: str,
        model_ref: str,
        engine_options: Mapping[str, object],
    ) -> None:
        signature = self._generation_signature(engine_key, model_ref, engine_options)
        with self._run_lock:
            prev = self._last_generation_signature
            if prev is not None and prev != signature:
                logger.info(
                    "Generation signature changed; purging VRAM before load. engine=%s model=%s",
                    engine_key,
                    model_ref,
                )
                self._purge_vram(reason="checkpoint/text-encoder selection changed")
            self._last_generation_signature = signature

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
        canonical_key = self._canonical_engine_key(engine_key)
        engine_opts = engine_options or {}
        if model_ref is not None:
            try:
                self._maybe_purge_vram_for_generation(
                    engine_key=canonical_key,
                    model_ref=str(model_ref),
                    engine_options=engine_opts,
                )
            except Exception as exc:
                raise EngineLoadError(
                    f"Failed load preflight for engine '{engine_key}' with model '{model_ref}': {exc}"
                ) from exc
        engine = self._resolve_engine(canonical_key, engine_opts)

        logger.info(
            "Orchestrator dispatch: task=%s engine=%s requested_engine=%s model=%s",
            task.value,
            canonical_key,
            engine_key,
            model_ref or "default",
        )

        capabilities = engine.capabilities()
        if not capabilities.supports(task):
            raise UnsupportedTaskError(
                f"Engine '{engine_key}' does not support task '{task.value}'. Supported: {capabilities.tasks}"
            )

        if model_ref is not None:
            needs_load = False
            device_mismatch = False
            try:
                reload_fingerprint = self._reload_fingerprint(engine_opts)
            except Exception as exc:
                raise EngineLoadError(f"Invalid load-affecting engine option(s): {exc}") from exc
            if not engine._is_loaded:  # noqa: SLF001 (intentional internal check)
                needs_load = True
            else:
                try:
                    cur_model = engine.status().get("model_ref")
                    needs_load = cur_model != model_ref
                except Exception:
                    needs_load = True
                # Reload when load-affecting engine options changed.
                prev = self._engine_options_fingerprint.get(canonical_key)
                if prev is not None and prev != reload_fingerprint:
                    needs_load = True
                # Reload if the primary device changed since last load
                try:
                    from apps.backend.runtime.memory import memory_management as _mem
                    desired = _mem.manager.primary_device()
                    dcur = self._engine_primary_component_device(engine)
                    if dcur is not None:
                        device_mismatch = str(dcur) != str(desired)
                except Exception:
                    device_mismatch = False

            if needs_load or device_mismatch:
                load_stage = LoadAuthorityStage.RELOAD if engine._is_loaded else LoadAuthorityStage.LOAD
                try:
                    with coordinator_load_permit(
                        owner="core.orchestrator.run",
                        stage=load_stage,
                    ):
                        if engine._is_loaded:
                            self._unload_engine_with_residency_verification(
                                engine,
                                source="core.orchestrator.run.reload_transition",
                            )
                            logger.info(
                                "Running VRAM cleanup barrier between unload/load transition. engine=%s model=%s",
                                engine_key,
                                model_ref,
                            )
                            self._purge_vram(reason="engine unload/load transition barrier")
                        self._guarded_engine_load(engine, model_ref, engine_opts)
                        engine.mark_loaded()
                        self._engine_options_fingerprint[canonical_key] = reload_fingerprint
                except LoadAuthorityViolationError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    dump_exception(
                        type(exc),
                        exc,
                        exc.__traceback__,
                        where="core.orchestrator.run.load",
                        context={"engine_key": engine_key, "model_ref": str(model_ref)},
                    )
                    logger.error(
                        "Engine '%s' failed during load (model=%s): %s",
                        engine_key,
                        model_ref,
                        summarize_exception_for_console(exc),
                        exc_info=False,
                    )
                    cleanup_failures: list[str] = []
                    with coordinator_load_permit(
                        owner="core.orchestrator.run",
                        stage=LoadAuthorityStage.CLEANUP,
                    ):
                        try:
                            self._unload_engine_with_residency_verification(
                                engine,
                                source="core.orchestrator.run.load_failure_cleanup",
                            )
                        except Exception as cleanup_exc:  # noqa: BLE001
                            cleanup_failures.append(f"engine_unload:{cleanup_exc}")
                    try:
                        self._purge_vram(reason="engine load failure", clear_engine_cache=True)
                    except Exception as purge_exc:  # noqa: BLE001
                        cleanup_failures.append(f"purge:{purge_exc}")
                    if cleanup_failures:
                        detail = "; ".join(cleanup_failures)
                        raise EngineLoadError(
                            f"Failed to load engine '{engine_key}' for model '{model_ref}': {exc}. "
                            f"Additional cleanup failure(s): {detail}"
                        ) from exc
                    raise EngineLoadError(
                        f"Failed to load engine '{engine_key}' for model '{model_ref}': {exc}"
                    ) from exc

        handler = getattr(engine, task.value, None)
        if handler is None:
            raise UnsupportedTaskError(f"Engine '{engine_key}' is missing handler for task '{task.value}'")

        try:
            yield ProgressEvent(stage="start", percent=0.0, message="Starting inference")
            with coordinator_load_permit(
                owner="core.orchestrator.run",
                stage=LoadAuthorityStage.MATERIALIZE,
            ):
                for event in handler(request):
                    yield event
        except UnsupportedTaskError:
            raise
        except LoadAuthorityViolationError:
            raise
        except Exception as exc:  # noqa: BLE001
            dump_exception(
                type(exc),
                exc,
                exc.__traceback__,
                where="core.orchestrator.run.execution",
                context={"engine_key": engine_key, "task": task.value},
            )
            logger.error(
                "Engine '%s' failed during '%s': %s",
                engine_key,
                task.value,
                summarize_exception_for_console(exc),
                exc_info=False,
            )
            cleanup_failures: list[str] = []
            with coordinator_load_permit(
                owner="core.orchestrator.run",
                stage=LoadAuthorityStage.UNLOAD,
            ):
                try:
                    self._unload_engine_with_residency_verification(
                        engine,
                        source="core.orchestrator.run.execution_failure_cleanup",
                    )
                except Exception as cleanup_exc:  # noqa: BLE001
                    cleanup_failures.append(f"engine_unload:{cleanup_exc}")
            try:
                self._purge_vram(reason="engine execution failure", clear_engine_cache=True)
            except Exception as purge_exc:  # noqa: BLE001
                cleanup_failures.append(f"purge:{purge_exc}")
            if cleanup_failures:
                detail = "; ".join(cleanup_failures)
                raise EngineExecutionError(
                    f"Engine '{engine_key}' failed during '{task.value}': {exc}. "
                    f"Additional cleanup failure(s): {detail}"
                ) from exc
            raise EngineExecutionError(f"Engine '{engine_key}' failed during '{task.value}': {exc}") from exc

        elapsed = time.perf_counter() - start
        yield ProgressEvent(stage="end", percent=100.0, message="Inference complete", data={"elapsed": elapsed})

    # ------------------------------------------------------------------
    def _resolve_engine(self, engine_key: str, engine_options: Mapping[str, object]) -> BaseInferenceEngine:
        canonical_key = self._canonical_engine_key(engine_key)
        if self._enable_cache and canonical_key in self._engine_cache:
            return self._engine_cache[canonical_key]

        try:
            # Do not pass engine_options to constructor: options are applied on load()
            engine = self._registry.create(canonical_key)
        except EngineNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EngineExecutionError(f"Failed to create engine '{engine_key}': {exc}") from exc

        if self._enable_cache:
            self._engine_cache[canonical_key] = engine

        return engine

    # ------------------------------------------------------------------
    def evict(self, engine_key: str) -> None:
        try:
            canonical_key = self._canonical_engine_key(engine_key)
        except EngineNotFoundError:
            canonical_key = engine_key.strip().lower()
        engine = self._engine_cache.pop(canonical_key, None)
        self._engine_options_fingerprint.pop(canonical_key, None)
        if engine is None:
            return
        with coordinator_load_permit(
            owner="core.orchestrator.evict",
            stage=LoadAuthorityStage.CLEANUP,
        ):
            self._unload_engine_with_residency_verification(
                engine,
                source="core.orchestrator.evict",
            )
        logger.info("Evicted engine '%s'", canonical_key)

    def clear_cache(self) -> None:
        for key in list(self._engine_cache.keys()):
            self.evict(key)

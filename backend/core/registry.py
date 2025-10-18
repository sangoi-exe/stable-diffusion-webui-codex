"""Registry utilities for inference engines."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar

from .engine_interface import BaseInferenceEngine
from .exceptions import EngineNotFoundError, EngineRegistrationError


logger = logging.getLogger(__name__)

EngineT = TypeVar("EngineT", bound=BaseInferenceEngine)


@dataclass(frozen=True)
class EngineDescriptor:
    key: str
    cls: Type[EngineT]
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)


class EngineRegistry:
    """Thread-safe in-memory registry for inference engines."""

    def __init__(self) -> None:
        self._descriptors: Dict[str, EngineDescriptor] = {}
        self._alias_to_key: Dict[str, str] = {}
        self._lock = RLock()

    # ------------------------------------------------------------------
    def register(
        self,
        key: str,
        engine_cls: Type[EngineT],
        *,
        aliases: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, object]] = None,
        replace: bool = False,
    ) -> None:
        normalized_key = key.strip().lower()
        alias_tuple = tuple(sorted({*(aliases or []), normalized_key}))

        with self._lock:
            if not replace and normalized_key in self._descriptors:
                raise EngineRegistrationError(f"Engine '{normalized_key}' already registered")

            descriptor = EngineDescriptor(
                key=normalized_key,
                cls=engine_cls,
                aliases=alias_tuple,
                metadata=dict(metadata or {}),
            )

            # Remove old aliases if replacing
            if replace and normalized_key in self._descriptors:
                old_descriptor = self._descriptors[normalized_key]
                for alias in old_descriptor.aliases:
                    self._alias_to_key.pop(alias, None)

            self._descriptors[normalized_key] = descriptor
            for alias in alias_tuple:
                self._alias_to_key[alias] = normalized_key

            logger.debug("Registered engine '%s' -> %s (aliases=%s)", normalized_key, engine_cls, alias_tuple)

    # ------------------------------------------------------------------
    def list(self) -> Sequence[str]:
        with self._lock:
            return sorted(self._descriptors.keys())

    def get_descriptor(self, key: str) -> EngineDescriptor:
        normalized = key.strip().lower()
        with self._lock:
            target = self._alias_to_key.get(normalized, normalized)
            descriptor = self._descriptors.get(target)
            if descriptor is None:
                raise EngineNotFoundError(f"Engine '{key}' is not registered")
            return descriptor

    def create(self, key: str, **init_kwargs: object) -> BaseInferenceEngine:
        descriptor = self.get_descriptor(key)
        engine = descriptor.cls(**init_kwargs)
        logger.info("Created engine instance for key '%s' (%s)", descriptor.key, descriptor.cls.__name__)
        return engine

    def clear(self) -> None:
        with self._lock:
            self._descriptors.clear()
            self._alias_to_key.clear()


# Global registry instance used by default
registry = EngineRegistry()


def register_engine(
    key: str,
    engine_cls: Type[EngineT],
    *,
    aliases: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, object]] = None,
    replace: bool = False,
) -> None:
    registry.register(key, engine_cls, aliases=aliases, metadata=metadata, replace=replace)


def list_engines() -> Sequence[str]:
    return registry.list()


def create_engine(key: str, **init_kwargs: object) -> BaseInferenceEngine:
    return registry.create(key, **init_kwargs)

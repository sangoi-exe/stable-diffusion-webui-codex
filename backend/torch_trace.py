from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Optional

import torch

_log = logging.getLogger("backend.trace")
_enabled: bool = False
_limit: int = 500
_events: int = 0
_stack: list[str] = []
_orig_module_to = torch.nn.Module.to


def _maybe_log(msg: str, *args: Any) -> None:
    global _events
    if not _enabled:
        return
    if _events >= _limit:
        return
    _log.debug(msg, *args)
    _events += 1


def _format_device(dev: Any) -> str:
    try:
        if hasattr(dev, "type"):
            return f"{dev.type}:{getattr(dev, 'index', 0) if getattr(dev, 'type', '')=='cuda' else ''}".rstrip(":")
        return str(dev)
    except Exception:
        return str(dev)


def _patched_module_to(self: torch.nn.Module, *args: Any, **kwargs: Any):  # type: ignore[override]
    # Log a compact move event: module class, args we care about
    if _enabled:
        dev = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        if len(args) >= 1 and dev is None:
            dev = args[0]
        if len(args) >= 2 and dtype is None:
            dtype = args[1]
        _maybe_log(
            "to(): mod=%s dev=%s dtype=%s", self.__class__.__name__, _format_device(dev), getattr(dtype, "__repr__", lambda: str(dtype))()
        )
    return _orig_module_to(self, *args, **kwargs)


def enable(section: str = "") -> None:
    global _enabled, _limit, _events
    _enabled = bool(int(os.environ.get("CODEX_TRACE_TORCH", "0") or "0"))
    try:
        _limit = int(os.environ.get("CODEX_TRACE_LIMIT", "500"))
    except Exception:
        _limit = 500
    _events = 0
    if not _enabled:
        return
    _maybe_log("trace: enable section=%s limit=%d", section, _limit)
    # Patch selectively
    torch.nn.Module.to = _patched_module_to  # type: ignore[assignment]


def disable() -> None:
    global _enabled
    if not _enabled:
        return
    # Restore
    torch.nn.Module.to = _orig_module_to  # type: ignore[assignment]
    _maybe_log("trace: disable (events=%d)", _events)
    _enabled = False


def event(what: str, **meta: Any) -> None:
    if not _enabled:
        return
    if _events >= _limit:
        return
    if meta:
        _maybe_log("event: %s %s", what, meta)
    else:
        _maybe_log("event: %s", what)


@contextmanager
def trace_section(name: str):
    enable(name)
    start = time.time()
    _stack.append(name)
    try:
        yield
    finally:
        dur = (time.time() - start) * 1000.0
        event("section_end", name=name, ms=f"{dur:.2f}")
        _stack.pop()
        disable()


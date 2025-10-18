"""Centralized logging configuration for WebUI Codex backend.

Reads level and basic settings from environment and configures root logging
once per interpreter. Defaults to DEBUG as requested.

Environment variables (settable via webui.settings.bat / webui-user.bat):
 - CODEX_LOG_LEVEL / SDWEBUI_LOG_LEVEL / WEBUI_LOG_LEVEL: logging level name
   e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL (case-insensitive). Defaults DEBUG.
 - CODEX_LOG_FILE: optional path to append logs (disabled if empty).
 - CODEX_LOG_FORMAT: optional logging format string.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_CONFIGURED = False


def _parse_level(value: Optional[str]) -> int:
    if not value:
        return logging.DEBUG
    v = value.strip().upper()
    mapping = {
        "TRACE": 5,  # custom: lower than DEBUG if someone sets it
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,
    }
    return mapping.get(v, logging.DEBUG)


def setup_logging() -> None:
    """Initialize root logger based on env vars, only once.

    - Sets root level to env-provided level (default DEBUG).
    - Adds a stderr StreamHandler with a concise, actionable format.
    - Optionally adds a file handler if CODEX_LOG_FILE is set.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Determine level
    level = _parse_level(
        os.environ.get("CODEX_LOG_LEVEL")
        or os.environ.get("SDWEBUI_LOG_LEVEL")
        or os.environ.get("WEBUI_LOG_LEVEL")
    )

    # Basic, readable format
    fmt = os.environ.get(
        "CODEX_LOG_FORMAT",
        "%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s: %(message)s",
    )
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if upstream configured something already
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(sh)

    # Ensure a dedicated handler for the 'backend' logger hierarchy so DEBUG
    # logs are not filtered by third-party handlers (e.g., uvicorn/gradio)
    codex = logging.getLogger("backend")
    codex.setLevel(level)
    # mark our handler to avoid duplicates on re-entry
    has_codex = False
    for h in codex.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "_codex", False):
            has_codex = True
            break
    if not has_codex:
        h = logging.StreamHandler(stream=sys.stderr)
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        setattr(h, "_codex", True)
        codex.addHandler(h)
    # prevent double printing via root handlers
    codex.propagate = False

    log_file = os.environ.get("CODEX_LOG_FILE")
    if log_file and not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_file)
        for h in root.handlers
    ):
        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            root.addHandler(fh)
        except Exception:
            # If file handler fails, keep stderr logging only; do not crash startup
            logging.getLogger(__name__).exception("Failed to attach file handler: %s", log_file)

    logging.getLogger(__name__).debug(
        "logging configured level=%s file=%s handlers=%d",
        logging.getLevelName(level),
        log_file or "<stderr-only>",
        len(root.handlers),
    )

    _CONFIGURED = True

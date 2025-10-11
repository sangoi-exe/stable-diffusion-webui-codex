from __future__ import annotations

import logging
import os
from typing import Optional

try:
    from colorama import init as colorama_init
except ModuleNotFoundError:  # pragma: no cover - optional during bootstrap
    colorama_init = None  # type: ignore

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None

try:
    from rich.console import Console
    from rich.logging import RichHandler
except ModuleNotFoundError:  # pragma: no cover
    Console = None  # type: ignore
    RichHandler = None  # type: ignore


if colorama_init is not None:
    colorama_init(autoreset=True)


class TqdmAwareHandler(logging.Handler):
    """Proxy handler that cooperates with tqdm progress bars."""

    def __init__(self, inner: logging.Handler):
        super().__init__()
        self.inner = inner

    def setFormatter(self, fmt: logging.Formatter) -> None:  # noqa: N802 (match logging API)
        super().setFormatter(fmt)
        self.inner.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        if tqdm is not None and getattr(tqdm, "_instances", None):
            try:
                tqdm.write(self.format(record))
                return
            except Exception:  # pragma: no cover - fall back silently
                pass
        self.inner.emit(record)


def _resolve_level(level: Optional[str]) -> int:
    if level:
        candidate = getattr(logging, level.upper(), None)
        if isinstance(candidate, int):
            return candidate
    env_level = os.environ.get("SD_WEBUI_LOG_LEVEL")
    if env_level:
        candidate = getattr(logging, env_level.upper(), None)
        if isinstance(candidate, int):
            return candidate
    return logging.INFO


def _build_handler(level: int) -> logging.Handler:
    if RichHandler is not None and Console is not None:
        console = Console(color_system="auto", soft_wrap=True, emoji=False, highlight=False)
        handler: logging.Handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=False,
            markup=False,
        )
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:  # pragma: no cover - fallback for bootstrap/install
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        )

    if tqdm is not None:
        handler = TqdmAwareHandler(handler)

    return handler


def setup_logging(loglevel: Optional[str]) -> None:
    """Configure root logger with Rich/Colorama styling."""

    level = _resolve_level(loglevel)

    root = logging.getLogger()
    if root.handlers:
        # Avoid accumulating handlers when re-launching from UI reloads.
        root.handlers.clear()

    handler = _build_handler(level)
    root.setLevel(level)
    root.addHandler(handler)

    # Ensure warnings emitted via warnings.warn use the same handler.
    logging.captureWarnings(True)

    if RichHandler is None or Console is None:
        logging.getLogger(__name__).warning(
            "Running without rich/colorama enhancements; install dependencies to enable colored logging."
        )

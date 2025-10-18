"""Backend package for WebUI Codex.

Contains service abstractions and backend helpers decoupled from FastAPI.

Also initializes logging early so all submodules share consistent verbosity.
The default level is DEBUG, overridable via env vars
CODEX_LOG_LEVEL/SDWEBUI_LOG_LEVEL/WEBUI_LOG_LEVEL or webui.settings.bat.
"""

# Configure logging once per interpreter early in import chain
try:
    from .logging_utils import setup_logging as _codex_setup_logging  # type: ignore
    _codex_setup_logging()
except Exception:
    # Never block imports; downstream modules may configure their own logging
    pass

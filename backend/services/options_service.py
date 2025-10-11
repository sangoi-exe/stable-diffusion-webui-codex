from __future__ import annotations

from modules import shared


class OptionsService:
    """Thin wrapper around options/config interactions.

    Centralizes read/write to reduce direct coupling to modules.* in routes.
    """

    def get_config(self):
        from modules.sysinfo import get_config
        return get_config()

    def set_config(self, req: dict):
        from modules.sysinfo import set_config
        set_config(req)

    def get_cmd_flags(self):
        return vars(shared.cmd_opts)


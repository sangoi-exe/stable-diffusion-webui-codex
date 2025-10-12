import os
from modules import localization, shared, scripts, util
import os
import posixpath
from modules.paths import script_path, data_path


def webpath(fn):
    return f'file={util.truncate_path(fn)}?{os.path.getmtime(fn)}'


def _parse_js_allowlist():
    """Return a set of basenames allowed for JS injection.

    Controls:
    - GRADIO_JS_ALLOWLIST unset: None (disabled; inject all as before)
    - GRADIO_JS_ALLOWLIST in ("", "1", "true", "yes", "on", "auto"): curated default allowlist
    - otherwise: comma-separated basenames
    """
    env = os.getenv("GRADIO_JS_ALLOWLIST")
    if env is None:
        return None  # keep legacy behavior

    truthy = {"", "1", "true", "yes", "on", "auto"}
    if env.strip().lower() in truthy:
        # Curated default: keep core behavior; exclude fragile or replaced scripts.
        return set([
            "script.js",
            "ui.js",
            "progressbar.js",
            "localization.js",
            "imageviewer.js",
            "resizeHandle.js",
            "hints.js",
            "contextMenus.js",
            "extraNetworks.js",
            "imageMaskFix.js",
            "dragdrop.js",
            "notification.js",
            "edit-attention.js",
            "edit-order.js",
            "generationParams.js",
            "aspectRatioOverlay.js",
            "imageviewerGamepad.js",
            "profilerVisualization.js",
            "textualInversion.js",
        ])

    # Custom list
    return set(x.strip() for x in env.split(",") if x.strip())


def _parse_js_denylist():
    """Return a set of basenames to exclude from JS injection regardless of allowlist.

    - Default (unset): deny a small set of fragile legacy scripts that are now replaced by Python logic.
    - Override with GRADIO_JS_DENYLIST to customize (comma-separated basenames). To disable defaults, set to "none".
    """
    # Aggressively disable legacy JS by default; Python-side events replace these hooks
    default_deny = {"settings.js", "gradio.js", "ui.js", "token-counters.js", "inputAccordion.js"}
    env = os.getenv("GRADIO_JS_DENYLIST")
    if env is None:
        return default_deny
    env = env.strip()
    if env.lower() == "none":
        return set()
    if not env:
        return default_deny
    return set(x.strip() for x in env.split(",") if x.strip())


def javascript_html():
    # Bootstrap minimal stubs before any other script to avoid ReferenceErrors
    bootstrap = (
        "<script type=\"text/javascript\">(function(){\n"
        "window.onUiUpdate=window.onUiUpdate||function(cb){(window._uiUpdQ=window._uiUpdQ||[]).push(cb)};\n"
        "window.onAfterUiUpdate=window.onAfterUiUpdate||function(cb){(window._uiAfterQ=window._uiAfterQ||[]).push(cb)};\n"
        "window.onUiLoaded=window.onUiLoaded||function(cb){(window._uiLoadQ=window._uiLoadQ||[]).push(cb)};\n"
        "})();</script>\n"
    )
    # Ensure localization is in `window` before scripts
    head = bootstrap + f'<script type="text/javascript">{localization.localization_js(shared.opts.localization)}</script>\n'

    script_js = os.path.join(script_path, "script.js")
    head += f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'

    allow = _parse_js_allowlist()
    deny = _parse_js_denylist()

    def _is_allowed(p: str) -> bool:
        if allow is None:
            base = posixpath.basename(p.replace("\\", "/"))
            return base not in deny
        base = posixpath.basename(p.replace("\\", "/"))
        return base in allow and base not in deny

    for script in scripts.list_scripts("javascript", ".js"):
        if _is_allowed(script.path):
            head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'

    for script in scripts.list_scripts("javascript", ".mjs"):
        if _is_allowed(script.path):
            head += f'<script type="module" src="{webpath(script.path)}"></script>\n'

    if shared.cmd_opts.theme:
        head += f'<script type="text/javascript">set_theme(\"{shared.cmd_opts.theme}\");</script>\n'

    return head


def css_html():
    head = ""

    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    for cssfile in scripts.list_files_with_name("style.css"):
        head += stylesheet(cssfile)

    user_css = os.path.join(data_path, "user.css")
    if os.path.exists(user_css):
        head += stylesheet(user_css)

    from modules.shared_gradio_themes import resolve_var
    light = resolve_var('background_fill_primary')
    dark = resolve_var('background_fill_primary_dark')
    head += f'<style>html {{ background-color: {light}; }} @media (prefers-color-scheme: dark) {{ html {{background-color:  {dark}; }} }}</style>'

    return head


def head_includes():
    """Return HTML to include in <head>: localization preload, core and extension JS, CSS, and meta.

    Used by Blocks(head=...) to avoid TemplateResponse monkeypatching (Gradio 5-friendly).
    """
    return css_html() + javascript_html() + '<meta name="referrer" content="no-referrer"/>'


def reload_javascript():
    """Legacy no-op kept for compatibility.

    Previously monkeypatched TemplateResponse. With Gradio 5 we inject via Blocks(head=...).
    """
    return None

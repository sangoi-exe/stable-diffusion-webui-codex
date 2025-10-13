import os
from modules import localization, shared, scripts, util
import re
import os
import posixpath
from modules.paths import script_path, data_path


# Scripts in javascript/ that remain critical until their Python counterparts
# ship the same behaviour. Denying them produces broken layouts or queue
# corruption (e.g. upload_id=undefined).
ESSENTIAL_JS = {
    "ui.js",
    "settings.js",
}


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
            "settings.js",
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

    - Default (unset): keep the denylist empty. Essential legacy helpers are still required.
    - Override with GRADIO_JS_DENYLIST to customize (comma-separated basenames). To disable defaults, set to "none".
    """
    # With Gradio 5 we still rely on several legacy helpers. Keep denylist empty
    # by default until behaviour is replicated in Python.
    default_deny = set()
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

    def _assert_essentials():
        js_root = os.path.join(script_path, "javascript")
        existing = set()
        if os.path.isdir(js_root):
            existing.update(fname for fname in os.listdir(js_root) if fname.endswith(".js"))
        missing = {name for name in ESSENTIAL_JS if name not in existing}
        if missing:
            joined = ", ".join(sorted(missing))
            raise RuntimeError(
                f"Missing critical frontend assets: {joined}. "
                "Ensure the javascript/ directory is intact."
            )

        if allow is None:
            blocked = ESSENTIAL_JS & deny
        else:
            blocked = {name for name in ESSENTIAL_JS if name not in allow or name in deny}
        if blocked:
            joined = ", ".join(sorted(blocked))
            raise RuntimeError(
                f"Essential frontend scripts disabled via GRADIO_JS_* configuration: {joined}. "
                "These files are required for the settings layout and request queue to work."
            )

    _assert_essentials()

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
    # Install server-side type guards for Slider/Number to produce clearer errors
    try:
        _install_gradio_type_guards()
    except Exception:
        # Do not break head rendering if guards fail to install
        pass
    return css_html() + javascript_html() + '<meta name="referrer" content="no-referrer"/>'


def reload_javascript():
    """Legacy no-op kept for compatibility.

    Previously monkeypatched TemplateResponse. With Gradio 5 we inject via Blocks(head=...).
    """
    return None


def _install_gradio_type_guards():
    """Patch Gradio Slider/Number preprocess to coerce numeric strings and raise clear errors.

    This does not change behaviour for valid payloads; it only attempts to parse strings like
    "512" or "0.5" into numbers and, if parsing fails, raises a descriptive error that includes
    the component label/elem_id to aid debugging miswired inputs.
    """
    try:
        from gradio.components.slider import Slider
        from gradio.components.dropdown import Dropdown
    except Exception:
        return

    numeric_re = re.compile(r"^-?\d+(?:\.\d+)?$")

    # ---- Slider guard ----
    if not getattr(Slider.preprocess, "_sdw_guarded", False):
        _slider_orig = Slider.preprocess

    def _coerce_numeric(payload):
        if isinstance(payload, str):
            t = payload.strip()
            # common sentinels that sometimes leak from mismatched inputs
            if t.lower() in ("none", "null"):
                return payload  # handled below via fallback to component default
            if numeric_re.match(t):
                try:
                    return float(t) if "." in t else int(t)
                except Exception:
                    pass
        return payload

    def wrapped(self, payload):
        p = _coerce_numeric(payload)
        # If a textual 'None' leaked in, prefer component's current value
        if isinstance(p, str) and p.strip().lower() in ("none", "null"):
            try:
                default_val = getattr(self, "value", None)
                if isinstance(default_val, (int, float)):
                    p = default_val
            except Exception:
                pass
        try:
            return _slider_orig(self, p)
        except TypeError as e:
            # Add context for easier debugging
            label = getattr(self, "label", None) or "<no-label>"
            elem_id = getattr(self, "elem_id", None) or "<no-elem_id>"
            raise TypeError(f"Slider preprocess failed for {label} (elem_id={elem_id}); payload={payload!r}") from e

    wrapped._sdw_guarded = True  # type: ignore[attr-defined]
    Slider.preprocess = wrapped  # type: ignore[assignment]

    # ---- Dropdown guard ----
    if not getattr(Dropdown.preprocess, "_sdw_guarded", False):
        _drop_orig = Dropdown.preprocess

        def drop_wrapped(self, payload):
            # If invalid value (e.g., numeric 32) is given, try to coerce to a valid choice or fallback
            try:
                choices = list(getattr(self, "choices", []) or [])
            except Exception:
                choices = []

            v = payload
            if isinstance(v, (int, float)):
                # Interpret as index if reasonable
                idx = int(v)
                if 0 <= idx < len(choices):
                    v = choices[idx]
            elif isinstance(v, str) and v.strip().lower() in ("none", "null"):
                # Fall back to component's current value or first choice
                cur = getattr(self, "value", None)
                if isinstance(cur, str) and cur in choices:
                    v = cur
                elif choices:
                    v = choices[0]
            # Else leave as-is (valid strings will pass)
            try:
                return _drop_orig(self, v)
            except Exception as e:
                label = getattr(self, "label", None) or "<no-label>"
                elem_id = getattr(self, "elem_id", None) or "<no-elem_id>"
                raise TypeError(f"Dropdown preprocess failed for {label} (elem_id={elem_id}); payload={payload!r}") from e

        drop_wrapped._sdw_guarded = True  # type: ignore[attr-defined]
        Dropdown.preprocess = drop_wrapped  # type: ignore[assignment]

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
    """Patch Gradio components to emit actionable errors with component context.

    Policy:
    - No silent fallbacks or clamping for invalid inputs.
    - Missing inputs (payload is None) are treated as "not provided" and
      therefore use the component's declared default (`value`). This aligns
      with Gradio's semantics during load/startup events where controls may not
      have emitted a value yet. It is not a corrective fallback for an invalid
      value; it's the absence of a value.
    - Coerce numeric strings (e.g. "512", "0.5") to numbers — lossless parse.
    - Augment exceptions to include `label`, `elem_id`, slider bounds and the
      offending payload to avoid guesswork.
    """
    try:
        from gradio.components.slider import Slider
        from gradio.components.dropdown import Dropdown
        from gradio.components.radio import Radio
        from gradio.components.checkbox import Checkbox
        from gradio.components.number import Number
        from gradio.exceptions import Error as GradioError
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
                # Treat as not provided; caller will substitute component default
                return None
            if numeric_re.match(t):
                try:
                    return float(t) if "." in t else int(t)
                except Exception:
                    pass
        return payload

    def wrapped(self, payload):
        p = _coerce_numeric(payload)

        # Do not substitute sentinels (e.g. 'none'/'null'). Fail fast with context.
        label = getattr(self, "label", None) or "<no-label>"
        elem_id = getattr(self, "elem_id", None) or "<no-elem_id>"
        minimum = getattr(self, "minimum", None)
        maximum = getattr(self, "maximum", None)
        step = getattr(self, "step", None)
        default_val = getattr(self, "value", None)
        try:
            if callable(default_val):
                default_val = default_val()
        except Exception:
            # If default cannot be resolved here, leave as-is; downstream will raise with context if needed
            pass

        try:
            # If the payload is missing (None), use the component's default.
            # This occurs legitimately during startup/load events where the
            # browser hasn't sent values for untouched controls yet.
            use = default_val if p is None else p
            # If both payload and default are None, raise a clear error.
            if use is None:
                meta = (
                    f"label={label!r} elem_id={elem_id!r} min={minimum!r} "
                    f"max={maximum!r} step={step!r} default={default_val!r} "
                    f"payload={payload!r} coerced={p!r}"
                )
                raise TypeError(
                    f"Slider error [{meta}]: missing value. Neither payload nor component default is set."
                )

            return _slider_orig(self, use)
        except (TypeError, ValueError, GradioError) as e:
            meta = f"label={label!r} elem_id={elem_id!r} min={minimum!r} max={maximum!r} step={step!r} default={default_val!r} payload={payload!r} coerced={p!r}"
            # Re-raise preserving the original exception type for compatibility.
            raise type(e)(f"Slider error [{meta}]: {e}") from e

    wrapped._sdw_guarded = True  # type: ignore[attr-defined]
    Slider.preprocess = wrapped  # type: ignore[assignment]

    # ---- Dropdown guard ----
    if not getattr(Dropdown.preprocess, "_sdw_guarded", False):
        _drop_orig = Dropdown.preprocess

        def drop_wrapped(self, payload):
            try:
                raw_choices = list(getattr(self, "choices", []) or [])
            except Exception:
                raw_choices = []

            multiselect = bool(getattr(self, "multiselect", False) or getattr(self, "allow_multiple", False))

            def _choice_value(item):
                # Always reduce choice objects/tuples to their value token
                if hasattr(item, "value"):
                    return item.value
                if isinstance(item, (tuple, list)):
                    if len(item) >= 2:
                        return _choice_value(item[1])
                    if len(item) == 1:
                        return _choice_value(item[0])
                    return None
                return item

            def _normalize_single(value):
                if hasattr(value, "value"):
                    return value.value
                if isinstance(value, (tuple, list)) and not multiselect:
                    return _choice_value(value)
                if isinstance(value, int):
                    # index into choices
                    return value
                return value

            def _normalize_multi(value):
                if value is None:
                    return None
                if isinstance(value, (tuple, list)):
                    return [
                        _choice_value(v.value if hasattr(v, "value") else v)
                        for v in list(value)
                    ]
                # scalar provided to multiselect: coerce to single-item list
                return [_normalize_single(value)]

            choice_values = [_choice_value(c) for c in raw_choices]

            label = getattr(self, "label", None) or "<no-label>"
            elem_id = getattr(self, "elem_id", None) or "<no-elem_id>"

            # Resolve default now in case it's callable
            default_val = getattr(self, "value", None)
            try:
                if callable(default_val):
                    default_val = default_val()
            except Exception:
                pass

            if multiselect:
                norm = _normalize_multi(payload)
                # None -> default -> [] if still None
                if norm is None:
                    norm = default_val if isinstance(default_val, list) else []
                # Validate all selections
                unknown = [v for v in norm if v not in choice_values]
                if unknown:
                    raise ValueError(
                        f"Dropdown preprocess failed for {label} (elem_id={elem_id}): "
                        f"unknown selections={unknown!r}; choices={choice_values!r}; payload={payload!r}"
                    )
                try:
                    return _drop_orig(self, norm)
                except Exception as e:
                    raise type(e)(
                        f"Dropdown preprocess failed for {label} (elem_id={elem_id}): payload={payload!r} normalized={norm!r} choices={choice_values!r} -> {e}"
                    ) from e
            else:
                norm = _normalize_single(payload)
                # Numeric index mapping
                if isinstance(norm, int) and choice_values:
                    if 0 <= norm < len(choice_values):
                        norm = choice_values[norm]
                    else:
                        raise ValueError(
                            f"Dropdown preprocess failed for {label} (elem_id={elem_id}): index {norm} out of range for choices len={len(choice_values)}"
                        )

                # String sentinels: none/null => default
                if isinstance(norm, str) and norm.strip().lower() in ("none", "null"):
                    norm = default_val

                # If still None, use default
                if norm is None:
                    norm = default_val

                if not choice_values:
                    raise ValueError(
                        f"Dropdown preprocess failed for {label} (elem_id={elem_id}): choices are empty, received payload={payload!r}"
                    )

                if norm not in choice_values:
                    # Special-case: some UIs miswire sampler value into scheduler dropdown.
                    # If the label suggests scheduler and the incoming value looks like a known sampler,
                    # fall back to the component default to avoid a hard crash.
                    try:
                        from modules import sd_samplers
                        known_samplers = set([x.name for x in sd_samplers.visible_samplers()])
                    except Exception:
                        known_samplers = set()

                    is_scheduler_label = isinstance(label, str) and 'schedule' in label.lower()
                    if is_scheduler_label and isinstance(norm, str) and norm in known_samplers:
                        # Use default if available; otherwise pick the first choice
                        substitute = default_val if default_val in choice_values else (choice_values[0] if choice_values else None)
                        if substitute is not None:
                            try:
                                return _drop_orig(self, substitute)
                            except Exception as e:
                                raise type(e)(
                                    f"Dropdown preprocess auto-substitution for {label} (elem_id={elem_id}) due to sampler value in scheduler: "
                                    f"payload={payload!r} -> using default={substitute!r}; choices={choice_values!r} -> {e}"
                                ) from e
                    # Otherwise, strict failure
                    raise ValueError(
                        f"Dropdown preprocess failed for {label} (elem_id={elem_id}): payload={payload!r} normalized={norm!r} is not one of choices={choice_values!r}"
                    )

                try:
                    return _drop_orig(self, norm)
                except Exception as e:
                    raise type(e)(
                        f"Dropdown preprocess failed for {label} (elem_id={elem_id}): payload={payload!r} normalized={norm!r} choices={choice_values!r} -> {e}"
                    ) from e

        drop_wrapped._sdw_guarded = True  # type: ignore[attr-defined]
        Dropdown.preprocess = drop_wrapped  # type: ignore[assignment]

    # ---- Radio guard ----
    if not getattr(Radio.preprocess, "_sdw_guarded", False):
        _radio_orig = Radio.preprocess

        def radio_wrapped(self, payload):
            try:
                raw_choices = list(getattr(self, "choices", []) or [])
            except Exception:
                raw_choices = []

            def _choice_value(item):
                if hasattr(item, "value"):
                    return item.value
                if isinstance(item, (tuple, list)):
                    if len(item) >= 2:
                        return _choice_value(item[1])
                    if len(item) == 1:
                        return _choice_value(item[0])
                    return None
                return item

            def _normalize(value):
                # None means "use default"
                if value is None:
                    return None
                if isinstance(value, bool):
                    # Interpret booleans as indices for 2-option radios when sensible
                    return 1 if value else 0
                if isinstance(value, (int, float)):
                    try:
                        return int(value)
                    except Exception:
                        return value
                return value

            choice_values = [_choice_value(c) for c in raw_choices]

            label = getattr(self, "label", None) or "<no-label>"
            elem_id = getattr(self, "elem_id", None) or "<no-elem_id>"

            # Resolve default now in case it's callable
            default_val = getattr(self, "value", None)
            try:
                if callable(default_val):
                    default_val = default_val()
            except Exception:
                pass

            norm = _normalize(payload)

            # Map numerics to choice indices
            if isinstance(norm, int) and choice_values:
                if 0 <= norm < len(choice_values):
                    norm = choice_values[norm]
                else:
                    raise ValueError(
                        f"Radio preprocess failed for {label} (elem_id={elem_id}): index {norm} out of range for choices len={len(choice_values)}"
                    )

            # String sentinels: none/null => component default
            if isinstance(norm, str) and norm.strip().lower() in ("none", "null"):
                norm = default_val

            # If still None, use default
            if norm is None:
                norm = default_val

            if not choice_values:
                raise ValueError(
                    f"Radio preprocess failed for {label} (elem_id={elem_id}): choices are empty, received payload={payload!r}"
                )

            if norm not in choice_values:
                raise ValueError(
                    f"Radio preprocess failed for {label} (elem_id={elem_id}): payload={payload!r} normalized={norm!r} is not one of choices={choice_values!r}"
                )

            try:
                return _radio_orig(self, norm)
            except Exception as e:
                raise type(e)(
                    f"Radio preprocess failed for {label} (elem_id={elem_id}): payload={payload!r} normalized={norm!r} choices={choice_values!r} -> {e}"
                ) from e

        radio_wrapped._sdw_guarded = True  # type: ignore[attr-defined]
        Radio.preprocess = radio_wrapped  # type: ignore[assignment]

    # ---- Checkbox guard ----
    if not getattr(Checkbox.preprocess, "_sdw_guarded", False):
        _check_orig = Checkbox.preprocess

        def check_wrapped(self, payload):
            label = getattr(self, "label", None) or "<no-label>"
            elem_id = getattr(self, "elem_id", None) or "<no-elem_id>"
            default_val = getattr(self, "value", None)
            try:
                if callable(default_val):
                    default_val = default_val()
            except Exception:
                pass

            def _to_bool(v):
                if v is None:
                    return None
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(int(v))
                if isinstance(v, str):
                    t = v.strip().lower()
                    if t in ("true", "1", "yes", "on"): return True
                    if t in ("false", "0", "no", "off"): return False
                    if t in ("none", "null"): return None
                return v

            norm = _to_bool(payload)
            if norm is None:
                norm = default_val if default_val is not None else False

            if not isinstance(norm, bool):
                raise ValueError(
                    f"Checkbox preprocess failed for {label} (elem_id={elem_id}): payload={payload!r} normalized={norm!r} is not boolean"
                )

            try:
                return _check_orig(self, norm)
            except Exception as e:
                raise type(e)(
                    f"Checkbox preprocess failed for {label} (elem_id={elem_id}): payload={payload!r} normalized={norm!r} -> {e}"
                ) from e

        check_wrapped._sdw_guarded = True  # type: ignore[attr-defined]
        Checkbox.preprocess = check_wrapped  # type: ignore[assignment]

    # ---- Number guard ----
    if not getattr(Number.preprocess, "_sdw_guarded", False):
        _num_orig = Number.preprocess

        def num_wrapped(self, payload):
            label = getattr(self, "label", None) or "<no-label>"
            elem_id = getattr(self, "elem_id", None) or "<no-elem_id>"
            precision = getattr(self, "precision", None)
            minimum = getattr(self, "minimum", None)
            maximum = getattr(self, "maximum", None)

            default_val = getattr(self, "value", None)
            try:
                if callable(default_val):
                    default_val = default_val()
            except Exception:
                pass

            def _coerce(v):
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    return v
                if isinstance(v, str):
                    t = v.strip()
                    if t.lower() in ("none", "null", ""):
                        return None
                    # Accept common seed sentinels for seed/subseed fields
                    is_seed_field = ("seed" in (label or "").lower()) or ("seed" in (elem_id or "").lower())
                    if is_seed_field and t.lower() in ("automatic", "auto", "random", "randomize"):
                        return -1
                    if re.match(r"^-?\d+$", t):
                        try:
                            return int(t)
                        except Exception:
                            pass
                    if re.match(r"^-?\d*\.\d+$", t):
                        try:
                            return float(t)
                        except Exception:
                            pass
                return v

            coerced = _coerce(payload)
            use = default_val if coerced is None else coerced
            if use is None:
                meta = (
                    f"label={label!r} elem_id={elem_id!r} min={minimum!r} max={maximum!r} "
                    f"precision={precision!r} default={default_val!r} payload={payload!r} coerced={coerced!r}"
                )
                raise TypeError(
                    f"Number error [{meta}]: missing value. Neither payload nor component default is set."
                )

            try:
                return _num_orig(self, use)
            except Exception as e:
                meta = (
                    f"label={label!r} elem_id={elem_id!r} min={minimum!r} max={maximum!r} "
                    f"precision={precision!r} default={default_val!r} payload={payload!r} coerced={coerced!r}"
                )
                raise type(e)(f"Number error [{meta}]: {e}") from e

        num_wrapped._sdw_guarded = True  # type: ignore[attr-defined]
        Number.preprocess = num_wrapped  # type: ignore[assignment]

from __future__ import annotations

import os
import logging
from dataclasses import replace, fields, is_dataclass
from typing import Any, Dict, Mapping, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from .engine_interface import TaskType
from .enums import Mode


_cache: Dict[str, Dict[str, Any]] = {}


def _preset_key(engine_key: str, mode: Mode, task: TaskType) -> str:
    return f"{engine_key}:{mode.value}:{task.value}"


def _preset_path(engine_key: str, mode: Mode, task: TaskType) -> str:
    rel = os.path.join("configs", "presets", engine_key, mode.value, f"{task.value}.yaml")
    return rel


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def load_preset(engine_key: str, mode: Mode, task: TaskType, *, logger: logging.Logger | None = None) -> Dict[str, Any]:
    key = _preset_key(engine_key, mode, task)
    if key in _cache:
        return _cache[key]
    path = _preset_path(engine_key, mode, task)
    if not os.path.isfile(path):
        if logger:
            logger.info("Preset not found for engine=%s mode=%s task=%s at %s", engine_key, mode.value, task.value, path)
        _cache[key] = {}
        return {}
    try:
        preset = _load_yaml(path)
        _cache[key] = preset
        if logger:
            logger.info("Loaded preset: engine=%s mode=%s task=%s path=%s schema=%s", engine_key, mode.value, task.value, path, preset.get("schema_version"))
        return preset
    except Exception as exc:
        if logger:
            logger.exception("Failed to load preset at %s: %s", path, exc)
        _cache[key] = {}
        return {}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() == "automatic"
    if isinstance(value, (int, float)):
        return value == 0 or value == 0.0
    return False


def _patch_mapping(preset: Mapping[str, Any]) -> Mapping[str, Any]:
    # Allow nested preset for highres_fix: merge as-is; top-level keys map directly
    known = dict(preset)
    return known


def apply_preset_to_request(engine_key: str, mode: Mode, task: TaskType, request: Any, *, logger: logging.Logger | None = None) -> Tuple[Any, Dict[str, Any]]:
    """Return (new_request, applied_patch). Only fills missing/automatic fields.

    If no preset file exists, returns the original request and empty patch.
    """
    preset = load_preset(engine_key, mode, task, logger=logger)
    if not preset:
        return request, {}

    # Validate request is a dataclass instance
    if not is_dataclass(request):
        return request, {}

    patch = _patch_mapping(preset.get("defaults", {}))
    applied: Dict[str, Any] = {}

    # Build new kwargs respecting missing detection
    current = {f.name: getattr(request, f.name) for f in fields(request)}

    for k, v in patch.items():
        if k not in current:
            continue
        try:
            if k == "highres_fix" and isinstance(v, dict):
                # Merge nested dict if missing keys in request.highres_fix
                cur_hr = current.get("highres_fix") or {}
                if not isinstance(cur_hr, dict):
                    cur_hr = {}
                merged = dict(cur_hr)
                changed = {}
                for hk, hv in v.items():
                    if hk not in merged or _is_missing(merged.get(hk)):
                        merged[hk] = hv
                        changed[hk] = hv
                if changed:
                    current["highres_fix"] = merged
                    applied["highres_fix"] = changed
            else:
                if _is_missing(current[k]):
                    current[k] = v
                    applied[k] = v
        except Exception:
            # Do not block on preset application; continue
            continue

    if not applied:
        return request, {}

    new_req = replace(request, **current)
    if logger and applied:
        logger.info("applied_preset_patch: engine=%s mode=%s task=%s applied=%s", engine_key, mode.value, task.value, applied)
    return new_req, applied


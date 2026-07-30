"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model and asset inventory API routes.
Exposes checkpoints, inventories (sync + async refresh task start), path scan/add helpers (with file size metadata), executable sampler/scheduler
surfaces, embeddings, and engine capabilities.
Inventory payloads include first-class IP-Adapter model and image-encoder lists from the dedicated roots in `apps/paths.json`.
Capability surfaces include semantic-engine asset contracts (owner-resolved from canonical engine ids), backend-owned exact-engine inpaint-mode maps,
and backend-owned dependency checks so the UI can enforce sha-only external asset selection and readiness gating deterministically. Also provides prompt token-counting
(`/api/models/prompt-token-count`) using vendored offline tokenizers for supported prompt-token engine ids, including FLUX.2 Klein 4B, Z-Image L2P, LTX2, exact WAN ids, and
Anima runtime-equivalent Qwen+T5 prompt tokenization/max-length checks with the same family-owned offline tokenizer rules used by the runtime. Raw `/api/samplers`
inventory keeps unsupported rows visible with non-executable metadata (`supported=false`, `default_scheduler=null`, `allowed_schedulers=[]`),
while supported rows must resolve complete executable registry metadata.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for model/inventory endpoints.
- `_count_anima_tokens` (function): Counts Anima prompt tokens with runtime-equivalent preprocessing and fail-loud max-length checks.
- `_count_prompt_tokens` (function): Returns tokenizer-accurate prompt token counts for supported prompt-token engine ids.
- `_sanitize_model_path_input` (function): Sanitizes incoming model path strings (quotes/slashes/whitespace normalization).
- `_normalize_library_kind` (function): Validates `checkpoint|vae|text_encoder` path-library kinds.
- `_kind_for_library_key` (function): Resolves library kind from a paths.json key suffix (`_ckpt|_vae|_tenc`).
- `_allowed_exts_for_kind` (function): Returns allowed model file extensions per path-library kind.
- `_is_supported_library_file` (function): Validates extension/blacklist policy for scan/add candidate files.
- `_paths_config_path` (function): Returns `apps/paths.json` location for path-library mutations.
- `_load_paths_config_for_mutation` (function): Loads and validates paths config payload for mutable operations.
- `_save_paths_config` (function): Persists paths config updates (fail-loud).
- `_resolve_paths_config_entry_path` (function): Resolves one paths config entry to an absolute filesystem path.
- `_paths_config_entry_for_file` (function): Converts absolute file paths into paths.json entry semantics.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import asyncio
import logging
import math
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException, Query

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.interfaces.api.public_errors import build_public_task_error
from apps.backend.interfaces.api.task_registry import TaskEntry, register_task
from apps.backend.interfaces.api.json_store import _load_json, _save_json
from apps.backend.runtime.sampling import SAMPLER_OPTIONS, SCHEDULER_OPTIONS
from apps.backend.interfaces.api.path_utils import _normalize_inventory_for_api
from apps.backend.interfaces.api.serializers import _serialize_checkpoint
from apps.backend.interfaces.api.file_metadata import read_file_metadata

_REPO_ROOT = Path(__file__).resolve().parents[5]
_HF_ROOT = _REPO_ROOT / "apps/backend/huggingface"

_PROMPT_TOKENIZER_DIRS: Dict[str, Path] = {
    "sd15": _HF_ROOT / "runwayml/stable-diffusion-v1-5/tokenizer",
    "sdxl": _HF_ROOT / "stabilityai/stable-diffusion-xl-base-1.0/tokenizer",
    "flux1": _HF_ROOT / "black-forest-labs/FLUX.1-dev/tokenizer_2",
    "flux2": _HF_ROOT / "black-forest-labs/FLUX.2-klein-4B/tokenizer",
    "ltx2": _HF_ROOT / "Lightricks/LTX-2/tokenizer",
    "chroma": _HF_ROOT / "Chroma/tokenizer",
    "zimage": _HF_ROOT / "Tongyi-MAI/Z-Image/tokenizer",
    "wan": _HF_ROOT / "Wan-AI/Wan2.2-T2V-A14B-Diffusers/tokenizer",
    "anima_qwen": _HF_ROOT / "circlestone-labs/Anima/qwen25_tokenizer",
    "anima_t5": _HF_ROOT / "circlestone-labs/Anima/t5_tokenizer",
}

_ENGINE_TOKENIZER_KEY: Dict[str, str] = {
    "sd15": "sd15",
    "sd20": "sd15",
    "sdxl": "sdxl",
    "flux1": "flux1",
    "flux1_kontext": "flux1",
    "flux2": "flux2",
    "ltx2": "ltx2",
    "chroma": "chroma",
    "flux1_chroma": "chroma",
    "zimage": "zimage",
    "zimage_l2p": "zimage",
    "anima": "anima",
    "wan22_5b": "wan",
    "wan22_14b": "wan",
    "wan22_14b_animate": "wan",
}

_LIBRARY_KIND_CHECKPOINT = "checkpoint"
_LIBRARY_KIND_VAE = "vae"
_LIBRARY_KIND_TEXT_ENCODER = "text_encoder"
_VALID_LIBRARY_KINDS = frozenset(
    {
        _LIBRARY_KIND_CHECKPOINT,
        _LIBRARY_KIND_VAE,
        _LIBRARY_KIND_TEXT_ENCODER,
    }
)
_CHECKPOINT_SCAN_EXTS = frozenset({".ckpt", ".safetensor", ".safetensors", ".pt", ".pth", ".bin", ".gguf"})
_CHECKPOINT_BLACKLIST_SUFFIXES = frozenset({".vae.ckpt", ".vae.safetensor", ".vae.safetensors", ".vae.pt", ".vae.pth", ".vae.bin"})
_VAE_SCAN_EXTS = frozenset({".safetensor", ".safetensors", ".pt", ".bin"})
_TENC_SCAN_EXTS = frozenset({".safetensor", ".safetensors", ".pt", ".bin", ".gguf"})
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\\\]")


@lru_cache(maxsize=32)
def _load_tokenizer(tokenizer_dir: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)


@lru_cache(maxsize=4)
def _load_anima_qwen_tokenizer(tokenizer_dir: str) -> Any:
    from apps.backend.runtime.families.anima.text_encoder import load_anima_qwen_tokenizer

    return load_anima_qwen_tokenizer(tokenizer_dir)


@lru_cache(maxsize=4)
def _load_anima_t5_tokenizer(tokenizer_dir: str) -> Any:
    from apps.backend.runtime.families.anima.text_encoder import load_anima_t5_tokenizer

    return load_anima_t5_tokenizer(tokenizer_dir)


def _tokenize_len(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer([prompt], truncation=False, add_special_tokens=False, verbose=False)
    ids = encoded.get("input_ids")
    if not (isinstance(ids, list) and ids and isinstance(ids[0], list)):
        raise RuntimeError("Prompt tokenizer returned invalid 'input_ids' payload.")
    return len(ids[0])


def _resolve_tokenizer_path(key: str) -> Path:
    candidate = _PROMPT_TOKENIZER_DIRS.get(key)
    if candidate is None:
        raise RuntimeError(f"Unsupported prompt tokenizer key '{key}'.")
    if not candidate.exists():
        raise RuntimeError(
            f"Prompt tokenizer directory missing for '{key}': {candidate}. "
            "Expected vendored Hugging Face assets under apps/backend/huggingface."
        )
    return candidate


def _count_anima_tokens(prompt: str) -> int:
    from apps.backend.runtime.families.anima.text_encoder import count_anima_prompt_tokens

    qwen_tok = _load_anima_qwen_tokenizer(str(_resolve_tokenizer_path("anima_qwen")))
    t5_tok = _load_anima_t5_tokenizer(str(_resolve_tokenizer_path("anima_t5")))
    return count_anima_prompt_tokens(
        str(prompt or ""),
        qwen_tokenizer=qwen_tok,
        t5_tokenizer=t5_tok,
    )


def _count_prompt_tokens(engine: str, prompt: str) -> int:
    normalized = str(engine or "").strip().lower()
    if not normalized:
        raise RuntimeError("Prompt token count requires a non-empty engine id.")
    tokenizer_key = _ENGINE_TOKENIZER_KEY.get(normalized)
    if tokenizer_key is None:
        raise RuntimeError(
            f"Unsupported engine '{engine}' for prompt token count. "
            f"Supported: {', '.join(sorted(_ENGINE_TOKENIZER_KEY.keys()))}."
        )
    if not prompt and tokenizer_key != "anima":
        return 0

    if tokenizer_key == "anima":
        return _count_anima_tokens(prompt)

    tokenizer = _load_tokenizer(str(_resolve_tokenizer_path(tokenizer_key)))
    return _tokenize_len(tokenizer, prompt)


def _sanitize_model_path_input(raw: object) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""

    while len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()

    value = value.replace("\\", "/")
    has_unc_prefix = value.startswith("//")
    drive_prefix = value[:2] if _WINDOWS_DRIVE_RE.match(value) else ""

    if drive_prefix:
        rest = value[2:]
        rest = re.sub(r"/{2,}", "/", rest)
        if rest and not rest.startswith("/"):
            rest = f"/{rest}"
        value = f"{drive_prefix}{rest}"
    else:
        value = re.sub(r"/{2,}", "/", value)
        if has_unc_prefix:
            value = f"//{value.lstrip('/')}"
    if len(value) > 1 and value.endswith("/") and not re.fullmatch(r"[A-Za-z]:/", value):
        value = value.rstrip("/")
    return value.strip()


def _normalize_library_kind(raw: object) -> str:
    kind = str(raw or "").strip().lower()
    if kind not in _VALID_LIBRARY_KINDS:
        allowed = ", ".join(sorted(_VALID_LIBRARY_KINDS))
        raise HTTPException(status_code=400, detail=f"invalid model library kind {kind!r}; expected one of: {allowed}")
    return kind


def _kind_for_library_key(key: str) -> str:
    normalized = str(key or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="library key is required")
    if normalized.endswith("_ckpt"):
        return _LIBRARY_KIND_CHECKPOINT
    if normalized.endswith("_vae"):
        return _LIBRARY_KIND_VAE
    if normalized.endswith("_tenc"):
        return _LIBRARY_KIND_TEXT_ENCODER
    raise HTTPException(
        status_code=400,
        detail=(
            f"unsupported library key {normalized!r}; expected a paths.json key ending with "
            "'_ckpt', '_vae', or '_tenc'"
        ),
    )


def _allowed_exts_for_kind(kind: str) -> frozenset[str]:
    if kind == _LIBRARY_KIND_CHECKPOINT:
        return _CHECKPOINT_SCAN_EXTS
    if kind == _LIBRARY_KIND_VAE:
        return _VAE_SCAN_EXTS
    if kind == _LIBRARY_KIND_TEXT_ENCODER:
        return _TENC_SCAN_EXTS
    raise RuntimeError(f"unsupported library kind: {kind!r}")


def _is_supported_library_file(path: Path, *, kind: str) -> bool:
    suffix = path.suffix.lower()
    if suffix not in _allowed_exts_for_kind(kind):
        return False
    if kind == _LIBRARY_KIND_CHECKPOINT:
        lower_name = path.name.lower()
        if any(lower_name.endswith(suf) for suf in _CHECKPOINT_BLACKLIST_SUFFIXES):
            return False
    return True


def _paths_config_path() -> Path:
    return _REPO_ROOT / "apps" / "paths.json"


def _load_paths_config_for_mutation() -> Dict[str, list[str]]:
    cfg_path = _paths_config_path()
    if not cfg_path.is_file():
        raise HTTPException(status_code=500, detail=f"paths config missing: {cfg_path}")
    try:
        payload = _load_json(str(cfg_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to read paths config: {exc}") from exc

    normalized: Dict[str, list[str]] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise HTTPException(status_code=500, detail=f"paths config contains a non-string key: {key!r}")
        if not isinstance(value, list):
            raise HTTPException(status_code=500, detail=f"paths config entry {key!r} must be a list")
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise HTTPException(status_code=500, detail=f"paths config entry {key!r} contains a non-string value")
            cleaned = item.strip()
            if not cleaned:
                raise HTTPException(status_code=500, detail=f"paths config entry {key!r} contains an empty value")
            out.append(cleaned)
        normalized[key] = out
    return normalized


def _save_paths_config(payload: Dict[str, list[str]]) -> None:
    cfg_path = _paths_config_path()
    try:
        _save_json(str(cfg_path), payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to write paths config: {exc}") from exc


def _resolve_paths_config_entry_path(entry: str) -> Path:
    raw = str(entry or "").strip()
    if not raw:
        raise RuntimeError("paths config entry must not be empty")
    candidate = Path(os.path.expanduser(raw))
    if not candidate.is_absolute() and not _WINDOWS_DRIVE_RE.match(raw):
        candidate = _REPO_ROOT / candidate
    return candidate.resolve(strict=False)


def _paths_config_entry_for_file(path: Path) -> str:
    resolved = path.resolve(strict=False)
    repo_root = _REPO_ROOT.resolve(strict=False)
    try:
        rel = resolved.relative_to(repo_root)
        return rel.as_posix()
    except Exception:
        return resolved.as_posix()


def build_router(
    *,
    model_api: Any,
) -> APIRouter:
    router = APIRouter()
    log = get_backend_logger("backend.api")
    inventory_log = get_backend_logger("inventory")
    repo_root = _REPO_ROOT.resolve(strict=False)
    from apps.backend.services.model_catalog import (
        current_models_revision,
        invalidate_model_catalog,
        refresh_model_catalog,
    )

    def _resolve_library_target(payload: Dict[str, Any], *, require_key: bool) -> tuple[str | None, str]:
        key_raw = str(payload.get("key") or "").strip()
        kind_raw = str(payload.get("kind") or "").strip().lower()

        key: str | None = key_raw or None
        kind: str | None = None
        if kind_raw:
            kind = _normalize_library_kind(kind_raw)

        if key is not None:
            derived = _kind_for_library_key(key)
            if kind is not None and kind != derived:
                raise HTTPException(
                    status_code=400,
                    detail=f"library key {key!r} implies kind={derived!r}, but payload provided kind={kind!r}",
                )
            return key, derived

        if require_key:
            raise HTTPException(status_code=400, detail="'key' is required for add operations")
        if kind is None:
            raise HTTPException(status_code=400, detail="either 'key' or 'kind' is required")
        return None, kind

    def _resolve_payload_path(raw: object) -> Path:
        sanitized = _sanitize_model_path_input(raw)
        if not sanitized:
            raise HTTPException(status_code=400, detail="'path' is required")
        if _WINDOWS_DRIVE_RE.match(sanitized) and os.name != "nt":
            raise HTTPException(
                status_code=400,
                detail=f"windows-style path is not valid on this server: {sanitized!r}",
            )

        candidate = Path(os.path.expanduser(sanitized))
        if not candidate.is_absolute() and not _WINDOWS_DRIVE_RE.match(sanitized):
            candidate = repo_root / candidate
        return candidate.resolve(strict=False)

    def _scan_library_candidates(path: Path, *, kind: str) -> list[Path]:
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"path not found: {path}")
        candidates: list[Path] = []
        seen: set[str] = set()

        def _append_if_supported(file_path: Path) -> None:
            resolved = file_path.resolve(strict=False)
            if not _is_supported_library_file(resolved, kind=kind):
                return
            key = os.path.normcase(str(resolved))
            if key in seen:
                return
            seen.add(key)
            candidates.append(resolved)

        if path.is_file():
            _append_if_supported(path)
            return candidates
        if not path.is_dir():
            raise HTTPException(status_code=400, detail=f"path is neither a file nor a directory: {path}")

        try:
            for entry in sorted(path.rglob("*"), key=lambda item: str(item).lower()):
                if entry.is_file():
                    _append_if_supported(entry)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to scan path {path}: {exc}") from exc
        return candidates

    def _file_size_bytes(file_path: Path) -> int:
        try:
            return int(file_path.stat().st_size)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read file size for {file_path}: {exc}") from exc

    def _resolve_existing_library_paths(key: str) -> list[Path]:
        paths_cfg = _load_paths_config_for_mutation()
        current = paths_cfg.get(key)
        if current is None:
            return []
        if not isinstance(current, list):
            raise HTTPException(status_code=500, detail=f"paths config entry {key!r} must be a list")
        try:
            return [_resolve_paths_config_entry_path(entry) for entry in current]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"invalid paths config entry under {key!r}: {exc}") from exc

    def _is_file_already_in_library(*, file_path: Path, existing_paths: list[Path]) -> bool:
        resolved_file = file_path.resolve(strict=False)
        normalized_file = os.path.normcase(str(resolved_file))
        for existing in existing_paths:
            normalized_existing = os.path.normcase(str(existing))
            if normalized_existing == normalized_file:
                return True
            if existing.is_dir():
                try:
                    resolved_file.relative_to(existing)
                except ValueError:
                    pass
                else:
                    return True
        return False

    def _add_library_file(*, key: str, kind: str, file_path_raw: object) -> Dict[str, Any]:
        file_path = _resolve_payload_path(file_path_raw)
        if not file_path.exists():
            raise HTTPException(status_code=400, detail=f"path not found: {file_path}")
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail=f"path is not a file: {file_path}")
        if not _is_supported_library_file(file_path, kind=kind):
            allowed = ", ".join(sorted(_allowed_exts_for_kind(kind)))
            raise HTTPException(
                status_code=400,
                detail=f"unsupported file extension for kind={kind!r}: {file_path.name!r} (allowed: {allowed})",
            )

        existing_paths = _resolve_existing_library_paths(key)
        already_present = _is_file_already_in_library(file_path=file_path, existing_paths=existing_paths)

        added = False
        if not already_present:
            paths_cfg = _load_paths_config_for_mutation()
            current = paths_cfg.get(key)
            if current is None:
                current = []
            if not isinstance(current, list):
                raise HTTPException(status_code=500, detail=f"paths config entry {key!r} must be a list")
            current = [*current, _paths_config_entry_for_file(file_path)]
            paths_cfg[key] = current
            _save_paths_config(paths_cfg)
            added = True

        size_bytes = _file_size_bytes(file_path)
        try:
            sha256, short_hash = model_api.hash_for_file(str(file_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to compute hash for {file_path}: {exc}") from exc
        if not sha256:
            raise HTTPException(status_code=500, detail=f"failed to compute hash for {file_path}")

        return {
            "name": file_path.name,
            "path": str(file_path),
            "ext": file_path.suffix.lower(),
            "size_bytes": size_bytes,
            "already_in_library": already_present,
            "type": kind,
            "library_key": key,
            "added": added,
            "sha256": sha256,
            "short_hash": short_hash,
        }

    def _log_inventory_refresh_summary(inv: Dict[str, Any]) -> None:
        wan22_count = len(inv.get("wan22", []))
        ip_adapter_models_count = len(inv.get("ip_adapter_models", []))
        ip_adapter_image_encoders_count = len(inv.get("ip_adapter_image_encoders", []))
        inventory_log.info(
            "inventory: refreshed (vaes=%d, text_encoders=%d, loras=%d, ip_adapter_models=%d, ip_adapter_image_encoders=%d, wan22.gguf=%d, metadata=%d)",
            len(inv.get("vaes", [])),
            len(inv.get("text_encoders", [])),
            len(inv.get("loras", [])),
            ip_adapter_models_count,
            ip_adapter_image_encoders_count,
            wan22_count,
            len(inv.get("metadata", [])),
        )
        if wan22_count != 0:
            return

        wan22_roots = get_paths_for("wan22_ckpt")
        inventory_log.warning(
            "inventory: wan22.gguf=0 after refresh; scanned `wan22_ckpt` roots=%s (recursive). "
            "Place WAN22 .gguf files under one of these roots and refresh inventory "
            "(/api/models/inventory?refresh=1 or POST /api/models/inventory/refresh).",
            wan22_roots,
        )

    def _normalized_inventory_payload(inv: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "vaes": _normalize_inventory_for_api(inv.get("vaes", [])),
            "text_encoders": _normalize_inventory_for_api(inv.get("text_encoders", [])),
            "loras": _normalize_inventory_for_api(inv.get("loras", [])),
            "ip_adapter_models": _normalize_inventory_for_api(inv.get("ip_adapter_models", [])),
            "ip_adapter_image_encoders": _normalize_inventory_for_api(inv.get("ip_adapter_image_encoders", [])),
            "wan22": {"gguf": _normalize_inventory_for_api(inv.get("wan22", []))},
            "metadata": _normalize_inventory_for_api(inv.get("metadata", [])),
        }

    def _current_models_revision() -> int:
        return int(current_models_revision())

    def _refresh_model_catalog_or_raise(*, reason: str, detail_prefix: str) -> tuple[int, Dict[str, list[Dict[str, str]]]]:
        try:
            return refresh_model_catalog(reason=reason)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{detail_prefix}: {exc}") from exc

    def _invalidate_model_catalog_or_raise(*, reason: str, detail_prefix: str) -> int:
        try:
            return int(invalidate_model_catalog(reason=reason))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{detail_prefix}: {exc}") from exc

    @router.get("/api/models")
    def list_models(refresh: bool = Query(False, description="If true, re-scan checkpoint roots before returning.")) -> Dict[str, Any]:
        if refresh:
            _refresh_model_catalog_or_raise(
                reason="api.models.list(refresh=true)",
                detail_prefix="model catalog refresh failed",
            )
            entries = model_api.list_checkpoints(refresh=False)
        else:
            entries = model_api.list_checkpoints(refresh=False)
        models = [_serialize_checkpoint(entry) for entry in entries]
        models_info = [e.as_dict() for e in entries]
        current = models[0]["title"] if models else None
        return {
            "models": models,
            "current": current,
            "models_info": models_info,
            "models_revision": _current_models_revision(),
        }

    @router.get("/api/models/inventory")
    def list_models_inventory(refresh: bool = Query(False, description="If true, re-scan the models/ and huggingface/ folders.")) -> Dict[str, Any]:
        from apps.backend.inventory import cache as _inv_cache
        if refresh:
            revision, inv = _refresh_model_catalog_or_raise(
                reason="api.models.inventory.list(refresh=true)",
                detail_prefix="inventory refresh failed",
            )
            _log_inventory_refresh_summary(inv)
        else:
            inv = _inv_cache.get()
            revision = _current_models_revision()
        payload = _normalized_inventory_payload(inv)
        payload["models_revision"] = int(revision)
        return payload

    @router.post("/api/models/path-scan")
    def scan_model_path(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        key, kind = _resolve_library_target(payload, require_key=False)
        root_path = _resolve_payload_path(payload.get("path"))
        candidates = _scan_library_candidates(root_path, kind=kind)
        existing_paths = _resolve_existing_library_paths(key) if key is not None else []
        return {
            "kind": kind,
            "key": key,
            "root": str(root_path),
            "items": [
                {
                    "name": candidate.name,
                    "path": str(candidate),
                    "ext": candidate.suffix.lower(),
                    "size_bytes": _file_size_bytes(candidate),
                    "already_in_library": _is_file_already_in_library(file_path=candidate, existing_paths=existing_paths),
                }
                for candidate in candidates
            ],
        }

    @router.post("/api/models/path-add")
    def add_model_path_item(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        key, kind = _resolve_library_target(payload, require_key=True)
        assert key is not None
        item = _add_library_file(key=key, kind=kind, file_path_raw=payload.get("path"))
        if item.get("added") is True:
            models_revision = _invalidate_model_catalog_or_raise(
                reason=f"api.models.path-add:{key}",
                detail_prefix="path-add updated paths but model catalog invalidation failed",
            )
        else:
            models_revision = _current_models_revision()
        return {
            "key": key,
            "kind": kind,
            "item": item,
            "models_revision": int(models_revision),
        }

    @router.post("/api/models/path-add-all")
    def add_model_path_items(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        key, kind = _resolve_library_target(payload, require_key=True)
        assert key is not None
        root_path = _resolve_payload_path(payload.get("path"))
        candidates = _scan_library_candidates(root_path, kind=kind)

        results: list[Dict[str, Any]] = []
        total = len(candidates)
        added_count = 0
        error_count = 0
        for index, candidate in enumerate(candidates, start=1):
            size_bytes: int | None = None
            try:
                size_bytes = _file_size_bytes(candidate)
                item = _add_library_file(key=key, kind=kind, file_path_raw=str(candidate))
                if item.get("added") is True:
                    added_count += 1
                results.append(
                    {
                        "index": index,
                        "total": total,
                        "ok": True,
                        "item": item,
                    }
                )
            except HTTPException as exc:
                error_count += 1
                results.append(
                    {
                        "index": index,
                        "total": total,
                        "ok": False,
                        "item": {
                            "name": candidate.name,
                            "path": str(candidate),
                            "ext": candidate.suffix.lower(),
                            "size_bytes": size_bytes,
                            "type": kind,
                            "library_key": key,
                        },
                        "detail": exc.detail,
                    }
                )
            except Exception as exc:
                error_count += 1
                results.append(
                    {
                        "index": index,
                        "total": total,
                        "ok": False,
                        "item": {
                            "name": candidate.name,
                            "path": str(candidate),
                            "ext": candidate.suffix.lower(),
                            "size_bytes": size_bytes,
                            "type": kind,
                            "library_key": key,
                        },
                        "detail": str(exc),
                    }
                )
        if added_count > 0:
            models_revision = _invalidate_model_catalog_or_raise(
                reason=f"api.models.path-add-all:{key}",
                detail_prefix="path-add-all updated paths but model catalog invalidation failed",
            )
        else:
            models_revision = _current_models_revision()
        return {
            "key": key,
            "kind": kind,
            "root": str(root_path),
            "total": total,
            "added_count": added_count,
            "error_count": error_count,
            "results": results,
            "models_revision": int(models_revision),
        }

    @router.get("/api/models/file-metadata")
    def get_file_metadata(path: str = Query(..., description="Repo-relative or absolute path to a weights file.")) -> Dict[str, Any]:
        try:
            result = read_file_metadata(path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="file not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return result.as_dict()

    @router.get("/api/models/checkpoint-metadata")
    def get_checkpoint_metadata(value: str = Query(..., description="Checkpoint title/name/path to resolve.")) -> Dict[str, Any]:
        record = None
        try:
            record = model_api.find_checkpoint(value)
        except Exception:
            record = None

        if record is None:
            raise HTTPException(status_code=404, detail="checkpoint not found")

        raw_path = str(getattr(record, "filename", "") or "").strip()
        if not raw_path:
            raise HTTPException(status_code=500, detail="checkpoint record missing filename")

        weights_path = Path(raw_path).expanduser()
        resolved = weights_path.resolve(strict=False)
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="checkpoint file not found")
        if not resolved.is_file():
            raise HTTPException(status_code=400, detail="checkpoint is not a file")

        try:
            meta = read_file_metadata(str(resolved))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="file not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        size_bytes = int(resolved.stat().st_size)
        short_hash = getattr(record, "short_hash", None)
        return {
            "hash": short_hash,
            "file.name": resolved.stem,
            "file.path": str(resolved),
            "file.size.bytes": size_bytes,
            "file.size.megabytes": round(size_bytes / 1_000_000, 3),
            "file.size.gigabytes": round(size_bytes / 1_000_000_000, 3),
            "metadata": {"raw": dict(meta.flat), "nested": dict(meta.nested)},
        }

    @router.post("/api/models/prompt-token-count")
    def prompt_token_count(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        engine = str(payload.get("engine") or "").strip()
        if not engine:
            raise HTTPException(status_code=400, detail="'engine' is required.")
        prompt = str(payload.get("prompt") or "")
        try:
            count = _count_prompt_tokens(engine=engine, prompt=prompt)
        except (NotImplementedError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            message = str(exc)
            if "Unsupported engine" in message:
                raise HTTPException(status_code=400, detail=message)
            raise HTTPException(status_code=500, detail=message)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to count prompt tokens: {exc}")
        return {
            "engine": engine,
            "prompt_len": len(prompt),
            "count": max(0, math.trunc(count)),
        }

    @router.post("/api/models/inventory/refresh")
    def refresh_models_inventory() -> Dict[str, Any]:
        revision, inv = _refresh_model_catalog_or_raise(
            reason="api.models.inventory.refresh(sync)",
            detail_prefix="inventory refresh failed",
        )
        _log_inventory_refresh_summary(inv)
        payload = _normalized_inventory_payload(inv)
        payload["models_revision"] = int(revision)
        return payload

    @router.post("/api/models/inventory/refresh/async")
    async def refresh_models_inventory_async() -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-models-inventory-refresh-{uuid4().hex})"
        register_task(task_id, entry)
        inventory_log.info("inventory refresh task queued (task_id=%s)", task_id)

        def worker() -> None:
            try:
                entry.push_event({"type": "status", "stage": "running"})
                entry.push_event(
                    {
                        "type": "progress",
                        "stage": "inventory.refresh",
                        "percent": None,
                        "step": None,
                        "total_steps": None,
                        "eta_seconds": None,
                        "message": "Refreshing model inventory...",
                    }
                )
                inventory_log.info("inventory refresh task started (task_id=%s)", task_id)
                revision, inv = refresh_model_catalog(reason="api.models.inventory.refresh(async)")
                _log_inventory_refresh_summary(inv)
                payload = _normalized_inventory_payload(inv)
                payload["models_revision"] = int(revision)
                entry.result = {"result": {"inventory": payload, "models_revision": int(revision)}}
                entry.mark_finished(success=True)
                inventory_log.info(
                    "inventory refresh task completed (task_id=%s revision=%d vaes=%d text_encoders=%d loras=%d ip_adapter_models=%d ip_adapter_image_encoders=%d wan22.gguf=%d metadata=%d)",
                    task_id,
                    revision,
                    len(inv.get("vaes", [])),
                    len(inv.get("text_encoders", [])),
                    len(inv.get("loras", [])),
                    len(inv.get("ip_adapter_models", [])),
                    len(inv.get("ip_adapter_image_encoders", [])),
                    len(inv.get("wan22", [])),
                    len(inv.get("metadata", [])),
                )
            except Exception as exc:
                entry.error = build_public_task_error(exc)
                entry.mark_finished(success=False)
                inventory_log.error("inventory refresh task failed (task_id=%s): %s", task_id, exc)

        threading.Thread(target=worker, name=f"inventory-refresh-task-{task_id}", daemon=True).start()
        return {"task_id": task_id}

    @router.post("/api/models/load")
    def api_models_load(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        tab_id = str(payload.get("tab_id") or "")
        if not tab_id:
            raise HTTPException(status_code=400, detail="tab_id required")
        log.info("[models] load requested for tab %s", tab_id)
        return {"ok": True}

    @router.post("/api/models/unload")
    def api_models_unload(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        tab_id = str(payload.get("tab_id") or "")
        if not tab_id:
            raise HTTPException(status_code=400, detail="tab_id required")
        log.info("[models] unload requested for tab %s", tab_id)
        return {"ok": True}

    @router.get("/api/engines/capabilities")
    def list_engine_capabilities() -> Dict[str, Any]:
        try:
            from apps.backend.runtime.model_registry.capabilities import (
                ENGINE_ID_TO_SEMANTIC_ENGINE,
                serialize_exact_engine_inpaint_modes,
                serialize_engine_capabilities,
                serialize_family_capabilities,
                serialize_parked_exact_engines,
            )
            from apps.backend.core.contracts.asset_requirements import (
                contract_for_core_only,
                contract_for_engine,
                contract_owner_for_semantic_engine,
            )
            from apps.backend.interfaces.api.dependency_checks import build_engine_dependency_checks
            try:
                from apps.backend.runtime.memory.smart_offload import get_smart_cache_stats

                cache_stats = get_smart_cache_stats()
            except Exception:
                cache_stats = {}

            engine_id_to_semantic_engine: Dict[str, str] = {
                engine_id: semantic.value for engine_id, semantic in ENGINE_ID_TO_SEMANTIC_ENGINE.items()
            }
            engines = serialize_engine_capabilities()
            asset_contracts: Dict[str, Any] = {}
            for semantic_engine in sorted(engines.keys()):
                contract_owner_engine_id = contract_owner_for_semantic_engine(semantic_engine)
                asset_contracts[semantic_engine] = {
                    "base": contract_for_engine(contract_owner_engine_id).as_dict(),
                    "core_only": contract_for_core_only(contract_owner_engine_id).as_dict(),
                }
            dependency_checks = build_engine_dependency_checks(
                engine_capabilities=engines,
                model_api=model_api,
            )
            return {
                "engines": engines,
                "families": serialize_family_capabilities(),
                "smart_cache": cache_stats,
                "asset_contracts": asset_contracts,
                "engine_id_to_semantic_engine": engine_id_to_semantic_engine,
                "exact_engine_inpaint_modes": serialize_exact_engine_inpaint_modes(),
                "parked_exact_engines": serialize_parked_exact_engines(),
                "dependency_checks": dependency_checks,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to read engine capabilities: {exc}")

    @router.get("/api/samplers")
    def list_samplers() -> Dict[str, Any]:
        from apps.backend.runtime.sampling.registry import get_sampler_spec

        samplers = []
        for entry in SAMPLER_OPTIONS:
            supported = bool(entry.get("supported", True))
            spec = None
            if supported:
                try:
                    spec = get_sampler_spec(str(entry["name"]))
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"/api/samplers metadata drift for supported sampler {entry['name']!r}: {exc}",
                    ) from exc
            samplers.append(
                {
                    "name": entry["name"],
                    "supported": supported,
                    "default_scheduler": spec.default_scheduler if spec is not None else None,
                    "allowed_schedulers": sorted(spec.allowed_schedulers) if spec is not None else [],
                }
            )
        return {"samplers": samplers}

    @router.get("/api/schedulers")
    def list_schedulers() -> Dict[str, Any]:
        schedulers = []
        for entry in SCHEDULER_OPTIONS:
            if not entry.get("supported", True):
                continue
            schedulers.append(
                {
                    "name": entry["name"],
                    "supported": bool(entry.get("supported", True)),
                }
            )
        return {"schedulers": schedulers}

    @router.get("/api/embeddings")
    def list_embeddings() -> Dict[str, Any]:
        from apps.backend.infra.registry.embeddings import describe_embeddings as _describe

        info = [e.__dict__ for e in _describe()]
        loaded = {
            e["name"]: {
                "name": e["name"],
                "vectors": e.get("vectors"),
                "shape": e.get("dims"),
                "step": e.get("step"),
            }
            for e in info
            if e.get("vectors")
        }
        skipped = {
            e["name"]: {
                "name": e["name"],
                "vectors": e.get("vectors"),
                "shape": e.get("dims"),
                "step": e.get("step"),
            }
            for e in info
            if not e.get("vectors")
        }
        return {"loaded": loaded, "skipped": skipped, "embeddings_info": info}

    return router

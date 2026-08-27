"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Upscalers and standalone upscale API routes.
Exposes:
- local upscaler discovery (`GET /api/upscalers`)
- remote HF upscaler listing + downloads (`GET/POST /api/upscalers/*`), with optional manifest-based metadata enrichment
  (`upscalers/manifest.json`, schema v1) and explicit `manifest_error`/`manifest_errors` surfacing
- standalone upscaling tasks (`POST /api/upscale`)
- dedicated SeedVR2 video-upscale tasks (`POST /api/video-upscale`, source preflight, and explicit execution policy)
Remote listing/download respects the upscaler safeweights policy (`CODEX_SAFE_WEIGHTS=1` blocks non-`.safetensors` weights).

Symbols (top-level; keep in sync; no ghosts):
- `_parse_video_upscale_request` (function): Validates the strict dedicated SeedVR2 request payload.
- `_preflight_video_upscale_source` (function): Verifies a backend-visible source video before task registration.
- `build_router` (function): Build the APIRouter for upscaler endpoints.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import asyncio
import json
import logging
import math
from dataclasses import replace
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from apps.backend.interfaces.api.public_errors import public_http_error_detail
from apps.backend.interfaces.api.task_registry import TaskEntry, register_task
from apps.backend.infra.config.paths import get_paths_for
from apps.backend.interfaces.api.device_selection import parse_device_from_payload
from apps.backend.interfaces.api.upscalers_manifest import validate_upscalers_manifest
from apps.backend.runtime.vision.upscalers.safeweights import allowed_upscaler_weight_suffixes, safeweights_enabled


_HF_UPSCALERS_REPO_ID = "sangoi-exe/sd-webui-codex"
_HF_MANIFEST_PATH = "upscalers/manifest.json"
_router_log = get_backend_logger("backend.api.routers.upscale")
_SEEDVR2_DIT_MODELS = frozenset(
    {
        "seedvr2_ema_3b_fp16.safetensors",
        "seedvr2_ema_7b_fp16.safetensors",
        "seedvr2_ema_7b_sharp_fp16.safetensors",
    }
)
_SEEDVR2_COLOR_CORRECTIONS = frozenset({"lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"})


def _normalize_hf_repo_id(repo_id: str | None) -> str:
    repo = str(repo_id or "").strip()
    if not repo:
        return _HF_UPSCALERS_REPO_ID
    if repo != _HF_UPSCALERS_REPO_ID:
        raise HTTPException(status_code=400, detail=f"repo_id not allowed in v1 (allowed: {_HF_UPSCALERS_REPO_ID})")
    return repo


def _normalize_hf_revision(revision: str | None) -> str | None:
    rev = str(revision).strip() if isinstance(revision, str) else None
    return rev or None


def _parse_explicit_device(payload: Dict[str, Any]) -> str:
    """Parse/validate per-request device selection (fail loud).

    Note: do not call `switch_primary_device()` here; apply it only when the task starts running (single-flight-safe).
    """
    try:
        return parse_device_from_payload(payload)
    except ValueError as exc:
        _router_log.warning("upscale device selection validation failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=public_http_error_detail(exc, fallback="Invalid 'device' selection"),
        ) from None


def _safe_relpath(raw: str) -> str:
    s = str(raw or "").replace("\\", "/").lstrip("/")
    if ".." in s.split("/"):
        raise HTTPException(status_code=400, detail="invalid path")
    return s


def _parse_video_upscale_request(payload: Dict[str, Any]):
    allowed_fields = {
        "video_path",
        "device",
        "dit_model",
        "resolution",
        "max_resolution",
        "batch_size",
        "uniform_batch_size",
        "temporal_overlap",
        "prepend_frames",
        "color_correction",
        "input_noise_scale",
        "latent_noise_scale",
        "streaming",
        "smart_fallback",
    }
    unknown_fields = sorted(str(key) for key in payload if key not in allowed_fields)
    if unknown_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported /api/video-upscale field(s): {', '.join(unknown_fields)}.",
        )

    raw_video_path = payload.get("video_path")
    if not isinstance(raw_video_path, str) or not raw_video_path.strip():
        raise HTTPException(status_code=400, detail="'video_path' must be a non-empty backend-visible file path")
    video_path = raw_video_path.strip()

    raw_dit_model = payload.get("dit_model")
    if not isinstance(raw_dit_model, str) or not raw_dit_model.strip():
        raise HTTPException(status_code=400, detail="'dit_model' must be a non-empty curated SeedVR2 model id")
    dit_model = raw_dit_model.strip()
    if dit_model not in _SEEDVR2_DIT_MODELS:
        allowed_models = ", ".join(sorted(_SEEDVR2_DIT_MODELS))
        raise HTTPException(status_code=400, detail=f"'dit_model' must be one of {{{allowed_models}}}")

    def optional_int(field: str, *, default: int, minimum: int) -> int:
        value = payload.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise HTTPException(status_code=400, detail=f"'{field}' must be an integer")
        parsed = int(value)
        if parsed < minimum:
            raise HTTPException(status_code=400, detail=f"'{field}' must be >= {minimum}")
        return parsed

    def optional_float(field: str, *, default: float) -> float:
        value = payload.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{field}' must be a number")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
            raise HTTPException(status_code=400, detail=f"'{field}' must be a finite number within [0, 1]")
        return parsed

    resolution = optional_int("resolution", default=1080, minimum=16)
    max_resolution = optional_int("max_resolution", default=0, minimum=0)
    batch_size = optional_int("batch_size", default=5, minimum=1)
    if (batch_size - 1) % 4 != 0:
        raise HTTPException(status_code=400, detail="'batch_size' must satisfy 4n+1")
    temporal_overlap = optional_int("temporal_overlap", default=0, minimum=0)
    if temporal_overlap >= batch_size:
        raise HTTPException(status_code=400, detail="'temporal_overlap' must be lower than 'batch_size'")
    prepend_frames = optional_int("prepend_frames", default=0, minimum=0)

    uniform_batch_size = payload.get("uniform_batch_size", False)
    if not isinstance(uniform_batch_size, bool):
        raise HTTPException(status_code=400, detail="'uniform_batch_size' must be a boolean")

    streaming = payload.get("streaming", False)
    if not isinstance(streaming, bool):
        raise HTTPException(status_code=400, detail="'streaming' must be a boolean")

    smart_fallback = payload.get("smart_fallback", False)
    if not isinstance(smart_fallback, bool):
        raise HTTPException(status_code=400, detail="'smart_fallback' must be a boolean")
    if streaming and smart_fallback:
        raise HTTPException(
            status_code=400,
            detail="'streaming' and 'smart_fallback' cannot both be enabled.",
        )

    color_correction_raw = payload.get("color_correction", "lab")
    if not isinstance(color_correction_raw, str):
        raise HTTPException(status_code=400, detail="'color_correction' must be a string")
    color_correction = color_correction_raw.strip().lower()
    if color_correction not in _SEEDVR2_COLOR_CORRECTIONS:
        allowed_corrections = ", ".join(sorted(_SEEDVR2_COLOR_CORRECTIONS))
        raise HTTPException(status_code=400, detail=f"'color_correction' must be one of {{{allowed_corrections}}}")

    device = _parse_explicit_device(payload)
    if device not in {"cuda", "mps"}:
        raise HTTPException(
            status_code=400,
            detail="SeedVR2 video upscaling supports only cuda or mps.",
        )
    from apps.backend.core.params.video import SeedVR2UpscaleOptions
    from apps.backend.use_cases.video_upscale import VideoUpscaleRequest

    return VideoUpscaleRequest(
        video_path=video_path,
        device=device,
        options=SeedVR2UpscaleOptions(
            dit_model=dit_model,
            resolution=resolution,
            max_resolution=max_resolution,
            batch_size=batch_size,
            uniform_batch_size=uniform_batch_size,
            temporal_overlap=temporal_overlap,
            prepend_frames=prepend_frames,
            color_correction=color_correction,
            input_noise_scale=optional_float("input_noise_scale", default=0.0),
            latent_noise_scale=optional_float("latent_noise_scale", default=0.0),
            streaming=streaming,
            smart_fallback=smart_fallback,
        ),
    )


def _preflight_video_upscale_source(video_path: str) -> str:
    source_path = Path(video_path).expanduser()
    if not source_path.is_file():
        raise HTTPException(status_code=400, detail="'video_path' must identify a readable file")

    from apps.backend.video.io.ffmpeg import probe_video

    try:
        probe_video(str(source_path))
    except Exception as exc:
        _router_log.warning("video-upscale source probe failed for %s: %s", source_path, exc, exc_info=False)
        raise HTTPException(status_code=400, detail="'video_path' must identify a readable video file") from None
    return str(source_path.resolve())


def build_router(
    *,
    codex_root: Path,
    opts_get,
    generation_provenance,
    save_generated_images,
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/upscalers")
    async def get_upscalers() -> Dict[str, Any]:
        from apps.backend.runtime.vision.upscalers.registry import list_upscalers

        items = [
            {
                "id": u.id,
                "label": u.label,
                "kind": u.kind.value,
                "meta": u.meta,
            }
            for u in list_upscalers()
        ]
        return {"upscalers": items}

    @router.get("/api/upscalers/remote")
    def get_remote_upscalers(
        repo_id: str | None = None,
        revision: str | None = None,
    ) -> Dict[str, Any]:
        # v1: manifest is a plus. Always return the raw `upscalers/**` listing (suffix-filtered) and enrich it when the
        # manifest is present/valid.
        hf_repo_id = _normalize_hf_repo_id(repo_id)
        hf_revision = _normalize_hf_revision(revision)

        try:
            from huggingface_hub import HfApi, hf_hub_download  # type: ignore

            files = HfApi().list_repo_files(repo_id=hf_repo_id, revision=hf_revision)
        except Exception as exc:
            _router_log.warning("failed to query Hugging Face repo '%s': %s", hf_repo_id, exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="failed to query Hugging Face repo"),
            ) from None

        manifest_found = any(isinstance(name, str) and name == _HF_MANIFEST_PATH for name in files)
        manifest: Any | None = None
        manifest_error: str | None = None
        manifest_errors: list[str] = []
        manifest_weights_by_hf_path: dict[str, dict[str, Any]] = {}
        if manifest_found:
            try:
                local_path = hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=_HF_MANIFEST_PATH,
                    revision=hf_revision,
                )
                with open(local_path, "r", encoding="utf-8") as handle:
                    raw_manifest = json.load(handle)
                result = validate_upscalers_manifest(raw_manifest)
                manifest = result.manifest
                manifest_errors = list(result.errors or [])
                manifest_weights_by_hf_path = dict(result.weights_by_hf_path or {})
                if manifest_errors:
                    manifest_error = (
                        manifest_errors[0]
                        if len(manifest_errors) == 1
                        else f"{manifest_errors[0]} (+{len(manifest_errors) - 1} more)"
                    )
            except Exception as exc:
                _router_log.warning("failed to parse upscalers manifest from '%s': %s", hf_repo_id, exc)
                manifest_error = public_http_error_detail(exc, fallback="failed to load upscalers manifest")
                manifest_errors = [manifest_error]
                manifest_weights_by_hf_path = {}

        weights: list[dict[str, Any]] = []
        allowed_suffixes = allowed_upscaler_weight_suffixes()
        for name in files:
            if not isinstance(name, str):
                continue
            if not name.startswith("upscalers/"):
                continue
            if not name.lower().endswith(allowed_suffixes):
                continue
            base: dict[str, Any] = {"hf_path": name, "label": name.split("/")[-1], "curated": False, "meta": None}
            meta = manifest_weights_by_hf_path.get(name)
            if isinstance(meta, dict):
                base["curated"] = True
                base["label"] = meta["label"]
                base["meta"] = {
                    "id": meta["id"],
                    "arch": meta["arch"],
                    "scale": meta["scale"],
                    "license_name": meta["license_name"],
                    "license_url": meta["license_url"],
                    "license_spdx": meta.get("license_spdx"),
                    "sha256": meta["sha256"],
                    "tags": meta.get("tags") or [],
                    "notes": meta.get("notes"),
                }
            weights.append(base)

        weights.sort(
            key=lambda x: (
                0 if bool(x.get("curated")) else 1,
                str(x.get("label", "")).lower(),
                str(x.get("hf_path", "")).lower(),
            )
        )
        return {
            "repo_id": hf_repo_id,
            "revision": hf_revision,
            "manifest_path": _HF_MANIFEST_PATH,
            "manifest_found": bool(manifest_found),
            "manifest_error": manifest_error,
            "manifest_errors": list(manifest_errors),
            "manifest": manifest,
            "weights": weights,
            "safeweights_enabled": bool(safeweights_enabled()),
            "allowed_weight_suffixes": list(allowed_suffixes),
        }

    @router.post("/api/upscalers/download")
    async def download_upscalers(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")

        hf_repo_id = _normalize_hf_repo_id(payload.get("repo_id"))
        hf_revision = _normalize_hf_revision(payload.get("revision"))

        files = payload.get("files")
        if not isinstance(files, list) or not files:
            raise HTTPException(status_code=400, detail="Missing 'files' (list)")

        # Destination root: first configured upscale_models root.
        roots = get_paths_for("upscale_models")
        if not roots:
            raise HTTPException(status_code=400, detail="No 'upscale_models' path configured in apps/paths.json")
        dst_root = Path(roots[0])
        dst_root.mkdir(parents=True, exist_ok=True)

        items = []
        allowed_suffixes = set(allowed_upscaler_weight_suffixes())
        for entry in files:
            if not isinstance(entry, str) or not entry.strip():
                raise HTTPException(status_code=400, detail="Invalid file entry")
            hf_path = _safe_relpath(entry)
            if not hf_path.startswith("upscalers/"):
                raise HTTPException(status_code=400, detail="hf_path must be under upscalers/")
            suffix = Path(hf_path).suffix.lower()
            if suffix not in allowed_suffixes:
                allowed = "|".join(sorted(allowed_suffixes))
                if safeweights_enabled():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsafe weights blocked by CODEX_SAFE_WEIGHTS=1 (allowed: {allowed})",
                    )
                raise HTTPException(status_code=400, detail=f"Unsupported weights extension (allowed: {allowed})")
            rel = hf_path[len("upscalers/") :]
            dst = (dst_root / rel).resolve()
            # Keep writes within configured root.
            try:
                dst.relative_to(dst_root.resolve())
            except Exception:
                raise HTTPException(status_code=400, detail="invalid destination path") from None
            items.append({"hf_path": hf_path, "dst_path": str(dst)})

        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-upscalers-download-{uuid4().hex})"
        register_task(task_id, entry)

        from apps.backend.interfaces.api.tasks.upscale_tasks import _DownloadItem, run_upscaler_download_task

        dl_items = [_DownloadItem(hf_path=str(x["hf_path"]), dst_path=Path(str(x["dst_path"]))) for x in items]
        run_upscaler_download_task(
            task_id=task_id,
            items=dl_items,
            entry=entry,
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
        )
        return {"task_id": task_id}

    @router.post("/api/upscale")
    async def upscale(
        image: UploadFile | None = File(default=None),
        payload: str = Form(default="{}"),
    ) -> Dict[str, Any]:
        if image is None:
            raise HTTPException(status_code=400, detail="Missing 'image' file")
        try:
            data = json.loads(payload) if payload else {}
        except Exception as exc:
            _router_log.warning("upscale payload JSON parse failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="payload must be valid JSON"),
            ) from None
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="payload must be JSON object")

        device = _parse_explicit_device(data)

        try:
            image_bytes = await image.read()
        except Exception as exc:
            _router_log.warning("upscale upload read failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="failed to read upload"),
            ) from None
        if not image_bytes:
            raise HTTPException(status_code=400, detail="empty image upload")

        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-upscale-{uuid4().hex})"
        register_task(task_id, entry)

        from apps.backend.interfaces.api.tasks.upscale_tasks import run_upscale_task

        run_upscale_task(
            task_id=task_id,
            payload=data,
            image_bytes=image_bytes,
            entry=entry,
            device=device,
            opts_get=opts_get,
            generation_provenance=generation_provenance,
            save_generated_images=save_generated_images,
        )
        return {"task_id": task_id}

    @router.post("/api/video-upscale")
    async def video_upscale(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")
        request = _parse_video_upscale_request(payload)
        request = replace(
            request,
            video_path=await asyncio.to_thread(_preflight_video_upscale_source, request.video_path),
        )

        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-video-upscale-{uuid4().hex})"
        register_task(task_id, entry)

        from apps.backend.interfaces.api.tasks.video_upscale_tasks import run_video_upscale_task

        run_video_upscale_task(task_id=task_id, request=request, entry=entry)
        return {"task_id": task_id}

    return router

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
- dedicated SeedVR2 video-upscale tasks (`POST /api/video-upscale`, pinned-compatible source suffixes, immutable snapshot, cross-platform media/resource admission, and execution policy)
Remote listing/download respects the upscaler safeweights policy (`CODEX_SAFE_WEIGHTS=1` blocks non-`.safetensors` weights).

Symbols (top-level; keep in sync; no ghosts):
- `_parse_video_upscale_request` (function): Validates the strict dedicated SeedVR2 request payload.
- `_preflight_video_upscale_source` (function): Admits the source suffix, snapshots exact media, and verifies timing, host memory, allocation units, optional inode capacity, and copied-audio output space before registration.
- `build_router` (function): Build the APIRouter for upscaler endpoints.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import asyncio
import json
import logging
import math
import os
import shutil
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
from apps.backend.video.export.ffmpeg_exporter import (
    VideoExportError,
    validate_timestamped_mp4_source,
    video_export_output_root,
)
from apps.backend.video.io.ffmpeg import probe_video, probe_video_timing
from apps.backend.video.upscaling.seedvr2 import (
    calculate_seedvr2_host_memory_admission,
    calculate_seedvr2_target_dimensions,
)
from apps.backend.core.params.video import SeedVR2UpscaleOptions
from apps.backend.use_cases.video_upscale import (
    VideoUpscaleRequest,
    VideoUpscaleSourceAdmission,
    cleanup_video_upscale_source_admission,
    cleanup_video_upscale_work_dir,
    create_video_upscale_work_dir,
    video_upscale_work_root,
)


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
_SEEDVR2_BATCH_SIZES = frozenset({1, 5, 9, 13, 17, 33, 65, 129})
_SEEDVR2_SOURCE_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")
_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_WORK_SCRATCH_BASE_BYTES = 128 * _MIB
_WORK_SCRATCH_BYTES_PER_PADDED_PIXEL = 5
_WORK_SNAPSHOT_FIXED_INODES_AFTER_ROOT = 2
_WORK_EXECUTION_FIXED_INODES_AFTER_ROOT = 5
_WORK_SMART_FALLBACK_RETRY_INODES = 1
_OUTPUT_SCRATCH_BASE_BYTES = 128 * _MIB
_OUTPUT_SCRATCH_BYTES_PER_PADDED_PIXEL = 4
_OUTPUT_SCRATCH_FIXED_INODES = 7
_SCRATCH_OPERATIONAL_RESERVE_BYTES = 2 * _GIB


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


def _parse_video_upscale_request(payload: Dict[str, Any]) -> tuple[str, str, SeedVR2UpscaleOptions]:
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

    def optional_int(field: str, *, default: int, minimum: int, maximum: int | None = None) -> int:
        value = payload.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise HTTPException(status_code=400, detail=f"'{field}' must be an integer")
        parsed = int(value)
        if parsed < minimum:
            raise HTTPException(status_code=400, detail=f"'{field}' must be >= {minimum}")
        if maximum is not None and parsed > maximum:
            raise HTTPException(status_code=400, detail=f"'{field}' must be <= {maximum}")
        return parsed

    def optional_float(field: str, *, default: float) -> float:
        value = payload.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{field}' must be a number")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
            raise HTTPException(status_code=400, detail=f"'{field}' must be a finite number within [0, 1]")
        return parsed

    resolution = optional_int("resolution", default=1080, minimum=16, maximum=4096)
    max_resolution = optional_int("max_resolution", default=0, minimum=0, maximum=8192)
    if 0 < max_resolution < 16:
        raise HTTPException(status_code=400, detail="'max_resolution' must be 0 or >= 16")
    batch_size = optional_int("batch_size", default=5, minimum=1)
    if batch_size not in _SEEDVR2_BATCH_SIZES:
        allowed_batches = ", ".join(str(value) for value in sorted(_SEEDVR2_BATCH_SIZES))
        raise HTTPException(status_code=400, detail=f"'batch_size' must be one of {{{allowed_batches}}}")
    temporal_overlap = optional_int("temporal_overlap", default=0, minimum=0)
    if temporal_overlap >= batch_size:
        raise HTTPException(status_code=400, detail="'temporal_overlap' must be lower than 'batch_size'")
    prepend_frames = optional_int("prepend_frames", default=0, minimum=0, maximum=128)

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
    return (
        video_path,
        device,
        SeedVR2UpscaleOptions(
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


def _nearest_existing_capacity_path(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise RuntimeError(f"No existing filesystem owner was found for '{path}'.")
        candidate = parent
    return candidate


def _missing_directory_count(path: Path, *, existing_ancestor: Path) -> int:
    resolved_path = path.expanduser().resolve()
    resolved_ancestor = existing_ancestor.expanduser().resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_ancestor)
    except ValueError as exc:
        raise RuntimeError(
            f"Filesystem capacity owner '{resolved_ancestor}' is not an ancestor of '{resolved_path}'."
        ) from exc
    return len(relative_path.parts)


def _filesystem_allocation_unit_and_available_inodes(path: Path) -> tuple[int, int | None]:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        volume_path_buffer = ctypes.create_unicode_buffer(32768)
        get_volume_path_name = kernel32.GetVolumePathNameW
        get_volume_path_name.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_volume_path_name.restype = wintypes.BOOL
        if not get_volume_path_name(str(path), volume_path_buffer, len(volume_path_buffer)):
            error_code = ctypes.get_last_error()
            raise OSError(error_code, f"GetVolumePathNameW failed for '{path}'.")

        sectors_per_cluster = wintypes.DWORD()
        bytes_per_sector = wintypes.DWORD()
        free_clusters = wintypes.DWORD()
        total_clusters = wintypes.DWORD()
        get_disk_free_space = kernel32.GetDiskFreeSpaceW
        dword_pointer = ctypes.POINTER(wintypes.DWORD)
        get_disk_free_space.argtypes = [
            wintypes.LPCWSTR,
            dword_pointer,
            dword_pointer,
            dword_pointer,
            dword_pointer,
        ]
        get_disk_free_space.restype = wintypes.BOOL
        if not get_disk_free_space(
            volume_path_buffer.value,
            ctypes.byref(sectors_per_cluster),
            ctypes.byref(bytes_per_sector),
            ctypes.byref(free_clusters),
            ctypes.byref(total_clusters),
        ):
            error_code = ctypes.get_last_error()
            raise OSError(error_code, f"GetDiskFreeSpaceW failed for '{volume_path_buffer.value}'.")
        allocation_unit_bytes = int(sectors_per_cluster.value) * int(bytes_per_sector.value)
        if allocation_unit_bytes <= 0:
            raise RuntimeError(f"Filesystem '{path}' did not report a positive allocation unit.")
        return allocation_unit_bytes, None

    filesystem = os.statvfs(path)
    allocation_unit_bytes = int(filesystem.f_frsize or filesystem.f_bsize)
    if allocation_unit_bytes <= 0:
        raise RuntimeError(f"Filesystem '{path}' did not report a positive allocation unit.")
    available_inodes = int(filesystem.f_favail)
    if available_inodes < 0:
        raise RuntimeError(f"Filesystem '{path}' reported a negative available-inode count.")
    return allocation_unit_bytes, available_inodes


def _round_up_to_allocation_unit(value: int, allocation_unit_bytes: int) -> int:
    if value < 0:
        raise RuntimeError(f"Scratch byte estimate cannot be negative, got {value}.")
    return ((value + allocation_unit_bytes - 1) // allocation_unit_bytes) * allocation_unit_bytes


def _preflight_video_upscale_source(
    video_path: str,
    options: SeedVR2UpscaleOptions,
) -> VideoUpscaleSourceAdmission:
    source_path = Path(video_path).expanduser().resolve()
    if not source_path.is_file() or not os.access(source_path, os.R_OK):
        raise HTTPException(status_code=400, detail="'video_path' must identify a readable file")
    source_suffix = source_path.suffix.lower()
    if source_suffix not in _SEEDVR2_SOURCE_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                "SeedVR2 source suffix is unsupported by the pinned child runtime: "
                f"got={source_suffix or '<none>'!r}, supported={', '.join(_SEEDVR2_SOURCE_SUFFIXES)}."
            ),
        )
    try:
        source_size_bytes = int(source_path.stat().st_size)
        if source_size_bytes <= 0:
            raise HTTPException(status_code=400, detail="'video_path' must identify a non-empty video file")
        work_capacity_path = _nearest_existing_capacity_path(video_upscale_work_root())
        output_capacity_path = _nearest_existing_capacity_path(video_export_output_root())
        work_usage = shutil.disk_usage(work_capacity_path)
        output_usage = shutil.disk_usage(output_capacity_path)
        work_allocation_unit_bytes, work_scratch_available_inodes = (
            _filesystem_allocation_unit_and_available_inodes(work_capacity_path)
        )
        output_allocation_unit_bytes, output_scratch_available_inodes = (
            _filesystem_allocation_unit_and_available_inodes(output_capacity_path)
        )
        work_root_missing_inodes = _missing_directory_count(
            video_upscale_work_root(),
            existing_ancestor=work_capacity_path,
        )
        shared_scratch_filesystem = os.stat(work_capacity_path).st_dev == os.stat(output_capacity_path).st_dev
    except HTTPException:
        raise
    except Exception as exc:
        _router_log.error("video-upscale scratch admission failed: %s", exc, exc_info=False)
        raise HTTPException(
            status_code=500,
            detail="SeedVR2 admission could not read current task/output filesystem capacity.",
        ) from None

    work_scratch_available_bytes = int(work_usage.free)
    output_scratch_available_bytes = int(output_usage.free)
    snapshot_allocation_bytes = _round_up_to_allocation_unit(source_size_bytes, work_allocation_unit_bytes)
    if snapshot_allocation_bytes + _SCRATCH_OPERATIONAL_RESERVE_BYTES > work_scratch_available_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                "SeedVR2 source snapshot exceeds current task-work scratch admission: "
                f"required_bytes={snapshot_allocation_bytes}, available_bytes={work_scratch_available_bytes}, "
                f"reserve_bytes={_SCRATCH_OPERATIONAL_RESERVE_BYTES}."
            ),
        )
    snapshot_required_inodes = work_root_missing_inodes + _WORK_SNAPSHOT_FIXED_INODES_AFTER_ROOT
    if (
        work_scratch_available_inodes is not None
        and work_scratch_available_inodes < snapshot_required_inodes
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "SeedVR2 source snapshot exceeds current task-work inode admission: "
                f"required_inodes={snapshot_required_inodes}, "
                f"available_inodes={work_scratch_available_inodes}."
            ),
        )

    work_dir: Path | None = None
    try:
        work_dir = create_video_upscale_work_dir()
        snapshot_path = work_dir / f"source_snapshot{source_suffix}"
        with source_path.open("rb") as source_file, snapshot_path.open("xb") as snapshot_file:
            shutil.copyfileobj(source_file, snapshot_file, length=8 * _MIB)
            snapshot_file.flush()
            os.fsync(snapshot_file.fileno())
        snapshot_size_bytes = int(snapshot_path.stat().st_size)
        if snapshot_size_bytes <= 0:
            raise HTTPException(status_code=400, detail="SeedVR2 could not create a non-empty source snapshot.")

        try:
            source_probe = probe_video(str(snapshot_path), count_frames=True)
        except Exception as exc:
            _router_log.warning("video-upscale source probe failed for %s: %s", source_path, exc, exc_info=False)
            raise HTTPException(status_code=400, detail="'video_path' must identify a readable video file") from None
        source_frame_count = source_probe.decoded_frame_count
        if source_frame_count is None or source_frame_count <= 0:
            raise HTTPException(
                status_code=400,
                detail="Source video does not expose the exact decoded frame count required for SeedVR2 admission.",
            )
        if not source_probe.video_codec:
            raise HTTPException(status_code=400, detail="Source video does not expose a verifiable video codec.")
        if source_probe.has_audio and (
            not source_probe.audio_codec
            or source_probe.audio_duration_seconds is None
            or source_probe.audio_start_seconds is None
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Source audio stream does not expose codec, duration, and start-time evidence "
                    "required for verified preservation."
                ),
            )

        try:
            target_width, target_height, padded_width, padded_height = calculate_seedvr2_target_dimensions(
                source_width=source_probe.width,
                source_height=source_probe.height,
                options=options,
            )
        except Exception as exc:
            _router_log.warning(
                "video-upscale target geometry admission failed for %s: %s",
                source_path,
                exc,
                exc_info=False,
            )
            raise HTTPException(
                status_code=400,
                detail="Source video and target resolution produce invalid SeedVR2 geometry.",
            ) from None

        try:
            import psutil  # type: ignore

            host_available_bytes = int(psutil.virtual_memory().available)
            host_memory = calculate_seedvr2_host_memory_admission(
                source_width=int(source_probe.width),
                source_height=int(source_probe.height),
                source_frame_count=int(source_frame_count),
                target_width=target_width,
                target_height=target_height,
                padded_width=padded_width,
                padded_height=padded_height,
                options=options,
                available_bytes=host_available_bytes,
            )
        except Exception as exc:
            _router_log.error("video-upscale host-memory admission failed: %s", exc, exc_info=False)
            raise HTTPException(
                status_code=500,
                detail="SeedVR2 admission could not calculate current host-memory capacity.",
            ) from None

        if bool(options.streaming) and not host_memory.streaming_admitted:
            raise HTTPException(
                status_code=400,
                detail=(
                    "SeedVR2 streaming cannot admit one frame within current host memory: "
                    f"required_bytes={max(host_memory.streaming_first.required_bytes, host_memory.streaming_steady.required_bytes)}, "
                    f"available_bytes={host_memory.available_bytes}, "
                    f"reserve_bytes={host_memory.operational_reserve_bytes}."
                ),
            )
        if bool(options.smart_fallback) and not host_memory.streaming_admitted:
            raise HTTPException(
                status_code=400,
                detail=(
                    "SeedVR2 smart fallback cannot admit its required streaming route within current host memory: "
                    f"required_bytes={max(host_memory.streaming_first.required_bytes, host_memory.streaming_steady.required_bytes)}, "
                    f"available_bytes={host_memory.available_bytes}, "
                    f"reserve_bytes={host_memory.operational_reserve_bytes}."
                ),
            )
        if not bool(options.streaming) and not bool(options.smart_fallback) and not host_memory.direct_admitted:
            raise HTTPException(
                status_code=400,
                detail=(
                    "SeedVR2 direct execution exceeds current host-memory admission and smart fallback is disabled: "
                    f"required_bytes={host_memory.direct.required_bytes}, "
                    f"available_bytes={host_memory.available_bytes}, "
                    f"reserve_bytes={host_memory.operational_reserve_bytes}."
                ),
            )

        padded_frame_pixels = padded_width * padded_height
        work_frame_count = source_frame_count + int(options.prepend_frames or 0)
        work_frame_allocation_bytes = _round_up_to_allocation_unit(
            padded_frame_pixels * _WORK_SCRATCH_BYTES_PER_PADDED_PIXEL,
            work_allocation_unit_bytes,
        )
        work_scratch_required_bytes = (
            _round_up_to_allocation_unit(snapshot_size_bytes, work_allocation_unit_bytes)
            + _round_up_to_allocation_unit(_WORK_SCRATCH_BASE_BYTES, work_allocation_unit_bytes)
            + work_frame_count * work_frame_allocation_bytes
        )
        output_scratch_copied_audio_bytes = (
            _round_up_to_allocation_unit(snapshot_size_bytes, output_allocation_unit_bytes)
            if source_probe.has_audio
            else 0
        )
        output_scratch_required_bytes = _round_up_to_allocation_unit(
            _OUTPUT_SCRATCH_BASE_BYTES
            + source_frame_count * padded_frame_pixels * _OUTPUT_SCRATCH_BYTES_PER_PADDED_PIXEL,
            output_allocation_unit_bytes,
        ) + output_scratch_copied_audio_bytes
        work_scratch_required_inodes = (
            work_root_missing_inodes
            + _WORK_EXECUTION_FIXED_INODES_AFTER_ROOT
            + (_WORK_SMART_FALLBACK_RETRY_INODES if bool(options.smart_fallback) else 0)
            + work_frame_count
        )
        output_scratch_required_inodes = _OUTPUT_SCRATCH_FIXED_INODES
        if shared_scratch_filesystem:
            combined_required_bytes = work_scratch_required_bytes + output_scratch_required_bytes
            if combined_required_bytes + _SCRATCH_OPERATIONAL_RESERVE_BYTES > work_scratch_available_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SeedVR2 source exceeds current shared scratch admission: "
                        f"required_bytes={combined_required_bytes}, available_bytes={work_scratch_available_bytes}, "
                        f"reserve_bytes={_SCRATCH_OPERATIONAL_RESERVE_BYTES}."
                    ),
                )
            combined_required_inodes = work_scratch_required_inodes + output_scratch_required_inodes
            if (
                work_scratch_available_inodes is not None
                and combined_required_inodes > work_scratch_available_inodes
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SeedVR2 source exceeds current shared scratch inode admission: "
                        f"required_inodes={combined_required_inodes}, "
                        f"available_inodes={work_scratch_available_inodes}."
                    ),
                )
        else:
            if work_scratch_required_bytes + _SCRATCH_OPERATIONAL_RESERVE_BYTES > work_scratch_available_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SeedVR2 source exceeds current task-work scratch admission: "
                        f"required_bytes={work_scratch_required_bytes}, available_bytes={work_scratch_available_bytes}, "
                        f"reserve_bytes={_SCRATCH_OPERATIONAL_RESERVE_BYTES}."
                    ),
                )
            if (
                work_scratch_available_inodes is not None
                and work_scratch_required_inodes > work_scratch_available_inodes
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SeedVR2 source exceeds current task-work inode admission: "
                        f"required_inodes={work_scratch_required_inodes}, "
                        f"available_inodes={work_scratch_available_inodes}."
                    ),
                )
            if output_scratch_required_bytes + _SCRATCH_OPERATIONAL_RESERVE_BYTES > output_scratch_available_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SeedVR2 source exceeds current output scratch admission: "
                        f"required_bytes={output_scratch_required_bytes}, available_bytes={output_scratch_available_bytes}, "
                        f"reserve_bytes={_SCRATCH_OPERATIONAL_RESERVE_BYTES}."
                    ),
                )
            if (
                output_scratch_available_inodes is not None
                and output_scratch_required_inodes > output_scratch_available_inodes
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SeedVR2 source exceeds current output inode admission: "
                        f"required_inodes={output_scratch_required_inodes}, "
                        f"available_inodes={output_scratch_available_inodes}."
                    ),
                )

        try:
            source_timing = probe_video_timing(str(snapshot_path), source_probe=source_probe)
        except Exception as exc:
            _router_log.warning(
                "video-upscale source timing admission failed for %s: %s",
                source_path,
                exc,
                exc_info=False,
            )
            raise HTTPException(
                status_code=400,
                detail="Source video does not expose decoded timing required for verified SeedVR2 export.",
            ) from None
        if source_timing.frame_count != source_frame_count:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Source decoded frame evidence changed during admission: "
                    f"count_probe={source_frame_count}, timing_probe={source_timing.frame_count}."
                ),
            )
        try:
            validate_timestamped_mp4_source(source_probe=source_probe, timing=source_timing)
        except VideoExportError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        return VideoUpscaleSourceAdmission(
            source_name=source_path.name,
            snapshot_path=str(snapshot_path),
            snapshot_size_bytes=snapshot_size_bytes,
            work_dir=str(work_dir),
            probe=source_probe,
            timing=source_timing,
            target_width=target_width,
            target_height=target_height,
            padded_width=padded_width,
            padded_height=padded_height,
            host_memory=host_memory,
            work_scratch_required_bytes=work_scratch_required_bytes,
            work_scratch_available_bytes=work_scratch_available_bytes,
            work_scratch_allocation_unit_bytes=work_allocation_unit_bytes,
            work_scratch_required_inodes=work_scratch_required_inodes,
            work_scratch_available_inodes=work_scratch_available_inodes,
            output_scratch_required_bytes=output_scratch_required_bytes,
            output_scratch_available_bytes=output_scratch_available_bytes,
            output_scratch_allocation_unit_bytes=output_allocation_unit_bytes,
            output_scratch_copied_audio_bytes=output_scratch_copied_audio_bytes,
            output_scratch_required_inodes=output_scratch_required_inodes,
            output_scratch_available_inodes=output_scratch_available_inodes,
            scratch_reserve_bytes=_SCRATCH_OPERATIONAL_RESERVE_BYTES,
            shared_scratch_filesystem=shared_scratch_filesystem,
        )
    except BaseException as exc:
        if work_dir is not None:
            try:
                cleanup_video_upscale_work_dir(work_dir)
            except RuntimeError as cleanup_error:
                _router_log.error(
                    "video-upscale preflight failed and source snapshot cleanup also failed: %s; cleanup=%s",
                    exc,
                    cleanup_error,
                    exc_info=False,
                )
                raise HTTPException(
                    status_code=500,
                    detail="SeedVR2 preflight failed and could not remove its task-owned source snapshot.",
                ) from exc
        raise


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
        video_path, device, options = _parse_video_upscale_request(payload)
        source_admission = await asyncio.to_thread(
            _preflight_video_upscale_source,
            video_path,
            options,
        )
        worker_owns_admission = False
        try:
            request = VideoUpscaleRequest(
                source=source_admission,
                device=device,
                options=options,
            )

            loop = asyncio.get_running_loop()
            entry = TaskEntry(loop)
            task_id = f"task(api-video-upscale-{uuid4().hex})"
            register_task(task_id, entry)

            from apps.backend.interfaces.api.tasks.video_upscale_tasks import run_video_upscale_task

            run_video_upscale_task(task_id=task_id, request=request, entry=entry)
            worker_owns_admission = True
            return {"task_id": task_id}
        finally:
            if not worker_owns_admission:
                cleanup_video_upscale_source_admission(source_admission)

    return router

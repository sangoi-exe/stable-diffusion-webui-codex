"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical SeedVR2 video-upscale use case.
Consumes one router-created immutable source snapshot plus host, allocation-block, and inode admission, runs the dedicated cancellable SeedVR2
child runner, then delegates timestamp-aware staged MP4 assembly, source-audio/A-V-origin verification, atomic publication, and cleanup.

Symbols (top-level; keep in sync; no ghosts):
- `VideoUpscaleSourceAdmission` (dataclass): Immutable source snapshot, exact media, geometry, and dynamic host/scratch evidence.
- `VideoUpscaleRequest` (dataclass): Typed dedicated video-upscale request accepted after router validation.
- `video_upscale_work_root` (function): Returns the task-work filesystem root used by admission and execution.
- `create_video_upscale_work_dir` (function): Creates the task-owned directory before source snapshot admission.
- `cleanup_video_upscale_work_dir` (function): Idempotently removes one exact task-owned work directory.
- `cleanup_video_upscale_source_admission` (function): Idempotently removes one admission's complete task-work directory.
- `run_video_upscale` (function): Runs the SeedVR2 source-video to verified exported-video pipeline and yields task events.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
from uuid import uuid4

from apps.backend.core.params.video import SeedVR2UpscaleOptions
from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent
from apps.backend.runtime.logging import get_backend_logger
from apps.backend.video.export.ffmpeg_exporter import (
    VideoExportResult,
    export_timestamped_video,
    video_export_output_root,
)
from apps.backend.video.io.ffmpeg import VideoProbe, VideoTiming
from apps.backend.video.upscaling.seedvr2 import SeedVR2HostMemoryAdmission, run_seedvr2_upscaling


logger = get_backend_logger(__name__)


@dataclass(frozen=True, slots=True)
class VideoUpscaleSourceAdmission:
    """Source media and dynamic resource evidence accepted before task creation."""

    source_name: str
    snapshot_path: str
    snapshot_size_bytes: int
    work_dir: str
    probe: VideoProbe
    timing: VideoTiming
    target_width: int
    target_height: int
    padded_width: int
    padded_height: int
    host_memory: SeedVR2HostMemoryAdmission
    work_scratch_required_bytes: int
    work_scratch_available_bytes: int
    work_scratch_allocation_unit_bytes: int
    work_scratch_required_inodes: int
    work_scratch_available_inodes: int
    output_scratch_required_bytes: int
    output_scratch_available_bytes: int
    output_scratch_allocation_unit_bytes: int
    output_scratch_required_inodes: int
    output_scratch_available_inodes: int
    scratch_reserve_bytes: int
    shared_scratch_filesystem: bool


@dataclass(frozen=True, slots=True)
class VideoUpscaleRequest:
    """Typed source-video request for the dedicated SeedVR2 utility route."""

    source: VideoUpscaleSourceAdmission
    device: str
    options: SeedVR2UpscaleOptions


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise RuntimeError("cancelled")


def video_upscale_work_root() -> Path:
    return Path.home() / ".cache" / "codex" / "seedvr2-dedicated-video-upscale"


def create_video_upscale_work_dir() -> Path:
    root = video_upscale_work_root()
    root.mkdir(parents=True, exist_ok=True)
    work_dir = root / f"run-{uuid4().hex}"
    work_dir.mkdir(parents=False, exist_ok=False)
    return work_dir


def _remove_published_export(export_result: VideoExportResult) -> None:
    artifact_root = Path(export_result.artifact_root).resolve() if export_result.artifact_root else None
    if artifact_root is not None and artifact_root.exists():
        output_root = video_export_output_root().resolve()
        try:
            artifact_root.relative_to(output_root)
        except ValueError as exc:
            raise RuntimeError(
                f"SeedVR2 refused to remove published artifact outside output root: '{artifact_root}'."
            ) from exc
        try:
            shutil.rmtree(artifact_root)
        except OSError as exc:
            raise RuntimeError(
                f"SeedVR2 failed to remove incomplete published artifact directory '{artifact_root}': {exc}"
            ) from exc


def cleanup_video_upscale_work_dir(work_dir: str | Path) -> None:
    resolved_work_dir = Path(work_dir).expanduser().resolve()
    root = video_upscale_work_root().expanduser().resolve()
    if resolved_work_dir.parent != root or not resolved_work_dir.name.startswith("run-"):
        raise RuntimeError(f"SeedVR2 refused to remove non-task work directory '{resolved_work_dir}'.")
    try:
        shutil.rmtree(resolved_work_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(f"SeedVR2 failed to remove task work directory '{resolved_work_dir}': {exc}") from exc


def cleanup_video_upscale_source_admission(admission: VideoUpscaleSourceAdmission) -> None:
    """Remove the task-owned snapshot and all work artifacts for one admitted request."""

    cleanup_video_upscale_work_dir(admission.work_dir)


def _require_saved_export(export_result: VideoExportResult) -> VideoExportResult:
    if (
        not export_result.saved
        or not export_result.artifact_root
        or not export_result.path
        or not export_result.metadata_path
        or not export_result.rel_path
        or not export_result.mime
        or export_result.width is None
        or export_result.height is None
        or export_result.frame_count is None
        or not export_result.timing_verified
    ):
        reason = str(export_result.reason or "").strip()
        raise RuntimeError(
            "SeedVR2 timestamp-aware video export failed while output saving is required"
            + (f": {reason}" if reason else ".")
        )
    return export_result


def _source_metadata(admission: VideoUpscaleSourceAdmission) -> dict[str, object]:
    probe = admission.probe
    timing = admission.timing
    video_origin_seconds = float(timing.frames[0].presentation_seconds)
    audio_origin_seconds = float(probe.audio_start_seconds) if probe.audio_start_seconds is not None else None
    return {
        "name": admission.source_name,
        "snapshot_size_bytes": int(admission.snapshot_size_bytes),
        "width": int(probe.width),
        "height": int(probe.height),
        "fps": float(probe.fps),
        "decoded_frames": int(timing.frame_count),
        "duration_seconds": float(probe.duration_seconds) if probe.duration_seconds is not None else None,
        "video_codec": probe.video_codec,
        "video_origin_seconds": video_origin_seconds,
        "audio_codec": probe.audio_codec,
        "audio_duration_seconds": float(probe.audio_duration_seconds) if probe.audio_duration_seconds is not None else None,
        "audio_origin_seconds": audio_origin_seconds,
        "audio_video_origin_offset_seconds": (
            audio_origin_seconds - video_origin_seconds if audio_origin_seconds is not None else None
        ),
        "has_audio": bool(probe.has_audio),
    }


def _admission_metadata(admission: VideoUpscaleSourceAdmission) -> dict[str, object]:
    return {
        "target_size": {"width": admission.target_width, "height": admission.target_height},
        "padded_size": {"width": admission.padded_width, "height": admission.padded_height},
        "host_memory": admission.host_memory.as_dict(),
        "scratch": {
            "shared_filesystem": admission.shared_scratch_filesystem,
            "reserve_bytes": admission.scratch_reserve_bytes,
            "combined_required_bytes": (
                admission.work_scratch_required_bytes + admission.output_scratch_required_bytes
            ),
            "work": {
                "required_bytes": admission.work_scratch_required_bytes,
                "available_bytes": admission.work_scratch_available_bytes,
                "allocation_unit_bytes": admission.work_scratch_allocation_unit_bytes,
                "required_inodes": admission.work_scratch_required_inodes,
                "available_inodes": admission.work_scratch_available_inodes,
            },
            "output": {
                "required_bytes": admission.output_scratch_required_bytes,
                "available_bytes": admission.output_scratch_available_bytes,
                "allocation_unit_bytes": admission.output_scratch_allocation_unit_bytes,
                "required_inodes": admission.output_scratch_required_inodes,
                "available_inodes": admission.output_scratch_available_inodes,
            },
        },
    }


def run_video_upscale(
    request: VideoUpscaleRequest,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[InferenceEvent]:
    """Upscale a backend-visible source video with SeedVR2 and return one verified saved MP4 artifact."""

    admission = request.source
    source_path = Path(admission.snapshot_path).expanduser().resolve()
    if not source_path.is_file():
        raise RuntimeError("SeedVR2 admitted source snapshot does not point to a readable file.")
    work_dir = Path(admission.work_dir).expanduser().resolve()
    if not work_dir.is_dir() or source_path.parent != work_dir:
        raise RuntimeError("SeedVR2 admitted source snapshot is outside its task-owned work directory.")
    if source_path.stat().st_size != admission.snapshot_size_bytes:
        raise RuntimeError("SeedVR2 admitted source snapshot size changed before execution.")
    source_probe = admission.probe
    source_timing = admission.timing
    if source_probe.decoded_frame_count != source_timing.frame_count:
        raise RuntimeError(
            "SeedVR2 admitted source frame evidence is inconsistent: "
            f"probe={source_probe.decoded_frame_count}, timing={source_timing.frame_count}."
        )

    active_work_dir: Path | None = work_dir
    export_result: VideoExportResult | None = None
    result_emitted = False
    try:
        _raise_if_cancelled(should_cancel)
        yield ProgressEvent(stage="admission", percent=12.0, message="Source media and resource budget admitted")
        source_metadata = _source_metadata(admission)
        admission_metadata = _admission_metadata(admission)

        _raise_if_cancelled(should_cancel)
        yield ProgressEvent(stage="upscale", percent=35.0, message="Upscaling source frames with SeedVR2")
        seedvr2_result = run_seedvr2_upscaling(
            str(source_path),
            source_probe=source_probe,
            source_frame_count=source_timing.frame_count,
            output_dir=work_dir / "seedvr2_frames",
            options=request.options,
            component_device=request.device,
            host_memory=admission.host_memory,
            should_cancel=should_cancel,
            logger_=logger,
        )

        _raise_if_cancelled(should_cancel)
        yield ProgressEvent(
            stage="export",
            percent=85.0,
            message="Encoding, preserving source timing, and verifying media",
        )
        export_result = _require_saved_export(
            export_timestamped_video(
                seedvr2_result.frame_paths,
                timing=source_timing,
                task="video-upscale",
                filename_prefix="seedvr2_upscale",
                source_probe=source_probe,
                audio_source_path=str(source_path) if source_probe.has_audio else None,
                extra_metadata={
                    "seedvr2": {
                        "options": request.options.as_dict(),
                        "runtime": seedvr2_result.metadata,
                    },
                    "source": source_metadata,
                    "admission": admission_metadata,
                },
                should_cancel=should_cancel,
            )
        )

        cleanup_video_upscale_work_dir(work_dir)
        active_work_dir = None
        _raise_if_cancelled(should_cancel)
        output = {
            "width": int(export_result.width),
            "height": int(export_result.height),
            "fps": float(export_result.fps) if export_result.fps is not None else None,
            "frames": int(export_result.frame_count),
            "rel_path": export_result.rel_path,
            "mime": export_result.mime,
            "timing_verified": bool(export_result.timing_verified),
            "timing_contract": "source_decoded_pts_offsets_and_relative_av_origin",
        }
        info = {
            "task": "video_upscale",
            "seedvr2": {
                "options": request.options.as_dict(),
                "runtime": seedvr2_result.metadata,
            },
            "source": source_metadata,
            "admission": admission_metadata,
            "output": output,
            "audio": {
                "source_has_audio": bool(source_probe.has_audio),
                "preserved": bool(export_result.audio_verified),
                "codec": source_probe.audio_codec if source_probe.has_audio else None,
                "duration_seconds": source_probe.audio_duration_seconds if source_probe.has_audio else None,
                "source_origin_seconds": source_probe.audio_start_seconds if source_probe.has_audio else None,
                "source_av_offset_seconds": (
                    float(source_probe.audio_start_seconds) - float(source_timing.frames[0].presentation_seconds)
                    if source_probe.has_audio and source_probe.audio_start_seconds is not None
                    else None
                ),
            },
        }
        yield ResultEvent(
            payload={
                "images": [],
                "info": info,
                "video": {
                    "rel_path": export_result.rel_path,
                    "mime": export_result.mime,
                },
            }
        )
        result_emitted = True
    finally:
        cleanup_failures: list[str] = []
        if export_result is not None and not result_emitted:
            try:
                _remove_published_export(export_result)
            except RuntimeError as exc:
                cleanup_failures.append(str(exc))
        if active_work_dir is not None:
            try:
                cleanup_video_upscale_work_dir(active_work_dir)
            except RuntimeError as exc:
                cleanup_failures.append(str(exc))
        if cleanup_failures:
            raise RuntimeError("SeedVR2 cleanup failed: " + "; ".join(cleanup_failures))

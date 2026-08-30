"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical SeedVR2 video-upscale use case.
Consumes router-admitted source media and resource evidence, runs the dedicated cancellable SeedVR2 child runner, then delegates timestamp-aware
staged MP4 assembly, source-audio/A-V-origin verification, ordered publication, and cleanup to the existing video owners.

Symbols (top-level; keep in sync; no ghosts):
- `VideoUpscaleSourceAdmission` (dataclass): Exact source media, geometry, and dynamic host/scratch evidence accepted before task creation.
- `VideoUpscaleRequest` (dataclass): Typed dedicated video-upscale request accepted after router validation.
- `video_upscale_work_root` (function): Returns the task-work filesystem root used by admission and execution.
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
from apps.backend.video.export.ffmpeg_exporter import VideoExportResult, export_timestamped_video
from apps.backend.video.io.ffmpeg import VideoProbe, VideoTiming
from apps.backend.video.upscaling.seedvr2 import run_seedvr2_upscaling


logger = get_backend_logger(__name__)


@dataclass(frozen=True, slots=True)
class VideoUpscaleSourceAdmission:
    """Source media and dynamic resource evidence accepted before task creation."""

    path: str
    probe: VideoProbe
    timing: VideoTiming
    target_width: int
    target_height: int
    padded_width: int
    padded_height: int
    host_timing_required_bytes: int
    host_available_bytes: int
    host_reserve_bytes: int
    work_scratch_required_bytes: int
    work_scratch_available_bytes: int
    output_scratch_required_bytes: int
    output_scratch_available_bytes: int
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


def _new_task_work_dir() -> Path:
    root = video_upscale_work_root()
    root.mkdir(parents=True, exist_ok=True)
    work_dir = root / f"run-{uuid4().hex}"
    work_dir.mkdir(parents=False, exist_ok=False)
    return work_dir


def _remove_published_export(export_result: VideoExportResult) -> None:
    video_path = Path(export_result.path) if export_result.path else None
    metadata_path = Path(export_result.metadata_path) if export_result.metadata_path else None
    if video_path is not None and video_path.is_file():
        try:
            video_path.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"SeedVR2 failed to remove incomplete published video '{video_path}': {exc}"
            ) from exc
    if metadata_path is not None and metadata_path.is_file():
        try:
            metadata_path.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"SeedVR2 failed to remove incomplete published metadata '{metadata_path}': {exc}"
            ) from exc


def _remove_task_work_dir(work_dir: Path) -> None:
    try:
        shutil.rmtree(work_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(f"SeedVR2 failed to remove task work directory '{work_dir}': {exc}") from exc


def _require_saved_export(export_result: VideoExportResult) -> VideoExportResult:
    if (
        not export_result.saved
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


def _source_metadata(source: Path, probe: VideoProbe, timing: VideoTiming) -> dict[str, object]:
    video_origin_seconds = float(timing.frames[0].presentation_seconds)
    audio_origin_seconds = float(probe.audio_start_seconds) if probe.audio_start_seconds is not None else None
    return {
        "name": source.name,
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
        "host_timing": {
            "required_bytes": admission.host_timing_required_bytes,
            "available_bytes": admission.host_available_bytes,
            "reserve_bytes": admission.host_reserve_bytes,
        },
        "scratch": {
            "shared_filesystem": admission.shared_scratch_filesystem,
            "reserve_bytes": admission.scratch_reserve_bytes,
            "combined_required_bytes": (
                admission.work_scratch_required_bytes + admission.output_scratch_required_bytes
            ),
            "work": {
                "required_bytes": admission.work_scratch_required_bytes,
                "available_bytes": admission.work_scratch_available_bytes,
            },
            "output": {
                "required_bytes": admission.output_scratch_required_bytes,
                "available_bytes": admission.output_scratch_available_bytes,
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
    source_path = Path(admission.path).expanduser().resolve()
    if not source_path.is_file():
        raise RuntimeError("SeedVR2 source video path does not point to a readable file.")
    source_probe = admission.probe
    source_timing = admission.timing
    if source_probe.decoded_frame_count != source_timing.frame_count:
        raise RuntimeError(
            "SeedVR2 admitted source frame evidence is inconsistent: "
            f"probe={source_probe.decoded_frame_count}, timing={source_timing.frame_count}."
        )

    work_dir: Path | None = None
    export_result: VideoExportResult | None = None
    result_emitted = False
    try:
        _raise_if_cancelled(should_cancel)
        yield ProgressEvent(stage="admission", percent=12.0, message="Source media and resource budget admitted")
        source_metadata = _source_metadata(source_path, source_probe, source_timing)
        admission_metadata = _admission_metadata(admission)

        work_dir = _new_task_work_dir()
        _raise_if_cancelled(should_cancel)
        yield ProgressEvent(stage="upscale", percent=35.0, message="Upscaling source frames with SeedVR2")
        seedvr2_result = run_seedvr2_upscaling(
            str(source_path),
            source_probe=source_probe,
            source_frame_count=source_timing.frame_count,
            output_dir=work_dir / "seedvr2_frames",
            options=request.options,
            component_device=request.device,
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

        _remove_task_work_dir(work_dir)
        work_dir = None
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
        if work_dir is not None:
            try:
                _remove_task_work_dir(work_dir)
            except RuntimeError as exc:
                cleanup_failures.append(str(exc))
        if cleanup_failures:
            raise RuntimeError("SeedVR2 cleanup failed: " + "; ".join(cleanup_failures))

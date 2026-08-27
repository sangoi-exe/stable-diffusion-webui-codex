"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical SeedVR2 video-upscale use case.
Probes a backend-visible source video and its decoded-frame timing, runs the dedicated cancellable SeedVR2 child runner, then delegates
timestamp-aware staged MP4 assembly, source-audio verification, and publication to the canonical video-export owner.

Symbols (top-level; keep in sync; no ghosts):
- `VideoUpscaleRequest` (dataclass): Typed dedicated video-upscale request accepted after router validation.
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
from apps.backend.video.io.ffmpeg import VideoProbe, VideoTiming, probe_video, probe_video_timing
from apps.backend.video.upscaling.seedvr2 import run_seedvr2_upscaling


logger = get_backend_logger(__name__)


@dataclass(frozen=True, slots=True)
class VideoUpscaleRequest:
    """Typed source-video request for the dedicated SeedVR2 utility route."""

    video_path: str
    device: str
    options: SeedVR2UpscaleOptions


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise RuntimeError("cancelled")


def _new_task_work_dir() -> Path:
    root = Path.home() / ".cache" / "codex" / "seedvr2-dedicated-video-upscale"
    root.mkdir(parents=True, exist_ok=True)
    work_dir = root / f"run-{uuid4().hex}"
    work_dir.mkdir(parents=False, exist_ok=False)
    return work_dir


def _remove_published_export(export_result: VideoExportResult) -> None:
    for raw_path in (export_result.metadata_path, export_result.path):
        if not raw_path:
            continue
        artifact_path = Path(raw_path)
        try:
            if artifact_path.is_file():
                artifact_path.unlink()
        except Exception as exc:
            logger.warning("failed to remove incomplete SeedVR2 published artifact %s: %s", artifact_path, exc, exc_info=False)


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
    return {
        "name": source.name,
        "width": int(probe.width),
        "height": int(probe.height),
        "fps": float(probe.fps),
        "frames": int(probe.frame_count) if probe.frame_count is not None else None,
        "decoded_timing_frames": int(timing.frame_count),
        "duration_seconds": float(probe.duration_seconds) if probe.duration_seconds is not None else None,
        "video_codec": probe.video_codec,
        "audio_codec": probe.audio_codec,
        "audio_duration_seconds": float(probe.audio_duration_seconds) if probe.audio_duration_seconds is not None else None,
        "has_audio": bool(probe.has_audio),
    }


def run_video_upscale(
    request: VideoUpscaleRequest,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[InferenceEvent]:
    """Upscale a backend-visible source video with SeedVR2 and return one verified saved MP4 artifact."""

    source_path = Path(request.video_path).expanduser().resolve()
    if not source_path.is_file():
        raise RuntimeError("SeedVR2 source video path does not point to a readable file.")

    work_dir: Path | None = None
    export_result: VideoExportResult | None = None
    result_emitted = False
    try:
        _raise_if_cancelled(should_cancel)
        yield ProgressEvent(stage="probe", percent=5.0, message="Probing source video")
        source_probe = probe_video(str(source_path))

        _raise_if_cancelled(should_cancel)
        yield ProgressEvent(stage="timing", percent=12.0, message="Reading source frame timing")
        source_timing = probe_video_timing(str(source_path))
        source_metadata = _source_metadata(source_path, source_probe, source_timing)

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
                audio_source_path=str(source_path) if source_probe.has_audio else None,
                extra_metadata={
                    "seedvr2": {
                        "options": request.options.as_dict(),
                        "runtime": seedvr2_result.metadata,
                    },
                    "source": source_metadata,
                },
                should_cancel=should_cancel,
            )
        )

        _raise_if_cancelled(should_cancel)
        output = {
            "width": int(export_result.width),
            "height": int(export_result.height),
            "fps": float(export_result.fps) if export_result.fps is not None else None,
            "frames": int(export_result.frame_count),
            "rel_path": export_result.rel_path,
            "mime": export_result.mime,
            "timing_verified": bool(export_result.timing_verified),
            "timing_contract": "source_decoded_pts_offsets",
        }
        info = {
            "task": "video_upscale",
            "seedvr2": {
                "options": request.options.as_dict(),
                "runtime": seedvr2_result.metadata,
            },
            "source": source_metadata,
            "output": output,
            "audio": {
                "source_has_audio": bool(source_probe.has_audio),
                "preserved": bool(export_result.audio_verified),
                "codec": source_probe.audio_codec if source_probe.has_audio else None,
                "duration_seconds": source_probe.audio_duration_seconds if source_probe.has_audio else None,
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
        if export_result is not None and not result_emitted:
            _remove_published_export(export_result)
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)

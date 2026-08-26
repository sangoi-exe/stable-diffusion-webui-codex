"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical SeedVR2 video-upscale use case.
Probes a backend-visible source video, extracts its frames, runs the dedicated SeedVR2 runtime, exports an H.264 MP4 artifact,
and verifies source-audio preservation before returning a task result.

Symbols (top-level; keep in sync; no ghosts):
- `VideoUpscaleRequest` (dataclass): Typed dedicated video-upscale request accepted after router validation.
- `run_video_upscale` (function): Runs the SeedVR2 source-video to exported-video pipeline and yields task events.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

from apps.backend.core.params.video import SeedVR2UpscaleOptions
from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent
from apps.backend.infra.config.repo_root import repo_scratch_path
from apps.backend.runtime.logging import get_backend_logger
from apps.backend.video.export.ffmpeg_exporter import VideoExportResult, export_video
from apps.backend.video.io.ffmpeg import VideoProbe, extract_frames, probe_video
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


def _load_source_frames(frame_paths: list[str]) -> list[Any]:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Pillow is required for SeedVR2 video upscaling: {exc}") from exc

    frames: list[Any] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))
    if not frames:
        raise RuntimeError("SeedVR2 video upscaling extracted no source frames.")
    return frames


def _require_saved_export(export_result: VideoExportResult | None) -> VideoExportResult:
    if export_result is None:
        raise RuntimeError("SeedVR2 video export returned no result while output saving is required.")
    if not export_result.saved or not export_result.path or not export_result.rel_path or not export_result.mime:
        reason = str(export_result.reason or "").strip()
        raise RuntimeError(
            "SeedVR2 video export failed while output saving is required"
            + (f": {reason}" if reason else ".")
        )
    return export_result


def _source_metadata(source: Path, probe: VideoProbe) -> dict[str, object]:
    return {
        "name": source.name,
        "width": int(probe.width),
        "height": int(probe.height),
        "fps": float(probe.fps),
        "frames": int(probe.frame_count) if probe.frame_count is not None else None,
        "duration_seconds": float(probe.duration_seconds) if probe.duration_seconds is not None else None,
        "video_codec": probe.video_codec,
        "audio_codec": probe.audio_codec,
        "has_audio": bool(probe.has_audio),
    }


def run_video_upscale(
    request: VideoUpscaleRequest,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[InferenceEvent]:
    """Upscale a backend-visible source video with SeedVR2 and return one saved MP4 artifact."""

    source_path = Path(request.video_path).expanduser()
    if not source_path.is_file():
        raise RuntimeError("SeedVR2 source video path does not point to a readable file.")

    _raise_if_cancelled(should_cancel)
    yield ProgressEvent(stage="probe", percent=5.0, message="Probing source video")
    source_probe = probe_video(str(source_path))

    work_dir = repo_scratch_path("video_upscale", f"seedvr2_{uuid4().hex}")
    frames_dir = work_dir / "source_frames"
    try:
        _raise_if_cancelled(should_cancel)
        yield ProgressEvent(stage="decode", percent=15.0, message="Decoding source video frames")
        frame_paths = extract_frames(str(source_path), out_dir=str(frames_dir))
        source_frames = _load_source_frames(frame_paths)

        _raise_if_cancelled(should_cancel)
        yield ProgressEvent(stage="upscale", percent=35.0, message="Upscaling frames with SeedVR2")
        upscaled_frames, seedvr2_metadata = run_seedvr2_upscaling(
            source_frames,
            options=request.options,
            component_device=request.device,
            logger_=logger,
        )

        _raise_if_cancelled(should_cancel)
        yield ProgressEvent(stage="export", percent=85.0, message="Exporting upscaled video")
        export_fps = float(source_probe.fps)
        source_has_audio = bool(source_probe.has_audio)
        export_result = _require_saved_export(
            export_video(
                upscaled_frames,
                fps=export_fps,
                options={
                    "filename_prefix": "seedvr2_upscale",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 23,
                    "save_metadata": True,
                    "save_output": True,
                },
                task="video-upscale",
                audio_source_path=str(source_path) if source_has_audio else None,
                extra_metadata={
                    "seedvr2": request.options.as_dict(),
                    "source": _source_metadata(source_path, source_probe),
                    "source_audio_preservation_required": source_has_audio,
                },
            )
        )

        output_probe: VideoProbe | None = None
        if source_has_audio:
            _raise_if_cancelled(should_cancel)
            yield ProgressEvent(stage="verify_audio", percent=95.0, message="Verifying preserved source audio")
            output_probe = probe_video(str(export_result.path))
            if not output_probe.has_audio:
                raise RuntimeError("SeedVR2 export completed without the required preserved source audio stream.")

        output_width = int(upscaled_frames[0].size[0])
        output_height = int(upscaled_frames[0].size[1])
        info = {
            "task": "video_upscale",
            "seedvr2": {
                "options": request.options.as_dict(),
                "runtime": seedvr2_metadata,
            },
            "source": _source_metadata(source_path, source_probe),
            "output": {
                "width": output_width,
                "height": output_height,
                "fps": float(export_result.fps if export_result.fps is not None else export_fps),
                "frames": len(upscaled_frames),
                "rel_path": export_result.rel_path,
                "mime": export_result.mime,
            },
            "audio": {
                "source_has_audio": source_has_audio,
                "preserved": bool(output_probe.has_audio) if output_probe is not None else False,
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
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

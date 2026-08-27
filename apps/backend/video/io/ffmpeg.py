"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: ffmpeg/ffprobe-backed video IO helpers (probe metadata, decoded-frame timing, and extract frames).
Provides small fail-fast wrappers around `ffprobe` to parse video metadata and decoded-frame presentation timestamps, plus a frame-extraction
helper using `ffmpeg`, intended for backend video tasks without pulling heavy dependencies (no cv2), with deterministic binary resolution from
repo-local runtime paths.

Symbols (top-level; keep in sync; no ghosts):
- `FFmpegUnavailableError` (class): Raised when `ffmpeg`/`ffprobe` cannot be resolved by the shared runtime dependency resolver.
- `_which` (function): Resolves a required binary via shared resolver precedence (env override → deterministic runtime path → downloader/PATH).
- `_parse_ratio` (function): Parses ffprobe ratio strings (e.g. `30000/1001`) into floats.
- `VideoProbe` (dataclass): Parsed container and per-stream video/audio metadata returned by `probe_video`.
- `VideoFrameTiming` (dataclass): One decoded video frame's presentation timestamp and optional duration.
- `VideoTiming` (dataclass): Ordered decoded-frame timing evidence returned by `probe_video_timing`.
- `probe_video` (function): Runs `ffprobe` and returns a parsed `VideoProbe`.
- `probe_video_timing` (function): Runs `ffprobe` and returns ordered decoded-frame presentation timestamps.
- `extract_frames` (function): Extracts frames from a video into an output directory (ffmpeg subprocess).
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from apps.backend.video.runtime_dependencies import VideoDependencyResolutionError, resolve_ffmpeg_binary


class FFmpegUnavailableError(RuntimeError):
    pass


def _which(name: str) -> str:
    try:
        return resolve_ffmpeg_binary(name)
    except VideoDependencyResolutionError as exc:
        raise FFmpegUnavailableError(str(exc)) from exc


def _parse_ratio(raw: str) -> Optional[float]:
    value = str(raw or "").strip()
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        num_s, den_s = value.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
        except Exception:
            return None
        if den == 0:
            return None
        return num / den
    try:
        return float(value)
    except Exception:
        return None


@dataclass(frozen=True)
class VideoProbe:
    path: str
    width: int
    height: int
    fps: float
    duration_seconds: float | None
    frame_count: int | None
    has_audio: bool
    format_name: str | None = None
    video_codec: str | None = None
    video_duration_seconds: float | None = None
    audio_codec: str | None = None
    audio_duration_seconds: float | None = None


@dataclass(frozen=True)
class VideoFrameTiming:
    index: int
    presentation_seconds: float
    duration_seconds: float | None = None


@dataclass(frozen=True)
class VideoTiming:
    path: str
    frames: tuple[VideoFrameTiming, ...]
    duration_seconds: float | None = None

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def probe_video(path: str) -> VideoProbe:
    ffprobe = _which("ffprobe")
    p = str(path)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        p,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        msg = exc.output.decode("utf-8", errors="replace") if exc.output else str(exc)
        raise RuntimeError(f"ffprobe failed for '{p}': {msg}") from exc
    try:
        data = json.loads(out.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc

    streams = data.get("streams") if isinstance(data, dict) else None
    if not isinstance(streams, list):
        streams = []
    fmt = data.get("format") if isinstance(data, dict) else None
    fmt_name = str(fmt.get("format_name")).strip() if isinstance(fmt, dict) and fmt.get("format_name") else None
    duration = None
    if isinstance(fmt, dict) and fmt.get("duration") not in (None, "", "N/A"):
        try:
            duration = float(fmt.get("duration"))
        except Exception:
            duration = None

    vstream: dict[str, Any] | None = None
    astream: dict[str, Any] | None = None
    for s in streams:
        if not isinstance(s, dict):
            continue
        if s.get("codec_type") == "video" and vstream is None:
            vstream = s
        if s.get("codec_type") == "audio" and astream is None:
            astream = s

    if vstream is None:
        raise RuntimeError(f"ffprobe: no video stream found in '{p}'")

    width = int(vstream.get("width") or 0)
    height = int(vstream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"ffprobe: invalid dimensions for '{p}': {width}x{height}")

    fps = _parse_ratio(str(vstream.get("avg_frame_rate") or "")) or _parse_ratio(
        str(vstream.get("r_frame_rate") or "")
    )
    if fps is None or fps <= 0:
        raise RuntimeError(f"ffprobe: unable to determine fps for '{p}'")

    frame_count = None
    nb = vstream.get("nb_frames")
    if nb not in (None, "", "N/A"):
        try:
            frame_count = int(nb)
        except Exception:
            frame_count = None
    if frame_count is None and duration is not None:
        try:
            frame_count = max(1, int(round(duration * fps)))
        except Exception:
            frame_count = None

    vcodec = str(vstream.get("codec_name")).strip() if vstream.get("codec_name") else None
    video_duration = _parse_optional_float(vstream.get("duration"))
    acodec = str(astream.get("codec_name")).strip() if isinstance(astream, dict) and astream.get("codec_name") else None
    audio_duration = _parse_optional_float(astream.get("duration")) if isinstance(astream, dict) else None

    return VideoProbe(
        path=p,
        width=width,
        height=height,
        fps=float(fps),
        duration_seconds=duration,
        frame_count=frame_count,
        has_audio=astream is not None,
        format_name=fmt_name,
        video_codec=vcodec,
        video_duration_seconds=video_duration,
        audio_codec=acodec,
        audio_duration_seconds=audio_duration,
    )


def _parse_optional_float(raw: object) -> float | None:
    if raw in (None, "", "N/A"):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def probe_video_timing(path: str) -> VideoTiming:
    """Return decoded video-frame timing in presentation order.

    Average FPS is not sufficient for VFR output validation. This function uses
    ffprobe's decoded-frame evidence and fails loud when a frame lacks an
    unambiguous presentation timestamp.
    """

    source_probe = probe_video(path)
    ffprobe = _which("ffprobe")
    p = str(path)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_pts_time,pkt_dts_time,pkt_duration_time,duration_time",
        "-print_format",
        "json",
        p,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        msg = exc.output.decode("utf-8", errors="replace") if exc.output else str(exc)
        raise RuntimeError(f"ffprobe failed to read decoded-frame timing for '{p}': {msg}") from exc
    try:
        data = json.loads(out.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError("ffprobe returned invalid JSON for decoded-frame timing") from exc

    raw_frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(raw_frames, list):
        raise RuntimeError(f"ffprobe returned no decoded-frame timing records for '{p}'")

    frames: list[VideoFrameTiming] = []
    previous_presentation: float | None = None
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            continue
        presentation = None
        for key in ("best_effort_timestamp_time", "pkt_pts_time", "pkt_dts_time"):
            presentation = _parse_optional_float(raw_frame.get(key))
            if presentation is not None:
                break
        if presentation is None:
            raise RuntimeError(
                "ffprobe returned a decoded video frame without a presentation timestamp "
                f"for '{p}' at frame index {len(frames)}."
            )
        if previous_presentation is not None and presentation <= previous_presentation:
            raise RuntimeError(
                "ffprobe returned non-increasing decoded-frame presentation timestamps "
                f"for '{p}' at frame index {len(frames)}."
            )
        duration = _parse_optional_float(raw_frame.get("pkt_duration_time"))
        if duration is None:
            duration = _parse_optional_float(raw_frame.get("duration_time"))
        frames.append(
            VideoFrameTiming(
                index=len(frames),
                presentation_seconds=presentation,
                duration_seconds=duration if duration is None or duration > 0 else None,
            )
        )
        previous_presentation = presentation

    if not frames:
        raise RuntimeError(f"ffprobe returned 0 decoded video frames for '{p}'")

    return VideoTiming(
        path=p,
        frames=tuple(frames),
        duration_seconds=source_probe.duration_seconds,
    )


def extract_frames(
    video_path: str,
    *,
    out_dir: str,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[str]:
    ffmpeg = _which("ffmpeg")
    src = str(video_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    vf_parts: list[str] = []
    if width and height and width > 0 and height > 0:
        vf_parts.append(f"scale={int(width)}:{int(height)}:flags=lanczos")
    if fps and fps > 0:
        vf_parts.append(f"fps={float(fps):.6f}")
    vf = ",".join(vf_parts) if vf_parts else None

    cmd: list[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if start_seconds is not None and start_seconds >= 0:
        cmd += ["-ss", str(float(start_seconds))]
    if end_seconds is not None and end_seconds > 0:
        cmd += ["-to", str(float(end_seconds))]
    cmd += ["-i", src, "-an"]
    if vf:
        cmd += ["-vf", vf]
    if max_frames is not None and int(max_frames) > 0:
        cmd += ["-frames:v", str(int(max_frames))]

    cmd += ["-vsync", "0", str(out_path / "frame_%06d.png")]

    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        msg = exc.output.decode("utf-8", errors="replace") if exc.output else str(exc)
        raise RuntimeError(f"ffmpeg failed to extract frames: {msg}") from exc

    frames = sorted(str(p) for p in out_path.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("ffmpeg extracted 0 frames (check start/end/fps settings)")
    return frames

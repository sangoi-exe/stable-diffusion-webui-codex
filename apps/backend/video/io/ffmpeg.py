"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: ffmpeg/ffprobe-backed video IO helpers (probe metadata, decoded-frame timing, and extract frames).
Provides cancellable fail-fast wrappers around `ffprobe` to parse stream origins, exact decoded-frame counts, metadata, and presentation
timestamps, plus a frame-extraction helper using `ffmpeg`, with deterministic binary resolution from repo-local runtime paths.

Symbols (top-level; keep in sync; no ghosts):
- `FFmpegUnavailableError` (class): Raised when `ffmpeg`/`ffprobe` cannot be resolved by the shared runtime dependency resolver.
- `_which` (function): Resolves a required binary via shared resolver precedence (env override → deterministic runtime path → downloader/PATH).
- `_parse_ratio` (function): Parses ffprobe ratio strings (e.g. `30000/1001`) into floats.
- `VideoProbe` (dataclass): Parsed container, stream-origin, and optional exact decoded-frame metadata returned by `probe_video`.
- `VideoFrameTiming` (dataclass): One decoded video frame's presentation timestamp and optional duration.
- `VideoTiming` (dataclass): Ordered decoded-frame timing evidence returned by `probe_video_timing`.
- `probe_video` (function): Runs `ffprobe` and returns a parsed `VideoProbe`.
- `probe_video_timing` (function): Runs `ffprobe` and returns ordered decoded-frame presentation timestamps.
- `extract_frames` (function): Extracts frames from a video into an output directory (ffmpeg subprocess).
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from apps.backend.video.runtime_dependencies import VideoDependencyResolutionError, resolve_ffmpeg_binary


class FFmpegUnavailableError(RuntimeError):
    pass


_PROCESS_POLL_SECONDS = 0.05
_PROCESS_TERMINATE_TIMEOUT_SECONDS = 5.0


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
    decoded_frame_count: int | None
    has_audio: bool
    format_name: str | None = None
    video_codec: str | None = None
    video_duration_seconds: float | None = None
    video_start_seconds: float | None = None
    audio_codec: str | None = None
    audio_duration_seconds: float | None = None
    audio_start_seconds: float | None = None


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


def _posix_process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_posix_process_group_exit(
    process: subprocess.Popen[bytes],
    process_group_id: int,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        process.poll()
        if not _posix_process_group_exists(process_group_id):
            return True
        time.sleep(_PROCESS_POLL_SECONDS)
    return not _posix_process_group_exists(process_group_id)


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=_PROCESS_TERMINATE_TIMEOUT_SECONDS,
            )
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        try:
            process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_SECONDS)
        except Exception:
            pass
        return

    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass
    if not _wait_for_posix_process_group_exit(
        process,
        process_group_id,
        timeout_seconds=_PROCESS_TERMINATE_TIMEOUT_SECONDS,
    ):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass
        _wait_for_posix_process_group_exit(
            process,
            process_group_id,
            timeout_seconds=_PROCESS_TERMINATE_TIMEOUT_SECONDS,
        )
    try:
        process.kill()
    except Exception:
        pass
    try:
        process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_SECONDS)
    except Exception:
        pass


def _run_media_subprocess(
    command: Sequence[str],
    *,
    purpose: str,
    should_cancel: Callable[[], bool] | None,
) -> bytes:
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(list(command), **popen_kwargs)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{purpose} failed because required executable '{command[0]}' was not found.") from exc
    except Exception as exc:
        raise RuntimeError(f"{purpose} failed to start: {exc}") from exc

    leader_exited_at: float | None = None
    while True:
        try:
            output, _ = process.communicate(timeout=_PROCESS_POLL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            if should_cancel is not None and should_cancel():
                _terminate_process_tree(process)
                process.communicate()
                raise RuntimeError("cancelled")
            if process.poll() is None:
                leader_exited_at = None
                continue
            if leader_exited_at is None:
                leader_exited_at = time.monotonic()
                continue
            if time.monotonic() - leader_exited_at < _PROCESS_TERMINATE_TIMEOUT_SECONDS:
                continue
            _terminate_process_tree(process)
            output, _ = process.communicate()
            break

    _terminate_process_tree(process)
    if process.returncode == 0:
        return output or b""
    message = (output or b"").decode("utf-8", errors="replace")
    raise RuntimeError(f"{purpose} failed with exit {process.returncode}: {message}")


def probe_video(
    path: str,
    *,
    count_frames: bool = False,
    should_cancel: Callable[[], bool] | None = None,
) -> VideoProbe:
    ffprobe = _which("ffprobe")
    p = str(path)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
    ]
    if count_frames:
        cmd.append("-count_frames")
    cmd += ["-show_streams", "-show_format", p]
    out = _run_media_subprocess(
        cmd,
        purpose=f"ffprobe metadata read for '{p}'",
        should_cancel=should_cancel,
    )
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

    decoded_frame_count = None
    if count_frames:
        nb_read = vstream.get("nb_read_frames")
        if nb_read not in (None, "", "N/A"):
            try:
                decoded_frame_count = int(nb_read)
            except Exception:
                decoded_frame_count = None
        if decoded_frame_count is not None and decoded_frame_count <= 0:
            decoded_frame_count = None

    frame_count = decoded_frame_count
    nb = vstream.get("nb_frames")
    if frame_count is None and nb not in (None, "", "N/A"):
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
    video_start = _parse_optional_float(vstream.get("start_time"))
    acodec = str(astream.get("codec_name")).strip() if isinstance(astream, dict) and astream.get("codec_name") else None
    audio_duration = _parse_optional_float(astream.get("duration")) if isinstance(astream, dict) else None
    audio_start = _parse_optional_float(astream.get("start_time")) if isinstance(astream, dict) else None

    return VideoProbe(
        path=p,
        width=width,
        height=height,
        fps=float(fps),
        duration_seconds=duration,
        frame_count=frame_count,
        decoded_frame_count=decoded_frame_count,
        has_audio=astream is not None,
        format_name=fmt_name,
        video_codec=vcodec,
        video_duration_seconds=video_duration,
        video_start_seconds=video_start,
        audio_codec=acodec,
        audio_duration_seconds=audio_duration,
        audio_start_seconds=audio_start,
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


def probe_video_timing(
    path: str,
    *,
    source_probe: VideoProbe | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> VideoTiming:
    """Return decoded video-frame timing in presentation order.

    Average FPS is not sufficient for VFR output validation. This function uses
    ffprobe's decoded-frame evidence and fails loud when a frame lacks an
    unambiguous presentation timestamp.
    """

    source_probe = source_probe or probe_video(path, should_cancel=should_cancel)
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
    out = _run_media_subprocess(
        cmd,
        purpose=f"ffprobe decoded-frame timing read for '{p}'",
        should_cancel=should_cancel,
    )
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
        if should_cancel is not None and should_cancel():
            raise RuntimeError("cancelled")
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
    should_cancel: Callable[[], bool] | None = None,
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

    _run_media_subprocess(
        cmd,
        purpose="ffmpeg frame extraction",
        should_cancel=should_cancel,
    )

    frames: list[str] = []
    for frame_path in sorted(out_path.glob("frame_*.png")):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("cancelled")
        frames.append(str(frame_path))
    if not frames:
        raise RuntimeError("ffmpeg extracted 0 frames (check start/end/fps settings)")
    return frames

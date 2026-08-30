"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Encode frame sequences to a video container via ffmpeg (mp4/webm/gif).
Writes CFR frame sequences to a workspace-local temp dir, and also validates and builds verified timestamp-aware MP4 artifacts from staged PNG
paths before atomic directory publication under `/api/output/{rel_path}`, including terminal VFR duration and relative source A/V-origin proof.

Symbols (top-level; keep in sync; no ghosts):
- `VideoExportError` (class): Explicit export error surfaced when ffmpeg/Pillow or encoding fails.
- `_which` (function): Resolves ffmpeg executable paths via shared resolver precedence (env override → deterministic runtime path → downloader/PATH).
- `video_export_output_root` (function): Resolves the repo-local output root (`CODEX_ROOT/output`).
- `_sanitize_filename_prefix` (function): Sanitizes a user/task-provided filename prefix for safe output paths.
- `resolve_video_export_container` (function): Maps a format token to an output container + codec kind.
- `validate_timestamped_mp4_source` (function): Validates source timing and AAC stream-copy evidence before SeedVR2 execution.
- `_audio_codec_for` (function): Chooses an audio codec for a given output container.
- `VideoExportResult` (dataclass): Export result container (saved flag + artifact root/path/rel_path/mime + metadata).
- `export_video` (function): Main entrypoint; writes frames and runs ffmpeg to produce the final video file.
- `export_timestamped_video` (function): Builds, verifies, and atomically publishes a VFR MP4 plus metadata from ordered frame paths.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from apps.backend.core.strict_values import parse_bool_value
from apps.backend.video.export.mp4_timing import (
    Mp4TimingError,
    expected_video_duration_seconds,
    patch_terminal_frame_duration,
    terminal_frame_duration_seconds,
)
from apps.backend.infra.config.repo_root import get_repo_root, repo_scratch_path
from apps.backend.video.io.ffmpeg import VideoProbe, VideoTiming, probe_video, probe_video_timing
from apps.backend.video.runtime_dependencies import VideoDependencyResolutionError, resolve_ffmpeg_binary


class VideoExportError(RuntimeError):
    pass


def _which(name: str) -> str:
    try:
        return resolve_ffmpeg_binary(name)
    except VideoDependencyResolutionError as exc:
        raise VideoExportError(str(exc)) from exc


def video_export_output_root() -> Path:
    return get_repo_root() / "output"


_FILENAME_PREFIX_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename_prefix(prefix: str) -> str:
    raw = str(prefix or "").strip()
    if not raw:
        return "video"
    raw = raw.replace("/", "_").replace("\\", "_")
    cleaned = _FILENAME_PREFIX_RE.sub("_", raw).strip("._-")
    if not cleaned:
        return "video"
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip("._-") or "video"
    return cleaned


def resolve_video_export_container(fmt: str) -> tuple[str, str]:
    v = (fmt or "").strip().lower()
    if v in {"video/h264-mp4", "h264", "mp4", "video/mp4"}:
        return "mp4", "h264"
    if v in {"video/h265-mp4", "h265", "hevc", "video/hevc"}:
        return "mp4", "h265"
    if v in {"video/webm", "webm"}:
        return "webm", "vp9"
    if v in {"video/gif", "image/gif", "gif"}:
        return "gif", "gif"
    raise VideoExportError(
        f"Unsupported video format '{fmt}'. Supported values: video/h264-mp4, video/h265-mp4, video/webm, video/gif."
    )


def _audio_codec_for(container: str) -> str | None:
    if container == "mp4":
        return "aac"
    if container == "webm":
        return "libopus"
    return None


@dataclass(frozen=True)
class VideoExportResult:
    saved: bool
    artifact_root: str | None = None
    path: str | None = None
    metadata_path: str | None = None
    rel_path: str | None = None
    mime: str | None = None
    reason: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    has_audio: bool = False
    audio_verified: bool = False
    timing_verified: bool = False


_TIMING_TOLERANCE_SECONDS = 0.001
_AUDIO_DURATION_TOLERANCE_SECONDS = 0.050
_AUDIO_ORIGIN_TOLERANCE_SECONDS = 0.001
_TIMESTAMP_MANIFEST_FRAMERATE = 1_000_000
_TIMESTAMPED_MP4_AUDIO_CODEC = "aac"
_PROCESS_OUTPUT_LIMIT = 4000
_PROCESS_TERMINATE_TIMEOUT_SECONDS = 5.0


def _terminal_frame_duration_seconds(timing: VideoTiming) -> float:
    try:
        return terminal_frame_duration_seconds(timing)
    except Mp4TimingError as exc:
        raise VideoExportError(str(exc)) from exc


def _expected_timestamped_video_duration(timing: VideoTiming) -> float:
    try:
        return expected_video_duration_seconds(timing)
    except Mp4TimingError as exc:
        raise VideoExportError(str(exc)) from exc


def validate_timestamped_mp4_source(*, source_probe: VideoProbe, timing: VideoTiming) -> None:
    """Validate the fixed timestamped H.264 MP4 source contract before expensive execution."""

    if not timing.frames:
        raise VideoExportError("Timestamp-aware export requires decoded source timing.")
    if Path(source_probe.path).expanduser().resolve() != Path(timing.path).expanduser().resolve():
        raise VideoExportError("Timestamp-aware export source probe and decoded timing refer to different files.")
    if source_probe.decoded_frame_count != timing.frame_count:
        raise VideoExportError(
            "Timestamp-aware export source frame evidence is inconsistent: "
            f"probe={source_probe.decoded_frame_count}, timing={timing.frame_count}."
        )
    first_video_origin = float(timing.frames[0].presentation_seconds)
    if not math.isfinite(first_video_origin):
        raise VideoExportError("Timestamp-aware export requires a finite decoded video-stream origin.")
    _expected_timestamped_video_duration(timing)

    if not source_probe.has_audio:
        return
    audio_codec = str(source_probe.audio_codec or "").strip().lower()
    if audio_codec != _TIMESTAMPED_MP4_AUDIO_CODEC:
        raise VideoExportError(
            "Timestamp-aware MP4 export supports stream-copy source audio only when the codec is "
            f"'{_TIMESTAMPED_MP4_AUDIO_CODEC}'; got {audio_codec or 'unknown'!r}."
        )
    audio_duration = source_probe.audio_duration_seconds
    if audio_duration is None or not math.isfinite(float(audio_duration)) or float(audio_duration) <= 0.0:
        raise VideoExportError("Timestamp-aware export requires a positive finite source-audio duration.")
    audio_origin = source_probe.audio_start_seconds
    if audio_origin is None or not math.isfinite(float(audio_origin)):
        raise VideoExportError("Timestamp-aware export requires a finite source-audio stream origin.")


def _patch_mp4_terminal_duration(*, staged_path: Path, timing: VideoTiming) -> None:
    try:
        patch_terminal_frame_duration(
            staged_path=staged_path,
            timing=timing,
            timing_tolerance_seconds=_TIMING_TOLERANCE_SECONDS,
        )
    except Mp4TimingError as exc:
        raise VideoExportError(str(exc)) from exc
    except OSError as exc:
        raise VideoExportError(f"Timestamp-aware MP4 terminal-duration patch failed: {exc}") from exc


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise RuntimeError("cancelled")


def _posix_process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_posix_process_group_exit(
    process: subprocess.Popen[str],
    process_group_id: int,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        process.poll()
        if not _posix_process_group_exists(process_group_id):
            return True
        time.sleep(0.05)
    return not _posix_process_group_exists(process_group_id)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
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


def _run_cancellable_ffmpeg(
    command: Sequence[str],
    *,
    should_cancel: Callable[[], bool] | None,
    failure_label: str,
) -> None:
    """Run one ffmpeg child while preserving the task cancellation boundary."""

    try:
        popen_kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(list(command), **popen_kwargs)
    except FileNotFoundError as exc:
        raise VideoExportError(f"{failure_label} failed because ffmpeg was not found.") from exc
    except Exception as exc:
        raise VideoExportError(f"{failure_label} failed to start: {exc}") from exc

    output_parts: list[str] = []
    output_size = 0

    def drain_output() -> None:
        nonlocal output_size
        if process.stdout is None:
            return
        for line in process.stdout:
            remaining = _PROCESS_OUTPUT_LIMIT - output_size
            if remaining <= 0:
                continue
            output_parts.append(line[:remaining])
            output_size += len(output_parts[-1])

    reader = threading.Thread(target=drain_output, name="video-export-ffmpeg-output", daemon=True)
    reader.start()
    cancellation_observed = False
    while True:
        if process.poll() is not None:
            break
        if should_cancel is not None and should_cancel():
            cancellation_observed = True
            _terminate_process(process)
            break
        time.sleep(0.05)

    if not cancellation_observed:
        _terminate_process(process)
    reader.join(timeout=_PROCESS_TERMINATE_TIMEOUT_SECONDS)
    if reader.is_alive():
        _terminate_process(process)
        reader.join()
    if cancellation_observed:
        raise RuntimeError("cancelled")

    if process.returncode == 0:
        return
    output = "".join(output_parts).strip()
    if len(output) > _PROCESS_OUTPUT_LIMIT:
        output = output[:_PROCESS_OUTPUT_LIMIT]
    detail = f": {output}" if output else ""
    raise VideoExportError(f"{failure_label} failed with exit {process.returncode}{detail}")


def _ffconcat_quote(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace("'", "\\'")


def _write_timestamp_manifest(
    *,
    manifest_path: Path,
    frame_paths: Sequence[str | Path],
    timing: VideoTiming,
    should_cancel: Callable[[], bool] | None,
) -> None:
    if len(frame_paths) != timing.frame_count:
        raise VideoExportError(
            "Timestamp-aware export frame/timing count mismatch: "
            f"frames={len(frame_paths)}, timing={timing.frame_count}."
        )
    if not frame_paths:
        raise VideoExportError("Timestamp-aware export requires at least one frame path.")

    offsets = [
        float(frame.presentation_seconds) - float(timing.frames[0].presentation_seconds)
        for frame in timing.frames
    ]
    if offsets[0] != 0.0:
        raise VideoExportError("Timestamp-aware export could not normalize the first presentation timestamp.")
    for index in range(1, len(offsets)):
        if offsets[index] <= offsets[index - 1]:
            raise VideoExportError(
                "Timestamp-aware export requires strictly increasing source presentation timestamps; "
                f"frame {index - 1}={offsets[index - 1]:.9f}, frame {index}={offsets[index]:.9f}."
            )

    lines = ["ffconcat version 1.0"]
    for index, raw_path in enumerate(frame_paths):
        _raise_if_cancelled(should_cancel)
        frame_path = Path(raw_path).resolve()
        if not frame_path.is_file():
            raise VideoExportError(f"Timestamp-aware export frame is missing: '{frame_path}'.")
        lines.append(f"file '{_ffconcat_quote(frame_path)}'")
        lines.append(f"option framerate {_TIMESTAMP_MANIFEST_FRAMERATE}")
        if index + 1 < len(offsets):
            duration = offsets[index + 1] - offsets[index]
        else:
            duration = _terminal_frame_duration_seconds(timing)
        lines.append(f"duration {format(duration, '.12g')}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_timestamped_export(
    *,
    staged_path: Path,
    expected_timing: VideoTiming,
    source_audio_probe: VideoProbe | None,
    should_cancel: Callable[[], bool] | None,
) -> tuple[VideoProbe, VideoTiming, float | None, float | None]:
    output_probe = probe_video(str(staged_path), should_cancel=should_cancel)
    output_timing = probe_video_timing(
        str(staged_path),
        source_probe=output_probe,
        should_cancel=should_cancel,
    )
    expected_video_duration = _expected_timestamped_video_duration(expected_timing)
    output_video_duration = output_probe.video_duration_seconds
    if output_video_duration is None:
        raise VideoExportError("Timestamp-aware export could not verify the encoded video-track duration.")
    if abs(float(output_video_duration) - expected_video_duration) > _TIMING_TOLERANCE_SECONDS:
        raise VideoExportError(
            "Timestamp-aware export did not preserve the terminal decoded-frame duration: "
            f"expected_video_duration={expected_video_duration:.9f}, "
            f"output_video_duration={output_video_duration:.9f}."
        )
    if output_timing.frame_count != expected_timing.frame_count:
        raise VideoExportError(
            "Timestamp-aware export frame count mismatch after encoding: "
            f"expected={expected_timing.frame_count}, got={output_timing.frame_count}."
        )
    source_av_offset: float | None = None
    output_av_offset: float | None = None
    if source_audio_probe is not None:
        if not output_probe.has_audio:
            raise VideoExportError("Timestamp-aware export completed without the required source audio stream.")
        if not source_audio_probe.audio_codec or not output_probe.audio_codec:
            raise VideoExportError("Timestamp-aware export could not verify the source audio codec after muxing.")
        if output_probe.audio_codec.lower() != source_audio_probe.audio_codec.lower():
            raise VideoExportError(
                "Timestamp-aware export changed the source audio codec despite required stream-copy preservation: "
                f"source={source_audio_probe.audio_codec!r}, output={output_probe.audio_codec!r}."
            )
        source_audio_duration = source_audio_probe.audio_duration_seconds
        output_audio_duration = output_probe.audio_duration_seconds
        if source_audio_duration is None or output_audio_duration is None:
            raise VideoExportError("Timestamp-aware export could not verify the preserved source audio duration.")
        if abs(float(source_audio_duration) - float(output_audio_duration)) > _AUDIO_DURATION_TOLERANCE_SECONDS:
            raise VideoExportError(
                "Timestamp-aware export did not preserve source audio duration: "
                f"source={source_audio_duration:.6f}, output={output_audio_duration:.6f}."
            )
        if source_audio_probe.audio_start_seconds is None or output_probe.audio_start_seconds is None:
            raise VideoExportError("Timestamp-aware export could not verify the source/output audio stream origin.")
        source_av_offset = (
            float(source_audio_probe.audio_start_seconds)
            - float(expected_timing.frames[0].presentation_seconds)
        )
        output_av_offset = (
            float(output_probe.audio_start_seconds)
            - float(output_timing.frames[0].presentation_seconds)
        )
        if abs(source_av_offset - output_av_offset) > _AUDIO_ORIGIN_TOLERANCE_SECONDS:
            raise VideoExportError(
                "Timestamp-aware export did not preserve the source audio/video start offset: "
                f"source_offset={source_av_offset:.9f}, output_offset={output_av_offset:.9f}."
            )

    source_origin = float(expected_timing.frames[0].presentation_seconds)
    output_origin = float(output_timing.frames[0].presentation_seconds)
    for index, (source_frame, output_frame) in enumerate(zip(expected_timing.frames, output_timing.frames)):
        _raise_if_cancelled(should_cancel)
        expected_offset = float(source_frame.presentation_seconds) - source_origin
        observed_offset = float(output_frame.presentation_seconds) - output_origin
        if abs(expected_offset - observed_offset) > _TIMING_TOLERANCE_SECONDS:
            raise VideoExportError(
                "Timestamp-aware export did not preserve decoded presentation timing: "
                f"frame={index}, expected_offset={expected_offset:.9f}, observed_offset={observed_offset:.9f}."
            )
    return output_probe, output_timing, source_av_offset, output_av_offset


def _publish_verified_artifacts(
    *,
    staging_dir: Path,
    final_artifact_dir: Path,
    should_cancel: Callable[[], bool] | None,
) -> None:
    """Publish one already-verified artifact directory with one atomic rename."""

    _raise_if_cancelled(should_cancel)
    if final_artifact_dir.exists():
        raise VideoExportError(f"Timestamp-aware export destination already exists: '{final_artifact_dir}'.")
    try:
        os.replace(staging_dir, final_artifact_dir)
    except OSError as exc:
        raise VideoExportError(
            f"Timestamp-aware export could not atomically publish '{final_artifact_dir}': {exc}"
        ) from exc


def export_timestamped_video(
    frame_paths: Sequence[str | Path],
    *,
    timing: VideoTiming,
    task: str,
    filename_prefix: str,
    source_probe: VideoProbe,
    audio_source_path: str | None,
    extra_metadata: Mapping[str, Any] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> VideoExportResult:
    """Encode a verified VFR MP4, then publish its media and sidecar only after verification."""

    _raise_if_cancelled(should_cancel)
    validate_timestamped_mp4_source(source_probe=source_probe, timing=timing)

    ffmpeg = _which("ffmpeg")
    source_audio_probe: VideoProbe | None = None
    normalized_audio_source = str(audio_source_path).strip() if isinstance(audio_source_path, str) else ""
    if normalized_audio_source:
        source_audio_path = Path(normalized_audio_source).expanduser().resolve()
        if not source_audio_path.is_file():
            raise VideoExportError(f"audio_source_path '{source_audio_path}' does not exist.")
        if Path(source_probe.path).expanduser().resolve() != source_audio_path:
            raise VideoExportError("Timestamp-aware export source probe does not match audio_source_path.")
        source_audio_probe = source_probe
        if not source_audio_probe.has_audio:
            raise VideoExportError("Timestamp-aware export was asked to preserve audio from a source with no audio stream.")
    else:
        source_audio_path = None
        if source_probe.has_audio:
            raise VideoExportError("Timestamp-aware export cannot omit audio from a source that contains audio.")

    prefix = _sanitize_filename_prefix(filename_prefix or task or "video")
    date_dir = datetime.now().strftime("%Y-%m-%d")
    out_dir = video_export_output_root() / f"{task}-videos" / date_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    artifact_name = f"{prefix}_{datetime.now().strftime('%H%M%S')}_{run_id}"
    output_name = f"{artifact_name}.mp4"
    final_artifact_dir = out_dir / artifact_name
    final_video_path = final_artifact_dir / output_name
    final_metadata_path = final_video_path.with_suffix(final_video_path.suffix + ".json")
    staging_dir = out_dir / f".{artifact_name}.{uuid4().hex}.staging"
    staging_video_path = staging_dir / output_name
    staging_metadata_path = staging_dir / final_metadata_path.name
    manifest_path = staging_dir / "frames.ffconcat"

    active_error: BaseException | None = None
    try:
        staging_dir.mkdir(parents=False, exist_ok=False)
        _write_timestamp_manifest(
            manifest_path=manifest_path,
            frame_paths=frame_paths,
            timing=timing,
            should_cancel=should_cancel,
        )
        command: list[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-copyts"]
        if source_audio_probe is not None:
            command += ["-itsoffset", format(float(timing.frames[0].presentation_seconds), ".12g")]
        command += [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
        ]
        if source_audio_path is not None:
            command += ["-i", str(source_audio_path)]
        command += ["-map", "0:v:0"]
        if source_audio_path is not None:
            command += ["-map", "1:a:0?", "-c:a", "copy"]
        command += [
            "-vsync",
            "0",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            str(_TIMESTAMP_MANIFEST_FRAMERATE),
            "-use_editlist",
            "1",
            "-movflags",
            "+faststart",
            str(staging_video_path),
        ]
        _run_cancellable_ffmpeg(
            command,
            should_cancel=should_cancel,
            failure_label="Timestamp-aware SeedVR2 video export",
        )
        _raise_if_cancelled(should_cancel)
        _patch_mp4_terminal_duration(staged_path=staging_video_path, timing=timing)
        _raise_if_cancelled(should_cancel)
        output_probe, output_timing, source_av_offset, output_av_offset = _verify_timestamped_export(
            staged_path=staging_video_path,
            expected_timing=timing,
            source_audio_probe=source_audio_probe,
            should_cancel=should_cancel,
        )
        metadata: dict[str, Any] = {
            "task": task,
            "format": "video/h264-mp4",
            "frames": output_timing.frame_count,
            "timing": {
                "verified": True,
                "source_frame_count": timing.frame_count,
                "tolerance_seconds": _TIMING_TOLERANCE_SECONDS,
                "timebase": _TIMESTAMP_MANIFEST_FRAMERATE,
                "expected_video_duration_seconds": _expected_timestamped_video_duration(timing),
                "video_duration_seconds": output_probe.video_duration_seconds,
                "source_video_origin_seconds": float(timing.frames[0].presentation_seconds),
                "output_video_origin_seconds": float(output_timing.frames[0].presentation_seconds),
            },
            "audio": {
                "source_has_audio": source_audio_probe is not None,
                "preserved": source_audio_probe is not None,
                "codec": output_probe.audio_codec if source_audio_probe is not None else None,
                "duration_seconds": output_probe.audio_duration_seconds if source_audio_probe is not None else None,
                "source_origin_seconds": source_audio_probe.audio_start_seconds if source_audio_probe is not None else None,
                "output_origin_seconds": output_probe.audio_start_seconds if source_audio_probe is not None else None,
                "source_av_offset_seconds": source_av_offset,
                "output_av_offset_seconds": output_av_offset,
            },
        }
        if extra_metadata:
            metadata.update(dict(extra_metadata))
        staging_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            manifest_path.unlink()
        except OSError as exc:
            raise VideoExportError(f"Timestamp-aware export could not remove its staged manifest: {exc}") from exc
        _raise_if_cancelled(should_cancel)
        _publish_verified_artifacts(
            staging_dir=staging_dir,
            final_artifact_dir=final_artifact_dir,
            should_cancel=should_cancel,
        )
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        try:
            shutil.rmtree(staging_dir)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            message = f"Timestamp-aware export staging cleanup failed for '{staging_dir}': {cleanup_error}"
            if active_error is not None:
                raise VideoExportError(message) from active_error
            raise VideoExportError(message) from cleanup_error

    return VideoExportResult(
        saved=True,
        artifact_root=str(final_artifact_dir),
        path=str(final_video_path),
        metadata_path=str(final_metadata_path),
        rel_path=str(final_video_path.relative_to(video_export_output_root())).replace(os.sep, "/"),
        mime="video/mp4",
        width=int(output_probe.width),
        height=int(output_probe.height),
        fps=float(output_probe.fps),
        frame_count=output_timing.frame_count,
        has_audio=bool(output_probe.has_audio),
        audio_verified=source_audio_probe is not None,
        timing_verified=True,
    )


def export_video(
    frames: Sequence[Any],
    *,
    fps: float,
    options: Mapping[str, Any] | None,
    task: str,
    audio_source_path: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> VideoExportResult | None:
    opts = dict(options or {})
    try:
        save_output = parse_bool_value(opts.get("save_output"), field="video_options.save_output", default=False)
    except RuntimeError as exc:
        raise VideoExportError(str(exc)) from exc
    if not save_output:
        return None

    ffmpeg = _which("ffmpeg")

    frames_list = list(frames or [])
    if not frames_list:
        return VideoExportResult(saved=False, reason="no-frames")

    try:
        fps_value = float(fps)
    except (TypeError, ValueError) as exc:
        raise VideoExportError(f"Video export fps must be numeric; got {fps!r}.") from exc
    if not math.isfinite(fps_value):
        raise VideoExportError(f"Video export fps must be finite; got {fps!r}.")
    if fps_value <= 0:
        fps_value = 24.0
    fps_text = format(fps_value, ".12g")
    ext, codec_kind = resolve_video_export_container(str(opts.get("format") or "video/h264-mp4"))
    normalized_audio_source = (
        str(audio_source_path).strip()
        if isinstance(audio_source_path, str) and audio_source_path.strip()
        else None
    )
    if normalized_audio_source and ext not in {"mp4", "webm"}:
        raise VideoExportError(
            f"Audio mux requires mp4 or webm output; got '{ext}'."
        )
    if normalized_audio_source and not os.path.isfile(normalized_audio_source):
        raise VideoExportError(f"audio_source_path '{normalized_audio_source}' does not exist.")

    prefix = _sanitize_filename_prefix(str(opts.get("filename_prefix") or task or "video"))
    date_dir = datetime.now().strftime("%Y-%m-%d")
    root = video_export_output_root()
    out_dir = root / f"{task}-videos" / date_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%H%M%S")
    run_id = uuid4().hex
    out_name = f"{prefix}_{ts}_{run_id}.{ext}"
    out_path = out_dir / out_name

    # Workspace-local temp dir (avoid /tmp surprises).
    work = repo_scratch_path("video_export", f"{task}_{run_id}")
    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Optional ping-pong: append reverse frames (excluding endpoints to avoid duplicates).
    try:
        pingpong_enabled = parse_bool_value(opts.get("pingpong"), field="video_options.pingpong", default=False)
    except RuntimeError as exc:
        raise VideoExportError(str(exc)) from exc
    if pingpong_enabled and len(frames_list) > 2:
        frames_list = list(frames_list) + list(reversed(frames_list[1:-1]))

    # Write frames as PNGs for ffmpeg.
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise VideoExportError(f"Pillow is required for video export: {exc}") from exc

    for idx, frame in enumerate(frames_list, start=1):
        try:
            if isinstance(frame, Image.Image):
                img = frame
            else:
                raise TypeError(f"frame {idx} is not a PIL.Image")
            img.save(frames_dir / f"frame_{idx:06d}.png", format="PNG")
        except Exception as exc:
            raise VideoExportError(f"Failed to write frame {idx}: {exc}") from exc

    pix_fmt = str(opts.get("pix_fmt") or "yuv420p").strip() or "yuv420p"
    crf = int(opts.get("crf", 23) or 23)
    loop_count = int(opts.get("loop_count", 0) or 0)
    try:
        trim_to_audio = parse_bool_value(opts.get("trim_to_audio"), field="video_options.trim_to_audio", default=False)
    except RuntimeError as exc:
        raise VideoExportError(str(exc)) from exc

    # Base encode command.
    cmd: list[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", fps_text, "-i", str(frames_dir / "frame_%06d.png")]

    include_audio = bool(normalized_audio_source)
    if include_audio:
        cmd += ["-i", normalized_audio_source]

    if ext == "gif":
        # High-quality GIF using palettegen/paletteuse.
        palette = work / "palette.png"
        cmd_palette = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            fps_text,
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-vf",
            "palettegen",
            str(palette),
        ]
        try:
            subprocess.check_output(cmd_palette, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            msg = exc.output.decode("utf-8", errors="replace") if exc.output else str(exc)
            raise VideoExportError(f"ffmpeg palettegen failed: {msg}") from exc

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            fps_text,
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-i",
            str(palette),
            "-lavfi",
            "paletteuse",
            "-loop",
            str(loop_count),
            str(out_path),
        ]
    else:
        if codec_kind == "h265":
            cmd += ["-c:v", "libx265", "-crf", str(crf), "-pix_fmt", pix_fmt]
        elif codec_kind == "vp9":
            cmd += ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf), "-pix_fmt", pix_fmt]
        else:
            cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", pix_fmt]

        if include_audio:
            ac = _audio_codec_for(ext)
            if ac:
                cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", ac]
            if trim_to_audio:
                cmd += ["-shortest"]

        # Faststart for mp4 helps browser playback.
        if ext == "mp4":
            cmd += ["-movflags", "+faststart"]
        cmd += [str(out_path)]

    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        msg = exc.output.decode("utf-8", errors="replace") if exc.output else str(exc)
        raise VideoExportError(f"ffmpeg export failed: {msg}") from exc
    finally:
        # Best-effort cleanup of intermediate frames.
        try:
            shutil.rmtree(work)
        except Exception:
            pass

    rel = os.path.relpath(out_path, root)
    mime = "video/mp4" if ext == "mp4" else ("video/webm" if ext == "webm" else "image/gif")

    try:
        save_metadata = parse_bool_value(opts.get("save_metadata"), field="video_options.save_metadata", default=False)
    except RuntimeError as exc:
        raise VideoExportError(str(exc)) from exc
    if save_metadata:
        meta_path = out_path.with_suffix(out_path.suffix + ".json")
        meta: dict[str, Any] = {
            "task": task,
            "fps": fps_value,
            "frames": len(frames_list),
            "format": str(opts.get("format") or ""),
            "pix_fmt": pix_fmt,
            "crf": crf,
            "loop_count": loop_count,
            "pingpong": pingpong_enabled,
            "trim_to_audio": trim_to_audio,
        }
        if extra_metadata:
            meta.update(dict(extra_metadata))
        try:
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return VideoExportResult(
        saved=True,
        path=str(out_path),
        rel_path=str(rel).replace(os.sep, "/"),
        mime=mime,
        fps=fps_value,
        frame_count=len(frames_list),
        has_audio=include_audio,
    )

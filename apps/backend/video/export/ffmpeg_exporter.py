"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Encode frame sequences to a video container via ffmpeg (mp4/webm/gif).
Writes frames to a workspace-local temp dir, runs ffmpeg (optional audio mux), and returns a structured export result suitable for
serving under `/api/output/{rel_path}` with deterministic ffmpeg binary resolution.

Symbols (top-level; keep in sync; no ghosts):
- `VideoExportError` (class): Explicit export error surfaced when ffmpeg/Pillow or encoding fails.
- `_which` (function): Resolves ffmpeg executable paths via shared resolver precedence (env override → deterministic runtime path → downloader/PATH).
- `_output_root` (function): Resolves the repo-local output root (`CODEX_ROOT/output`).
- `_sanitize_filename_prefix` (function): Sanitizes a user/task-provided filename prefix for safe output paths.
- `resolve_video_export_container` (function): Maps a format token to an output container + codec kind.
- `_audio_codec_for` (function): Chooses an audio codec for a given output container.
- `VideoExportResult` (dataclass): Export result container (saved flag + path/rel_path/mime + metadata).
- `export_video` (function): Main entrypoint; writes frames and runs ffmpeg to produce the final video file.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from apps.backend.core.strict_values import parse_bool_value
from apps.backend.infra.config.repo_root import get_repo_root, repo_scratch_path
from apps.backend.video.runtime_dependencies import VideoDependencyResolutionError, resolve_ffmpeg_binary


class VideoExportError(RuntimeError):
    pass


def _which(name: str) -> str:
    try:
        return resolve_ffmpeg_binary(name)
    except VideoDependencyResolutionError as exc:
        raise VideoExportError(str(exc)) from exc


def _output_root() -> Path:
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
    path: str | None = None
    rel_path: str | None = None
    mime: str | None = None
    reason: str | None = None
    fps: float | None = None
    frame_count: int | None = None
    has_audio: bool = False


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
    root = _output_root()
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

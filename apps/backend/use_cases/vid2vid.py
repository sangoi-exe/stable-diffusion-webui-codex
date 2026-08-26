"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend vid2vid use-case orchestration (native-family dispatch + WAN pipeline + optional flow guidance/interpolation/export).
Takes an input video (or frames), dispatches the active vid2vid family lane (currently native-family scaffolds plus WAN methods), optionally
applies optical-flow-based warping/guidance plus interpolation, and returns task events/results.

Symbols (top-level; keep in sync; no ghosts):
- `_build_pipeline_telemetry_scope` (function): Creates a mutable task-scoped telemetry context owner for vid2vid run/stage events.
- `_emit_pipeline_event` (function): Emits canonical structured pipeline telemetry events (`pipeline.*`) for vid2vid.
- `_blend` (function): Alpha-blends two PIL images (resizes `b` to `a` when needed).
- `_load_pil_images` (function): Loads a list of image paths into PIL images (copies + closes file handles).
- `_extract_vid2vid_options` (function): Extracts `vid2vid` dict options from request extras.
- `_extract_flow_options` (function): Extracts `vid2vid_flow` dict options from request extras.
- `_sanitize_img2vid_extras_for_flow_chunks` (function): Forces inner-img2vid mode to `solo` and strips window/chunk temporal controls from vid2vid flow-chunk extras to prevent nested temporal recursion.
- `_as_video_options` (function): Validates optional request `video_options` payload shape (mapping or `None`).
- `_as_wan_animate_mode` (function): Normalizes/validates WAN animate mode string (`animate` vs `replace`).
- `_validate_4n_plus_1` (function): Validates an integer is of the form `4N+1` (common WAN constraints).
- `_build_result_payload` (function): Builds the final ResultEvent payload (video export descriptor + optional preview frames) and attaches warnings.
- `_run_netflix_void_vid2vid` (function): Runs the native Netflix VOID vid2vid branch through the family-owned runtime seam.
- `_run_native_pipeline` (function): Runs the native WAN diffusers pipeline path for vid2vid (requires `comp.pipeline`).
- `_run_wan_animate` (function): Runs the WAN “animate” path (stage planning + prompt/text guidance; includes nested option handling).
- `_run_flow_chunks` (function): Applies flow-guided warping in chunks using RAFT and per-frame options (nested loop over frames).
- `run_vid2vid` (function): Main use-case entrypoint; parses input, dispatches method, emits progress/result events, and exports video.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import threading
import time
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4
from typing import Any, Dict, Iterator, Optional, Sequence

from apps.backend.core.engine_interface import TaskType
from apps.backend.core.requests import Img2VidRequest, InferenceEvent, ProgressEvent, ResultEvent, Vid2VidRequest
from apps.backend.core.strict_values import parse_bool_value
from apps.backend.engines.wan22.wan22_common import WanStageOptions
from apps.backend.infra.config.repo_root import repo_scratch_path
from apps.backend.runtime.logging import emit_backend_event
from apps.backend.runtime.pipeline_stages.hires_fix import resolve_pipeline_telemetry_context
from apps.backend.runtime.pipeline_stages.video import (
    apply_wan_stage_loras,
    apply_video_interpolation,
    assemble_video_metadata,
    build_video_plan,
    build_video_request_effective_snapshot,
    prepare_base_snapshot_video_options,
    read_video_interpolation_options,
    resolve_video_output_fps,
)
from apps.backend.video.export.ffmpeg_exporter import export_video
from apps.backend.video.flow.torchvision_raft import FlowGuidanceError, RaftFlowEstimator, warp_frame
from apps.backend.video.io.ffmpeg import extract_frames, probe_video


def _build_pipeline_telemetry_scope(*, mode: str) -> SimpleNamespace:
    scope = SimpleNamespace()
    setattr(scope, "_codex_pipeline_mode", str(mode))
    task_context = str(threading.current_thread().name or "").strip() or "unknown-thread"
    marker = "-task-"
    if marker in task_context:
        candidate = task_context.split(marker, 1)[1].strip()
        if candidate:
            setattr(scope, "_codex_task_id", candidate)
            setattr(scope, "_codex_correlation_id", candidate)
            setattr(scope, "_codex_hires_correlation_id", candidate)
            setattr(scope, "_codex_correlation_source", "task_id")
    resolve_pipeline_telemetry_context(
        scope,
        default_mode=str(mode),
        require_mode=True,
    )
    return scope


def _emit_pipeline_event(
    scope: Any,
    event: str,
    *,
    stage: str,
    **fields: object,
) -> None:
    telemetry = resolve_pipeline_telemetry_context(
        scope,
        default_mode="vid2vid",
        require_mode=True,
    )
    emit_backend_event(
        event,
        logger="backend.use_cases.vid2vid",
        mode=telemetry.mode,
        stage=stage,
        correlation_id=telemetry.correlation_id,
        correlation_source=telemetry.correlation_source,
        task_id=telemetry.task_id,
        **fields,
    )


def _blend(a: Any, b: Any, *, alpha: float) -> Any:
    from PIL import Image  # type: ignore

    if not isinstance(a, Image.Image) or not isinstance(b, Image.Image):
        raise RuntimeError("blend expects PIL images")
    aa = a.convert("RGB")
    bb = b.convert("RGB")
    if aa.size != bb.size:
        bb = bb.resize(aa.size)
    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    return Image.blend(aa, bb, alpha_clamped)


def _load_pil_images(paths: Sequence[str]) -> list[Any]:
    from PIL import Image  # type: ignore

    out: list[Any] = []
    for p in paths:
        img = Image.open(p)
        out.append(img.copy())
        img.close()
    return out


def _extract_vid2vid_options(extras: Dict[str, Any]) -> dict[str, Any]:
    cfg = extras.get("vid2vid") if isinstance(extras.get("vid2vid"), dict) else {}
    return dict(cfg) if isinstance(cfg, dict) else {}


def _extract_flow_options(extras: Dict[str, Any]) -> dict[str, Any]:
    cfg = extras.get("vid2vid_flow") if isinstance(extras.get("vid2vid_flow"), dict) else {}
    return dict(cfg) if isinstance(cfg, dict) else {}


def _sanitize_img2vid_extras_for_flow_chunks(extras: Dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(extras)
    sanitized["img2vid_mode"] = "solo"
    for key in (
        "img2vid_chunk_frames",
        "img2vid_overlap_frames",
        "img2vid_anchor_alpha",
        "img2vid_chunk_seed_mode",
        "img2vid_chunk_buffer_mode",
        "img2vid_window_frames",
        "img2vid_window_stride",
        "img2vid_window_commit_frames",
    ):
        sanitized.pop(key, None)
    return sanitized


def _as_video_options(raw: object) -> Mapping[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return raw
    raise RuntimeError(
        "vid2vid request field 'video_options' must be an object when provided "
        f"(got {type(raw).__name__})."
    )


def _as_wan_animate_mode(raw: str) -> str:
    v = str(raw or "").strip().lower()
    if v in {"animate", "animation"}:
        return "animate"
    if v in {"replace", "replacement"}:
        return "replace"
    raise RuntimeError("wan_animate mode must be 'animate' or 'replace'")


def _validate_4n_plus_1(value: int, *, name: str) -> int:
    v = int(value)
    if v <= 0 or (v - 1) % 4 != 0:
        raise RuntimeError(f"{name} must be 4N+1 (got {v})")
    return v


def _build_result_payload(
    *,
    engine: Any,
    info: dict[str, Any],
    preview_frames: Sequence[Any],
    request: Vid2VidRequest,
    video_meta: Any,
    video_export_error: str | None,
) -> dict[str, Any]:
    extras_raw = getattr(request, "extras", {}) or {}
    extras: dict[str, Any] = dict(extras_raw) if isinstance(extras_raw, Mapping) else {}
    user_return_frames = parse_bool_value(
        extras.get("video_return_frames"),
        field="extras.video_return_frames",
        default=False,
    )

    video_options = _as_video_options(getattr(request, "video_options", None))
    save_output = parse_bool_value(
        video_options.get("save_output") if video_options is not None else None,
        field="video_options.save_output",
        default=False,
    )

    video_saved = parse_bool_value(
        (video_meta.get("saved") if isinstance(video_meta, Mapping) else getattr(video_meta, "saved", None)),
        field="video_meta.saved",
        default=False,
    )
    export_failed = save_output and not video_saved

    effective_return_frames = user_return_frames or (not save_output) or export_failed

    warnings: list[str] = []
    if not save_output:
        warnings.append(
            "Save output is OFF: no video file was written. "
            "Frames are returned so you can download them from the Results viewer."
        )
    if export_failed:
        reason = str(video_export_error or getattr(video_meta, "reason", "") or "").strip() or "unknown error"
        warnings.append(
            f"Video export failed ({reason}). "
            "Frames are returned as a fallback."
        )

    if warnings:
        info["warnings"] = warnings

    if save_output:
        video_export: dict[str, Any] = {"saved": video_saved}
        if video_saved:
            video_export.update(
                {
                    "rel_path": getattr(video_meta, "rel_path", None),
                    "mime": getattr(video_meta, "mime", None),
                    "fps": getattr(video_meta, "fps", None),
                    "frames": getattr(video_meta, "frame_count", None),
                }
            )
        else:
            video_export["error"] = str(video_export_error or getattr(video_meta, "reason", "") or "").strip() or None
        info["video_export"] = video_export

    payload: dict[str, Any] = {"info": engine._to_json(info)}  # type: ignore[attr-defined]
    if effective_return_frames:
        payload["images"] = list(preview_frames)
    if video_saved:
        payload["video"] = {
            "rel_path": getattr(video_meta, "rel_path", None),
            "mime": getattr(video_meta, "mime", None),
        }
    return payload


def _run_netflix_void_vid2vid(
    *,
    comp: Any,
    request: Vid2VidRequest,
) -> Iterator[InferenceEvent]:
    run_vid2vid = getattr(comp, "run_vid2vid", None)
    if not callable(run_vid2vid):
        raise RuntimeError("Netflix VOID vid2vid runtime is missing callable run_vid2vid().")
    yield from run_vid2vid(request=request)


def _run_native_pipeline(
    *,
    engine: Any,
    comp: Any,
    request: Vid2VidRequest,
    frames_in: Sequence[Any],
) -> list[Any]:
    pipe = getattr(comp, "pipeline", None)
    if pipe is None:
        raise RuntimeError("vid2vid method 'native' requires a Diffusers WAN pipeline (comp.pipeline)")

    # Optional: stage LoRA (WAN extras) for lightx2v-style adapters.
    extras = getattr(request, "extras", {}) or {}
    wan_high_cfg = extras.get("wan_high") if isinstance(extras, dict) else None
    wan_hi_opts = WanStageOptions.from_mapping(wan_high_cfg) if isinstance(wan_high_cfg, dict) else None
    if wan_hi_opts and wan_hi_opts.loras:
        logger = getattr(engine, "_logger", None)
        apply_wan_stage_loras(
            pipe=pipe,
            stage_loras=wan_hi_opts.loras,
            logger_=logger,
            stage_label="high",
        )

    strength = request.strength
    if strength is None:
        strength = 0.8
    strength_val = max(0.0, min(1.0, float(strength)))

    import torch

    with torch.inference_mode():
        output = pipe(
            video=list(frames_in),
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", None),
            num_frames=int(getattr(request, "num_frames", len(frames_in)) or len(frames_in)),
            num_inference_steps=int(getattr(request, "steps", 30) or 30),
            height=int(getattr(request, "height", 432) or 432),
            width=int(getattr(request, "width", 768) or 768),
            guidance_scale=getattr(request, "guidance_scale", None),
            strength=strength_val,
        )
    if hasattr(output, "frames"):
        frames = list(output.frames[0])
        if not frames:
            raise RuntimeError("vid2vid pipeline returned 0 frames")
        return frames
    raise RuntimeError("vid2vid pipeline returned no frames")


def _run_wan_animate(
    *,
    engine: Any,
    comp: Any,
    request: Vid2VidRequest,
    cfg: dict[str, Any],
) -> tuple[list[Any], int, bool, str | None]:
    logger = getattr(engine, "_logger", None)
    pipe = getattr(comp, "pipeline", None)
    if pipe is None:
        raise RuntimeError("vid2vid method 'wan_animate' requires a Diffusers WanAnimatePipeline (comp.pipeline)")

    ref = getattr(request, "reference_image", None)
    if ref is None:
        raise RuntimeError("vid2vid wan_animate requires 'reference_image'")

    pose_path = str(getattr(request, "pose_video_path", "") or "").strip()
    face_path = str(getattr(request, "face_video_path", "") or "").strip()
    if not pose_path or not face_path:
        raise RuntimeError("vid2vid wan_animate requires pose_video_path and face_video_path")

    mode = _as_wan_animate_mode(getattr(request, "animate_mode", "animate"))
    bg_path = str(getattr(request, "background_video_path", "") or "").strip()
    mask_path = str(getattr(request, "mask_video_path", "") or "").strip()
    if mode == "replace" and (not bg_path or not mask_path):
        raise RuntimeError("vid2vid wan_animate mode 'replace' requires background_video_path and mask_video_path")

    yield_fps = parse_bool_value(
        cfg.get("use_source_fps"),
        field="vid2vid.use_source_fps",
        default=False,
    )
    fps_val = int(getattr(request, "fps", 24) or 24)
    pose_probe = probe_video(pose_path)
    if yield_fps:
        fps_val = max(1, int(round(float(pose_probe.fps))))

    max_frames = cfg.get("max_frames")
    try:
        max_frames = int(max_frames) if max_frames is not None else None
    except Exception:
        max_frames = None

    tag = uuid4().hex
    work = repo_scratch_path("vid2vid", f"task_{tag}")
    pose_dir = work / "pose_frames"
    face_dir = work / "face_frames"
    bg_dir = work / "bg_frames"
    mask_dir = work / "mask_frames"

    try:
        if logger:
            logger.info("[vid2vid.wan_animate] extracting frames (fps=%s max=%s)", fps_val, max_frames)
        pose_paths = extract_frames(
            pose_path,
            out_dir=str(pose_dir),
            fps=float(fps_val),
            max_frames=max_frames,
            width=int(getattr(request, "width", 1280) or 1280),
            height=int(getattr(request, "height", 720) or 720),
        )
        face_paths = extract_frames(
            face_path,
            out_dir=str(face_dir),
            fps=float(fps_val),
            max_frames=max_frames,
            width=int(getattr(request, "width", 1280) or 1280),
            height=int(getattr(request, "height", 720) or 720),
        )
        if len(pose_paths) != len(face_paths):
            raise RuntimeError(f"pose/face produced different frame counts (pose={len(pose_paths)} face={len(face_paths)})")

        bg_paths: list[str] | None = None
        mask_paths: list[str] | None = None
        if mode == "replace":
            bg_paths = extract_frames(
                bg_path,
                out_dir=str(bg_dir),
                fps=float(fps_val),
                max_frames=max_frames,
                width=int(getattr(request, "width", 1280) or 1280),
                height=int(getattr(request, "height", 720) or 720),
            )
            mask_paths = extract_frames(
                mask_path,
                out_dir=str(mask_dir),
                fps=float(fps_val),
                max_frames=max_frames,
                width=int(getattr(request, "width", 1280) or 1280),
                height=int(getattr(request, "height", 720) or 720),
            )
            if len(bg_paths) != len(pose_paths) or len(mask_paths) != len(pose_paths):
                raise RuntimeError(
                    "bg/mask frame count must match pose/face frame count "
                    f"(pose={len(pose_paths)} bg={len(bg_paths)} mask={len(mask_paths)})"
                )

        pose_frames = _load_pil_images(pose_paths)
        face_frames = _load_pil_images(face_paths)
        bg_frames = _load_pil_images(bg_paths) if bg_paths else None
        mask_frames = _load_pil_images(mask_paths) if mask_paths else None

        segment_len = _validate_4n_plus_1(int(getattr(request, "segment_frame_length", 77) or 77), name="segment_frame_length")
        prev_cond = _validate_4n_plus_1(
            int(getattr(request, "prev_segment_conditioning_frames", 1) or 1),
            name="prev_segment_conditioning_frames",
        )

        seed = getattr(request, "seed", None)
        generator = None
        if isinstance(seed, int) and seed >= 0:
            try:
                import torch  # type: ignore

                device_raw = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", None) or "cpu"
                device = str(device_raw) if not hasattr(device_raw, "type") else str(device_raw.type)
                generator = torch.Generator(device=device).manual_seed(int(seed))
            except Exception:
                generator = None

        width = int(getattr(request, "width", 1280) or 1280)
        height = int(getattr(request, "height", 720) or 720)
        steps = int(getattr(request, "steps", 20) or 20)
        guidance = getattr(request, "guidance_scale", None)
        if guidance is None:
            guidance = 1.0

        import torch

        with torch.inference_mode():
            output = pipe(
                image=ref,
                pose_video=pose_frames,
                face_video=face_frames,
                background_video=bg_frames,
                mask_video=mask_frames,
                prompt=request.prompt,
                negative_prompt=getattr(request, "negative_prompt", None) or None,
                height=height,
                width=width,
                segment_frame_length=segment_len,
                num_inference_steps=steps,
                mode=mode,
                prev_segment_conditioning_frames=prev_cond,
                motion_encode_batch_size=(getattr(request, "motion_encode_batch_size", None) or None),
                guidance_scale=float(guidance),
                generator=generator,
                output_type="pil",
            )

        frames_out: list[Any] = []
        if hasattr(output, "frames"):
            frames_out = list(output.frames[0])
        elif hasattr(output, "images"):
            frames_out = list(output.images)  # type: ignore[attr-defined]
        if not frames_out:
            raise RuntimeError("WanAnimate pipeline returned 0 frames")

        # Optional: copy audio from the driving/original video if provided and has audio.
        audio_source = str(getattr(request, "video_path", "") or "").strip() or None
        has_audio = False
        if audio_source:
            try:
                audio_probe = probe_video(audio_source)
            except Exception:
                has_audio = False
            else:
                has_audio = parse_bool_value(
                    getattr(audio_probe, "has_audio", None),
                    field="audio_probe.has_audio",
                    default=False,
                )
        return frames_out, fps_val, has_audio, audio_source
    finally:
        try:
            shutil.rmtree(work)
        except Exception:
            pass


def _run_flow_chunks(
    *,
    engine: Any,
    request: Vid2VidRequest,
    frames_in: Sequence[Any],
) -> list[Any]:
    extras = dict(getattr(request, "extras", {}) or {})
    inner_img2vid_extras = _sanitize_img2vid_extras_for_flow_chunks(extras)
    residual_temporal_keys = sorted(
        key
        for key in (
            "img2vid_chunk_frames",
            "img2vid_overlap_frames",
            "img2vid_anchor_alpha",
            "img2vid_chunk_seed_mode",
            "img2vid_chunk_buffer_mode",
            "img2vid_window_frames",
            "img2vid_window_stride",
            "img2vid_window_commit_frames",
        )
        if key in inner_img2vid_extras
    )
    if residual_temporal_keys:
        raise RuntimeError(
            "vid2vid flow_chunks: inner img2vid extras still contain temporal chunk/sliding controls "
            f"after sanitization ({', '.join(residual_temporal_keys)})."
        )

    cfg = _extract_vid2vid_options(extras)
    flow_cfg = _extract_flow_options(extras)

    method = str(cfg.get("method") or "flow_chunks").strip().lower()
    if method not in {"flow_chunks", "chunks"}:
        raise RuntimeError(f"Unsupported vid2vid chunk method: {method}")

    strength = request.strength
    if strength is None:
        strength = float(cfg.get("strength", 0.8) or 0.8)
    strength_val = max(0.0, min(1.0, float(strength)))
    anchor_alpha = max(0.0, min(1.0, float(cfg.get("anchor_alpha", 1.0 - strength_val))))

    chunk_frames = int(cfg.get("chunk_frames") or getattr(request, "num_frames", 16) or 16)
    chunk_frames = max(2, min(128, int(chunk_frames)))
    overlap = int(cfg.get("overlap_frames") or max(2, chunk_frames // 4))
    overlap = max(0, min(chunk_frames - 1, overlap))
    stride = max(1, chunk_frames - overlap)

    flow_enabled = parse_bool_value(
        flow_cfg.get("enabled"),
        field="vid2vid_flow.enabled",
        default=True,
    )
    estimator: Optional[RaftFlowEstimator] = None
    if flow_enabled:
        estimator_kwargs: dict[str, Any] = {
            "use_large": parse_bool_value(
                flow_cfg.get("use_large"),
                field="vid2vid_flow.use_large",
                default=False,
            ),
            "downscale": int(flow_cfg.get("downscale") or 2),
        }
        flow_device_raw = flow_cfg.get("device")
        if flow_device_raw not in (None, ""):
            estimator_kwargs["device"] = str(flow_device_raw)
        estimator = RaftFlowEstimator(**estimator_kwargs)

    out: list[Any] = []
    seed_base = getattr(request, "seed", None)
    seed_is_valid = isinstance(seed_base, int) and seed_base >= 0

    for start in range(0, len(frames_in), stride):
        needed = min(chunk_frames, len(frames_in) - start)
        if needed <= 0:
            break

        if start == 0 or not out:
            init = frames_in[0]
        else:
            prev_src = frames_in[start - 1]
            cur_src = frames_in[start]
            prev_out = out[start - 1] if (start - 1) < len(out) else out[-1]
            warped = prev_out
            if estimator is not None:
                try:
                    flow = estimator.estimate_backward_flow(target_frame=cur_src, source_frame=prev_src)
                    warped = warp_frame(prev_out, backward_flow=flow, device=estimator.device)
                except FlowGuidanceError as exc:
                    raise RuntimeError(f"Optical flow guidance failed: {exc}") from exc
            init = _blend(warped, cur_src, alpha=anchor_alpha)

        # Vary seed per chunk for seam robustness while keeping determinism.
        chunk_seed = None
        if seed_is_valid:
            chunk_seed = int(seed_base) + int(start)

        chunk_req = Img2VidRequest(
            task=TaskType.IMG2VID,
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", ""),
            init_image=init,
            width=int(getattr(request, "width", 768) or 768),
            height=int(getattr(request, "height", 432) or 432),
            steps=int(getattr(request, "steps", 30) or 30),
            num_frames=int(needed),
            fps=int(getattr(request, "fps", 24) or 24),
            seed=chunk_seed if chunk_seed is not None else getattr(request, "seed", None),
            guidance_scale=getattr(request, "guidance_scale", None),
            sampler=getattr(request, "sampler", None),
            scheduler=getattr(request, "scheduler", None),
            extras=inner_img2vid_extras,
        )

        # Engines may stream progress; we only need the final frames for stitching.
        frames_chunk: list[Any] = []
        for ev in engine.img2vid(chunk_req):
            if isinstance(ev, ResultEvent):
                payload = ev.payload or {}
                frames_chunk = list(payload.get("images", []) or [])
        if not frames_chunk:
            raise RuntimeError(f"vid2vid chunk produced 0 frames at start={start}")

        # Stitch with overlap crossfade when we already have frames at this index.
        overlap_count = min(overlap, len(frames_chunk), max(0, len(out) - start))
        for i in range(overlap_count):
            alpha = float(i + 1) / float(overlap_count)
            out[start + i] = _blend(out[start + i], frames_chunk[i], alpha=alpha)

        # Append new frames beyond current output length.
        for j in range(overlap_count, min(len(frames_chunk), len(frames_in) - start)):
            idx = start + j
            if idx < len(out):
                out[idx] = frames_chunk[j]
            else:
                out.append(frames_chunk[j])

    return out[: len(frames_in)]


def run_vid2vid(
    *,
    engine: Any,
    comp: Any,
    request: Vid2VidRequest,
) -> Iterator[InferenceEvent]:
    logger = getattr(engine, "_logger", None)
    telemetry_scope = _build_pipeline_telemetry_scope(mode="vid2vid")
    engine_id = str(getattr(engine, "engine_id", "") or "").strip().lower()
    if engine_id == "netflix_void":
        yield from _run_netflix_void_vid2vid(comp=comp, request=request)
        return
    extras = dict(getattr(request, "extras", {}) or {})
    cfg = _extract_vid2vid_options(extras)
    method = str(cfg.get("method") or "flow_chunks").strip().lower()
    if method not in {"native", "flow_chunks", "chunks", "wan_animate"}:
        raise RuntimeError(f"Unsupported vid2vid method: {method}")

    try:
        plan = build_video_plan(request)
        frames_in_count: int | None = None
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.run.start",
            stage="run.start",
            backend="diffusers",
            method=str(method),
            engine_id=str(getattr(engine, "engine_id", "") or "unknown"),
            requested_frames=int(plan.frames),
            requested_width=int(plan.width),
            requested_height=int(plan.height),
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="prepare.complete",
            stage_name="prepare",
            backend="diffusers",
            method=str(method),
            frames=int(plan.frames),
            width=int(plan.width),
            height=int(plan.height),
            steps=int(plan.steps),
        )

        if method == "wan_animate":
            yield ProgressEvent(stage="probe", percent=0.0, message="Probing WAN Animate inputs")
            start = time.perf_counter()
            frames_out, fps_val, has_audio, audio_source = _run_wan_animate(engine=engine, comp=comp, request=request, cfg=cfg)
            probe = None
            _emit_pipeline_event(
                telemetry_scope,
                "pipeline.stage.complete",
                stage="generation.complete",
                stage_name="generation",
                backend="diffusers",
                method=str(method),
                frame_count=int(len(frames_out)),
                has_audio=bool(has_audio),
            )
        else:
            video_path = str(getattr(request, "video_path", "") or "").strip()
            if not video_path:
                raise RuntimeError("vid2vid requires 'video_path'")

            src = Path(video_path)
            if not src.is_file():
                raise RuntimeError(f"vid2vid video not found: {video_path}")

            yield ProgressEvent(stage="probe", percent=0.0, message="Probing input video")
            probe = probe_video(video_path)
            _emit_pipeline_event(
                telemetry_scope,
                "pipeline.stage.complete",
                stage="probe.complete",
                stage_name="probe",
                backend="diffusers",
                method=str(method),
                source_has_audio=bool(parse_bool_value(getattr(probe, "has_audio", None), field="probe.has_audio", default=False)),
                source_frame_count=int(getattr(probe, "frame_count", 0) or 0),
            )

            use_source_fps = parse_bool_value(
                cfg.get("use_source_fps"),
                field="vid2vid.use_source_fps",
                default=False,
            )
            use_source_frames = parse_bool_value(
                cfg.get("use_source_frames"),
                field="vid2vid.use_source_frames",
                default=True,
            )

            fps_val = int(getattr(request, "fps", 24) or 24)
            if use_source_fps:
                fps_val = max(1, int(round(float(probe.fps))))
            plan.fps = int(fps_val)

            frames_target = int(getattr(request, "num_frames", 0) or 0)
            if use_source_frames and probe.frame_count:
                frames_target = int(probe.frame_count)
            if frames_target <= 0:
                frames_target = 16

            start_s = cfg.get("start_seconds")
            end_s = cfg.get("end_seconds")
            max_frames = cfg.get("max_frames")
            try:
                start_s = float(start_s) if start_s is not None else None
            except Exception:
                start_s = None
            try:
                end_s = float(end_s) if end_s is not None else None
            except Exception:
                end_s = None
            try:
                max_frames = int(max_frames) if max_frames is not None else None
            except Exception:
                max_frames = None

            # Honor explicit max_frames first, otherwise cap by target.
            cap = frames_target
            if max_frames is not None and max_frames > 0:
                cap = min(cap, max_frames)

            work = repo_scratch_path("vid2vid", f"task_{uuid4().hex}")
            frames_dir = work / "src_frames"
            yield ProgressEvent(stage="decode", percent=0.05, message="Decoding video frames")
            paths = extract_frames(
                video_path,
                out_dir=str(frames_dir),
                start_seconds=start_s,
                end_seconds=end_s,
                fps=float(fps_val),
                max_frames=int(cap) if cap > 0 else None,
                width=int(getattr(request, "width", 768) or 768),
                height=int(getattr(request, "height", 432) or 432),
            )
            frames_in = _load_pil_images(paths)
            try:
                shutil.rmtree(work)
            except Exception:
                pass
            _emit_pipeline_event(
                telemetry_scope,
                "pipeline.stage.complete",
                stage="decode.complete",
                stage_name="decode",
                backend="diffusers",
                method=str(method),
                decoded_frame_count=int(len(frames_in)),
                target_fps=int(fps_val),
            )

            yield ProgressEvent(stage="run", percent=0.1, message=f"Running vid2vid ({method})")
            start = time.perf_counter()

            if method == "native":
                frames_out = _run_native_pipeline(engine=engine, comp=comp, request=request, frames_in=frames_in)
            else:
                frames_out = _run_flow_chunks(engine=engine, request=request, frames_in=frames_in)
            frames_in_count = len(frames_in)
            has_audio = parse_bool_value(
                getattr(probe, "has_audio", None),
                field="probe.has_audio",
                default=False,
            )
            audio_source = video_path if has_audio else None
            _emit_pipeline_event(
                telemetry_scope,
                "pipeline.stage.complete",
                stage="generation.complete",
                stage_name="generation",
                backend="diffusers",
                method=str(method),
                frame_count=int(len(frames_out)),
                frames_in=int(frames_in_count),
                has_audio=bool(has_audio),
            )

        video_options = _as_video_options(getattr(request, "video_options", None))
        vfi_options = read_video_interpolation_options(extras)
        base_snapshot_options = prepare_base_snapshot_video_options(
            video_options,
            task="vid2vid",
            interpolation_options=vfi_options,
        )
        base_video_meta: Any = None
        if base_snapshot_options is not None:
            base_video_meta = export_video(
                frames_out,
                fps=fps_val,
                options=base_snapshot_options,
                task="vid2vid",
                audio_source_path=audio_source if has_audio else None,
                extra_metadata={"snapshot_stage": "base_before_postprocess", "method": method},
            )
            if not parse_bool_value(
                getattr(base_video_meta, "saved", None),
                field="base_video_meta.saved",
                default=False,
            ):
                reason = str(getattr(base_video_meta, "reason", "") or "").strip()
                raise RuntimeError(
                    "vid2vid: base snapshot export failed with save_output=true"
                    + (f" ({reason})" if reason else "")
                )
            base_rel_path = str(getattr(base_video_meta, "rel_path", "") or "").strip()
            if base_rel_path:
                logger.info(
                    "vid2vid: base snapshot exported before post-process: %s",
                    base_rel_path,
                )
            _emit_pipeline_event(
                telemetry_scope,
                "pipeline.stage.complete",
                stage="base_snapshot_export.complete",
                stage_name="base_snapshot_export",
                backend="diffusers",
                method=str(method),
                video_saved=bool(parse_bool_value(getattr(base_video_meta, "saved", None), field="base_video_meta.saved", default=False)),
            )

        if vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1:
            yield ProgressEvent(stage="interpolate", percent=0.95, message="Interpolating frames (VFI)")
        frames_out, vfi_opts = apply_video_interpolation(frames_out, options=vfi_options, logger_=logger)
        fps_out = resolve_video_output_fps(fps_val, vfi_opts)
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="interpolation.complete",
            stage_name="interpolation",
            backend="diffusers",
            method=str(method),
            interpolation_enabled=bool(vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1),
            output_fps=int(fps_out),
            frame_count=int(len(frames_out)),
        )

        elapsed = time.perf_counter() - start
        plan.fps = int(fps_out)

        @dataclass(frozen=True)
        class _SamplerOutcome:
            sampler_in: str | None
            scheduler_in: str | None
            sampler_effective: str | None
            scheduler_effective: str | None
            warnings: tuple[str, ...] = ()

        sampler_outcome = _SamplerOutcome(
            sampler_in=getattr(request, "sampler", None),
            scheduler_in=getattr(request, "scheduler", None),
            sampler_effective=getattr(request, "sampler", None),
            scheduler_effective=getattr(request, "scheduler", None),
        )

        info = assemble_video_metadata(
            engine,
            plan,
            sampler_outcome,
            elapsed=elapsed,
            frame_count=len(frames_out),
            task="vid2vid",
        )
        info.update(
            {
                "method": method,
                "frames_in": (frames_in_count if method != "wan_animate" else None),
                "frames_out": len(frames_out),
                "strength": request.strength,
                "audio_in": has_audio,
            }
        )
        if method == "wan_animate":
            info["animate_mode"] = getattr(request, "animate_mode", None)
            info["segment_frame_length"] = int(getattr(request, "segment_frame_length", 77) or 77)
            info["prev_segment_conditioning_frames"] = int(getattr(request, "prev_segment_conditioning_frames", 1) or 1)
        if vfi_opts is not None:
            info["video_interpolation"] = vfi_opts
        if base_video_meta is not None:
            info["video_base_snapshot"] = {
                "saved": bool(getattr(base_video_meta, "saved", False)),
                "rel_path": getattr(base_video_meta, "rel_path", None),
                "mime": getattr(base_video_meta, "mime", None),
                "reason": getattr(base_video_meta, "reason", None),
                "fps": getattr(base_video_meta, "fps", None),
                "frames": getattr(base_video_meta, "frame_count", None),
            }

        video_meta = export_video(
            frames_out,
            fps=fps_out,
            options=video_options,
            task="vid2vid",
            audio_source_path=audio_source if has_audio else None,
            extra_metadata=info if video_options else None,
        )
        save_output = parse_bool_value(
            video_options.get("save_output") if video_options else None,
            field="video_options.save_output",
            default=False,
        )
        if save_output and not parse_bool_value(
            getattr(video_meta, "saved", None),
            field="video_meta.saved",
            default=False,
        ):
            reason = str(getattr(video_meta, "reason", "") or "").strip()
            raise RuntimeError(
                "vid2vid: video export failed with save_output=true"
                + (f" ({reason})" if reason else "")
            )
        video_saved = bool(parse_bool_value(getattr(video_meta, "saved", None), field="video_meta.saved", default=False))
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="export.complete",
            stage_name="export",
            backend="diffusers",
            method=str(method),
            video_saved=video_saved,
            final_frame_count=int(len(frames_out)),
        )
        info["video_request_vs_effective_snapshot"] = build_video_request_effective_snapshot(
            request=request,
            plan=plan,
            video_meta=video_meta,
            interpolation_options=vfi_options,
            interpolation_meta=vfi_opts,
            base_video_meta=base_video_meta,
            audio_source_kind="input" if has_audio else "none",
            final_frame_count=len(frames_out),
        )

        preview_n = int(cfg.get("preview_frames") or 48)
        preview = list(frames_out[: max(1, min(preview_n, len(frames_out)))])

        payload = _build_result_payload(
            engine=engine,
            info=info,
            preview_frames=preview,
            request=request,
            video_meta=video_meta,
            video_export_error=None,
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.run.complete",
            stage="run.complete",
            backend="diffusers",
            method=str(method),
            total_pipeline_ms=max(0.0, float(elapsed) * 1000.0),
            final_frame_count=int(len(frames_out)),
            video_saved=video_saved,
        )

        yield ResultEvent(payload=payload)
    finally:
        # Uploaded-file cleanup is handled at the API layer; keep the use-case side-effect-free.
        pass

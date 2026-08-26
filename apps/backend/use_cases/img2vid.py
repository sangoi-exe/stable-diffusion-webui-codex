"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical img2vid orchestration for backend video engines.
Runs the selected video execution path (active WAN22 Diffusers/GGUF lanes plus the native LTX2 branch), including the
truthful LTX2 `executionProfile` stage flow (`distilled` and `one_stage` execute the one-stage native lane; `two_stage`
runs `stage1_sampling -> latent_upsample -> stage2_refine -> decode`), applies optional interpolation,
exports video, and yields progress/result events.
WAN22 keeps exact stage ownership truthful across runtimes: GGUF 5B runs the single-stage lane from `extras.wan_single`,
while dual-stage 14B keeps top-level prompt/negative on the request owner plus explicit second-stage
`extras.wan_low.prompt` / `extras.wan_low.negative_prompt`. Temporal routing requires explicit `extras.img2vid_mode`
(`solo|sliding|svi2|svi2_pro`) and rejects implicit mode fallbacks; WAN22 5B currently only supports the truthful
single-stage `solo` path, while non-solo temporal modes fail loud as not yet implemented. The native LTX2 branch consumes
a local `Ltx2RunResult` (`frames + AudioExportAsset + metadata`) and owns cleanup of generated temp audio after export.

Symbols (top-level; keep in sync; no ghosts):
- `_build_pipeline_telemetry_scope` (function): Creates a mutable task-scoped telemetry context owner for img2vid run/stage events.
- `_emit_pipeline_event` (function): Emits canonical structured pipeline telemetry events (`pipeline.*`) for img2vid.
- `_build_result_payload` (function): Builds the final ResultEvent payload (video export descriptor + optional frames) and attaches warnings.
- `_cleanup_owned_audio_asset` (function): Deletes owned temporary generated-audio artifacts after LTX2 export completes or fails.
- `_ltx_execution_profile` (function): Reads the normalized LTX execution profile from the shared `VideoPlan`.
- `_run_ltx2_img2vid` (function): Runs the native LTX2 img2vid branch, including explicit `two_stage` stage orchestration, and threads generated audio through the shared export seam.
- `_run_stage` (function): Runs a single Diffusers stage and returns its generated frames.
- `_parse_img2vid_temporal_options` (function): Parses and validates explicit img2vid temporal controls from request extras (`solo|sliding|svi2|svi2_pro`).
- `run_img2vid` (function): Orchestrates img2vid generation and yields an `InferenceEvent` stream.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterator, Optional

from apps.backend.core.requests import Img2VidRequest, InferenceEvent, ProgressEvent, ResultEvent
from apps.backend.core.strict_values import parse_bool_value
from apps.backend.engines.wan22.wan22_common import WanStageOptions
from apps.backend.runtime.logging import emit_backend_event
from apps.backend.runtime.processing.datatypes import VideoPlan
from apps.backend.runtime.pipeline_stages.hires_fix import resolve_pipeline_telemetry_context
from apps.backend.runtime.pipeline_stages.video import (
    AudioExportAsset,
    apply_engine_loras,
    apply_wan_stage_loras,
    apply_video_interpolation,
    build_ltx2_two_stage_geometry,
    build_ltx2_video_plan,
    build_video_request_effective_snapshot,
    build_video_plan,
    build_video_result,
    configure_sampler,
    export_video,
    prepare_base_snapshot_video_options,
    read_video_interpolation_options,
    resolve_generated_audio_export_policy,
    resolve_video_output_fps,
)
from apps.backend.use_cases._video_streaming import _yield_wan22_gguf_progress


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
        default_mode="img2vid",
        require_mode=True,
    )
    emit_backend_event(
        event,
        logger="backend.use_cases.img2vid",
        mode=telemetry.mode,
        stage=stage,
        correlation_id=telemetry.correlation_id,
        correlation_source=telemetry.correlation_source,
        task_id=telemetry.task_id,
        **fields,
    )


def _build_result_payload(
    *,
    engine: Any,
    result: Any,
    plan: VideoPlan,
    request: Img2VidRequest,
    video_meta: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = dict(getattr(result, "metadata", {}) or {})

    user_return_frames = parse_bool_value(
        plan.extras.get("video_return_frames"),
        field="extras.video_return_frames",
        default=False,
    )
    video_options = getattr(request, "video_options", None)
    save_output = parse_bool_value(
        video_options.get("save_output") if isinstance(video_options, Mapping) else None,
        field="video_options.save_output",
        default=False,
    )

    video_saved = parse_bool_value(
        video_meta.get("saved") if isinstance(video_meta, dict) else None,
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
        reason = video_meta.get("reason") if isinstance(video_meta, dict) else None
        warnings.append(
            f"Video export failed ({reason or 'unknown error'}). "
            "Frames are returned as a fallback."
        )

    if warnings:
        metadata["warnings"] = warnings

    payload: dict[str, Any] = {"info": engine._to_json(metadata)}  # type: ignore[attr-defined]
    if effective_return_frames:
        payload["images"] = getattr(result, "frames", [])
    if video_saved:
        payload["video"] = {
            "rel_path": video_meta.get("rel_path"),
            "mime": video_meta.get("mime"),
        }
    return payload


def _cleanup_owned_audio_asset(audio_asset: AudioExportAsset | None, *, logger: Any, task: str) -> None:
    if audio_asset is None or not audio_asset.owned_temp:
        return
    path = str(audio_asset.path or "").strip()
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except Exception as exc:
        if logger is not None:
            logger.warning("%s: failed to remove owned temp audio asset '%s': %s", task, path, exc)


def _ltx_execution_profile(plan: VideoPlan) -> str:
    extras = getattr(plan, "extras", None)
    if not isinstance(extras, Mapping):
        return ""
    return str(extras.get("ltx_execution_profile") or "").strip()


def _run_ltx2_img2vid(
    *,
    engine: Any,
    comp: Any,
    request: Img2VidRequest,
    plan: VideoPlan,
    start: float,
    logger: Any,
    telemetry_scope: Any,
) -> Iterator[InferenceEvent]:
    from apps.backend.runtime.families.ltx2.runtime import Ltx2RunResult

    @dataclass(frozen=True)
    class _SamplerOutcome:
        sampler_in: str | None
        scheduler_in: str | None
        sampler_effective: str | None
        scheduler_effective: str | None
        warnings: tuple[str, ...] = ()

    audio_asset: AudioExportAsset | None = None
    try:
        generated_audio_export_policy = resolve_generated_audio_export_policy(
            getattr(request, "video_options", None),
            task="img2vid",
        )
        apply_engine_loras(engine, logger)
        if _ltx_execution_profile(plan) == "two_stage":
            geometry = build_ltx2_two_stage_geometry(plan)
            request_generator = comp.build_request_generator(request=request)
            yield ProgressEvent(stage="stage1_sampling", percent=5.0, message="Running LTX2 stage 1 sampling")
            stage1_result = comp.sample_img2vid_stage(
                request=request,
                plan=plan,
                width=geometry.stage1_width,
                height=geometry.stage1_height,
                num_inference_steps=int(plan.steps),
                guidance_scale=float(plan.guidance_scale if plan.guidance_scale is not None else 4.0),
                generator=request_generator,
            )
            _emit_pipeline_event(
                telemetry_scope,
                "pipeline.stage.complete",
                stage="stage1_sampling.complete",
                stage_name="stage1_sampling",
                backend="ltx2",
                width=int(geometry.stage1_width),
                height=int(geometry.stage1_height),
                latent_height=int(stage1_result.latent_height),
                latent_width=int(stage1_result.latent_width),
            )

            yield ProgressEvent(stage="latent_upsample", percent=35.0, message="Upsampling LTX2 latents")
            upscaled_video_latents = comp.upsample_two_stage_video_latents(
                request=request,
                stage_result=stage1_result,
                geometry=geometry,
            )
            _emit_pipeline_event(
                telemetry_scope,
                "pipeline.stage.complete",
                stage="latent_upsample.complete",
                stage_name="latent_upsample",
                backend="ltx2",
                upscaled_latents_shape=tuple(int(dim) for dim in upscaled_video_latents.shape),
            )

            yield ProgressEvent(stage="stage2_refine", percent=55.0, message="Running LTX2 stage 2 refinement")
            stage2_result = comp.refine_img2vid_two_stage(
                request=request,
                plan=plan,
                geometry=geometry,
                upscaled_video_latents=upscaled_video_latents,
                stage1_result=stage1_result,
                generator=request_generator,
            )
            _emit_pipeline_event(
                telemetry_scope,
                "pipeline.stage.complete",
                stage="stage2_refine.complete",
                stage_name="stage2_refine",
                backend="ltx2",
                latent_height=int(stage2_result.latent_height),
                latent_width=int(stage2_result.latent_width),
                sigmas=tuple(float(value) for value in geometry.stage2_sigmas),
            )

            yield ProgressEvent(stage="decode", percent=80.0, message="Decoding LTX2 outputs")
            runtime_result = comp.decode_stage_result(
                request=request,
                plan=plan,
                stage_result=stage2_result,
                generated_audio_export_policy=generated_audio_export_policy,
                pipeline_name="ltx2_native_img2vid_two_stage",
                metadata_extra={
                    "ltx_two_stage": {
                        "stage1_width": int(geometry.stage1_width),
                        "stage1_height": int(geometry.stage1_height),
                        "final_width": int(geometry.final_width),
                        "final_height": int(geometry.final_height),
                        "stage2_sigmas": [float(value) for value in geometry.stage2_sigmas],
                        "stage2_guidance_scale": float(geometry.stage2_guidance_scale),
                        "stage2_noise_scale": float(geometry.stage2_noise_scale),
                        "distilled_lora": os.path.basename(str(getattr(request, "extras", {}).get("ltx_two_stage_distilled_lora_path") or "")),
                        "spatial_upsampler": os.path.basename(str(getattr(request, "extras", {}).get("ltx_two_stage_spatial_upsampler_path") or "")),
                    }
                },
            )
            _emit_pipeline_event(
                telemetry_scope,
                "pipeline.stage.complete",
                stage="decode.complete",
                stage_name="decode",
                backend="ltx2",
                frame_count=int(len(runtime_result.frames)),
                has_audio=bool(runtime_result.audio_asset is not None),
            )
        else:
            yield ProgressEvent(stage="run", percent=5.0, message="Running LTX2 img2vid")
            runtime_result = comp.run_img2vid(
                request=request,
                plan=plan,
                generated_audio_export_policy=generated_audio_export_policy,
            )
        if not isinstance(runtime_result, Ltx2RunResult):
            raise RuntimeError(
                "LTX2 img2vid runtime must return `Ltx2RunResult`; "
                f"got {type(runtime_result).__name__}."
            )

        frames = list(runtime_result.frames)
        audio_asset = runtime_result.audio_asset
        runtime_meta = dict(runtime_result.metadata)
        audio_source_kind = "generated" if audio_asset is not None else "none"
        generated_frame_count = int(len(frames))

        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="generation.complete",
            stage_name="generation",
            backend="ltx2",
            low_stage_enabled=False,
            frame_count=generated_frame_count,
            has_audio=bool(audio_asset is not None),
        )

        extras = dict(plan.extras) if isinstance(plan.extras, dict) else {}
        vfi_options = read_video_interpolation_options(extras)
        base_video_options = prepare_base_snapshot_video_options(
            getattr(request, "video_options", None),
            task="img2vid",
            interpolation_options=vfi_options,
        )
        base_video_meta: Any = None
        if base_video_options is not None:
            base_video_meta = export_video(
                engine,
                frames,
                plan,
                base_video_options,
                task="img2vid",
                audio_asset=audio_asset,
            )
            if isinstance(base_video_meta, Mapping):
                base_rel_path = str(base_video_meta.get("rel_path") or "").strip()
                if base_rel_path and logger is not None:
                    logger.info(
                        "img2vid: base snapshot exported before post-process: %s",
                        base_rel_path,
                    )

        if vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1:
            yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
        frames, vfi_opts = apply_video_interpolation(frames, options=vfi_options, logger_=logger)
        interpolated_frame_count = int(len(frames))
        plan.fps = resolve_video_output_fps(plan.fps, vfi_opts)
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="interpolation.complete",
            stage_name="interpolation",
            backend="ltx2",
            interpolation_enabled=bool(vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1),
            output_fps=int(plan.fps),
            frame_count=interpolated_frame_count,
        )

        video_meta = export_video(
            engine,
            frames,
            plan,
            getattr(request, "video_options", None),
            task="img2vid",
            audio_asset=audio_asset,
        )
        video_saved = parse_bool_value(
            video_meta.get("saved") if isinstance(video_meta, Mapping) else None,
            field="video_meta.saved",
            default=False,
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="export.complete",
            stage_name="export",
            backend="ltx2",
            video_saved=video_saved,
            final_frame_count=int(len(frames)),
            has_audio=bool(audio_asset is not None),
        )
        export_frame_count: int | None = None
        if isinstance(video_meta, Mapping):
            raw_export_frame_count = video_meta.get("frames", video_meta.get("frame_count"))
            if raw_export_frame_count is not None:
                try:
                    export_frame_count = int(raw_export_frame_count)
                except Exception:
                    export_frame_count = None

        extra_meta: dict[str, Any] = dict(extras)
        if runtime_meta:
            extra_meta["ltx2_runtime"] = runtime_meta
        if vfi_opts is not None:
            extra_meta["video_interpolation"] = vfi_opts
        if base_video_meta is not None:
            extra_meta["video_base_snapshot"] = base_video_meta
        extra_meta["video_request_vs_effective_snapshot"] = build_video_request_effective_snapshot(
            request=request,
            plan=plan,
            video_meta=video_meta,
            interpolation_options=vfi_options,
            interpolation_meta=vfi_opts,
            base_video_meta=base_video_meta,
            audio_source_kind=audio_source_kind,
            final_frame_count=len(frames),
        )
        extra_meta["frame_counts"] = {
            "requested": int(getattr(request, "num_frames", plan.frames) or plan.frames),
            "generated": generated_frame_count,
            "after_interpolation": interpolated_frame_count,
            "after_export": (int(export_frame_count) if export_frame_count is not None else None),
        }

        sampler_effective = str(
            runtime_meta.get("sampler_effective")
            or runtime_meta.get("sampler")
            or getattr(request, "sampler", None)
            or ""
        ).strip() or None
        scheduler_effective = str(
            runtime_meta.get("scheduler_effective")
            or runtime_meta.get("scheduler")
            or getattr(request, "scheduler", None)
            or ""
        ).strip() or None

        elapsed = time.perf_counter() - start
        result = build_video_result(
            engine,
            frames,
            plan,
            _SamplerOutcome(
                sampler_in=getattr(request, "sampler", None),
                scheduler_in=getattr(request, "scheduler", None),
                sampler_effective=sampler_effective,
                scheduler_effective=scheduler_effective,
            ),
            elapsed=elapsed,
            task="img2vid",
            extra=extra_meta,
            video_meta=video_meta,
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.run.complete",
            stage="run.complete",
            backend="ltx2",
            total_pipeline_ms=max(0.0, float(elapsed) * 1000.0),
            final_frame_count=int(len(frames)),
            video_saved=video_saved,
            has_audio=bool(audio_asset is not None),
        )

        yield ResultEvent(
            payload=_build_result_payload(
                engine=engine,
                result=result,
                plan=plan,
                request=request,
                video_meta=video_meta,
            )
        )
    finally:
        _cleanup_owned_audio_asset(audio_asset, logger=logger, task="img2vid")


def _run_stage(
    pipe: Any,
    plan: VideoPlan,
    *,
    prompt: str,
    negative_prompt: str | None,
    init_image: Any | None,
) -> list[Any]:
    if pipe is None:
        raise RuntimeError("img2vid requires a Diffusers pipeline (single or per-stage)")
    import torch

    with torch.inference_mode():
        output = pipe(
            image=init_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=plan.frames,
            num_inference_steps=plan.steps,
            height=plan.height,
            width=plan.width,
            guidance_scale=plan.guidance_scale,
        )
    if hasattr(output, "frames"):
        return list(output.frames[0])
    raise RuntimeError("img2vid pipeline returned no frames")

@dataclass(frozen=True)
class _Img2VidSlidingOptions:
    window_frames: int
    window_stride: int
    window_commit_frames: int
    anchor_alpha: float
    reset_anchor_to_base: bool
    chunk_seed_mode: str
    chunk_buffer_mode: str


@dataclass(frozen=True)
class _Img2VidTemporalOptions:
    mode: str
    sliding: _Img2VidSlidingOptions | None = None
    svi2: _Img2VidSlidingOptions | None = None
    svi2_pro: _Img2VidSlidingOptions | None = None


def _parse_img2vid_temporal_options(extras: Mapping[str, Any], *, total_frames: int) -> _Img2VidTemporalOptions:
    raw_mode = extras.get("img2vid_mode")
    if raw_mode is None:
        raise RuntimeError("img2vid_mode is required and must be one of ('solo','sliding','svi2','svi2_pro').")
    mode = str(raw_mode or "").strip().lower()
    if not mode:
        raise RuntimeError("img2vid_mode must not be empty.")
    if mode == "chunk":
        raise RuntimeError("img2vid_mode='chunk' is no longer supported; use 'solo','sliding','svi2', or 'svi2_pro'.")
    if mode not in {"solo", "sliding", "svi2", "svi2_pro"}:
        raise RuntimeError(f"img2vid_mode must be one of ('solo','sliding','svi2','svi2_pro'), got: {mode!r}")

    has_chunk = extras.get("img2vid_chunk_frames") not in (None, "")
    has_overlap = extras.get("img2vid_overlap_frames") not in (None, "")
    has_anchor = extras.get("img2vid_anchor_alpha") not in (None, "")
    has_reset_anchor_to_base = extras.get("img2vid_reset_anchor_to_base") not in (None, "")
    has_seed_mode = extras.get("img2vid_chunk_seed_mode") not in (None, "")
    has_buffer_mode = extras.get("img2vid_chunk_buffer_mode") not in (None, "")
    has_window_frames = extras.get("img2vid_window_frames") not in (None, "")
    has_window_stride = extras.get("img2vid_window_stride") not in (None, "")
    has_window_commit = extras.get("img2vid_window_commit_frames") not in (None, "")

    def _parse_anchor_alpha() -> float:
        raw_anchor = extras.get("img2vid_anchor_alpha", 0.2)
        try:
            anchor_alpha = float(raw_anchor)
        except Exception as exc:  # noqa: BLE001 - fail loud contract
            raise RuntimeError(f"img2vid_anchor_alpha must be a float, got: {raw_anchor!r}") from exc
        if anchor_alpha < 0.0 or anchor_alpha > 1.0:
            raise RuntimeError(f"img2vid_anchor_alpha must be within [0, 1], got: {anchor_alpha}")
        return anchor_alpha

    def _parse_reset_anchor_to_base(*, temporal_mode: str) -> bool:
        mode_value = str(temporal_mode).strip().lower()
        default_reset = False
        raw_value = extras.get("img2vid_reset_anchor_to_base", default_reset)
        if isinstance(raw_value, bool):
            reset_anchor = raw_value
        elif isinstance(raw_value, str):
            parsed = raw_value.strip().lower()
            if parsed in {"true", "1"}:
                reset_anchor = True
            elif parsed in {"false", "0"}:
                reset_anchor = False
            else:
                raise RuntimeError(
                    "img2vid_reset_anchor_to_base must be a boolean literal ('true'/'false'/'1'/'0'), "
                    f"got: {raw_value!r}"
                )
        elif isinstance(raw_value, (int, float)) and float(raw_value) in {0.0, 1.0}:
            reset_anchor = bool(int(raw_value))
        else:
            raise RuntimeError(f"img2vid_reset_anchor_to_base must be a boolean, got: {raw_value!r}")

        if mode_value in {"svi2", "svi2_pro"} and reset_anchor:
            raise RuntimeError(
                f"img2vid_mode='{mode_value}' requires img2vid_reset_anchor_to_base=false "
                "(SVI continuity profile is slot-locked)."
            )
        return bool(reset_anchor)

    def _parse_seed_mode(*, temporal_mode: str) -> str:
        if str(temporal_mode) == "sliding":
            default_seed_mode = "fixed"
        elif str(temporal_mode) in {"svi2", "svi2_pro"}:
            default_seed_mode = "increment"
        else:
            default_seed_mode = "increment"
        raw_seed_mode = str(extras.get("img2vid_chunk_seed_mode", default_seed_mode) or "").strip().lower()
        if raw_seed_mode not in {"fixed", "increment", "random"}:
            raise RuntimeError(
                "img2vid_chunk_seed_mode must be one of ('fixed','increment','random'), "
                f"got: {raw_seed_mode!r}"
            )
        return raw_seed_mode

    def _parse_buffer_mode() -> str:
        raw_buffer_mode = extras.get("img2vid_chunk_buffer_mode")
        if raw_buffer_mode in (None, ""):
            raw_buffer_mode = os.getenv("CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE", "hybrid")
        chunk_buffer_mode = str(raw_buffer_mode or "").strip().lower()
        if chunk_buffer_mode not in {"hybrid", "ram", "ram+hd"}:
            raise RuntimeError(
                "img2vid_chunk_buffer_mode must be one of ('hybrid','ram','ram+hd'), "
                f"got: {raw_buffer_mode!r}"
            )
        return chunk_buffer_mode

    if mode == "solo":
        has_temporal_fields = any(
            (
                has_chunk,
                has_overlap,
                has_anchor,
                has_reset_anchor_to_base,
                has_seed_mode,
                has_buffer_mode,
                has_window_frames,
                has_window_stride,
                has_window_commit,
            )
        )
        if has_temporal_fields:
            raise RuntimeError("img2vid_mode='solo' does not allow temporal controls (chunk/window/anchor/seed/buffer).")
        return _Img2VidTemporalOptions(mode="solo")

    mode_label = str(mode)
    if has_chunk or has_overlap:
        raise RuntimeError(f"img2vid_mode='{mode_label}' does not allow img2vid_chunk_frames/img2vid_overlap_frames.")
    if not (has_window_frames and has_window_stride and has_window_commit):
        raise RuntimeError(
            f"img2vid_mode='{mode_label}' requires img2vid_window_frames, img2vid_window_stride, "
            "and img2vid_window_commit_frames."
        )

    raw_window_frames = extras.get("img2vid_window_frames")
    try:
        window_frames = int(raw_window_frames)
    except Exception as exc:  # noqa: BLE001 - fail loud contract
        raise RuntimeError(f"img2vid_window_frames must be an integer, got: {raw_window_frames!r}") from exc
    if window_frames < 9 or window_frames > 401:
        raise RuntimeError(f"img2vid_window_frames must be within [9, 401], got: {window_frames}")
    if (window_frames - 1) % 4 != 0:
        raise RuntimeError(f"img2vid_window_frames must satisfy 4n+1, got: {window_frames}")
    if window_frames >= int(total_frames):
        raise RuntimeError(
            "img2vid_window_frames must be smaller than the requested total frame count "
            f"(window={window_frames} total={int(total_frames)})"
        )

    raw_window_stride = extras.get("img2vid_window_stride")
    try:
        window_stride = int(raw_window_stride)
    except Exception as exc:  # noqa: BLE001 - fail loud contract
        raise RuntimeError(f"img2vid_window_stride must be an integer, got: {raw_window_stride!r}") from exc
    if window_stride < 1:
        raise RuntimeError(f"img2vid_window_stride must be >= 1, got: {window_stride}")
    if window_stride >= window_frames:
        raise RuntimeError(
            "img2vid_window_stride must be smaller than img2vid_window_frames "
            f"(stride={window_stride} window={window_frames})"
        )
    if window_stride % 4 != 0:
        raise RuntimeError(
            "img2vid_window_stride must be aligned to temporal scale=4 "
            f"(stride={window_stride})."
        )

    raw_window_commit = extras.get("img2vid_window_commit_frames")
    try:
        window_commit_frames = int(raw_window_commit)
    except Exception as exc:  # noqa: BLE001 - fail loud contract
        raise RuntimeError(f"img2vid_window_commit_frames must be an integer, got: {raw_window_commit!r}") from exc
    if window_commit_frames < window_stride or window_commit_frames > window_frames:
        raise RuntimeError(
            "img2vid_window_commit_frames must be within "
            "[img2vid_window_stride, img2vid_window_frames] "
            f"(commit={window_commit_frames} stride={window_stride} window={window_frames})"
        )
    if (window_commit_frames - window_stride) < 4:
        raise RuntimeError(
            "img2vid_window_commit_frames must keep at least 4 committed overlap frames beyond stride "
            f"(commit={window_commit_frames} stride={window_stride})."
        )

    window_opts = _Img2VidSlidingOptions(
        window_frames=window_frames,
        window_stride=window_stride,
        window_commit_frames=window_commit_frames,
        anchor_alpha=_parse_anchor_alpha(),
        reset_anchor_to_base=_parse_reset_anchor_to_base(temporal_mode=mode_label),
        chunk_seed_mode=_parse_seed_mode(temporal_mode=mode_label),
        chunk_buffer_mode=_parse_buffer_mode(),
    )
    if mode_label == "sliding":
        return _Img2VidTemporalOptions(mode="sliding", sliding=window_opts)
    if mode_label == "svi2":
        return _Img2VidTemporalOptions(mode="svi2", svi2=window_opts)
    return _Img2VidTemporalOptions(mode="svi2_pro", svi2_pro=window_opts)


def run_img2vid(
    *,
    engine,
    comp,
    request: Img2VidRequest,
) -> Iterator[InferenceEvent]:
    logger = getattr(engine, "_logger", None)
    telemetry_scope = _build_pipeline_telemetry_scope(mode="img2vid")
    if getattr(request, "init_image", None) is None:
        raise RuntimeError("img2vid requires 'init_image'")

    engine_id = str(getattr(engine, "engine_id", "") or "").strip().lower()
    plan = build_ltx2_video_plan(request) if engine_id == "ltx2" else build_video_plan(request)
    start = time.perf_counter()
    if engine_id == "ltx2":
        pipe = None
        high_model = None
        low_model = None
        backend_variant = "ltx2"
    else:
        pipe = getattr(comp, "pipeline", None)
        high_model = getattr(comp, "pipeline_high", None)
        low_model = getattr(comp, "pipeline_low", None)
        backend_variant = "gguf" if (pipe is None and high_model is None and low_model is None) else "diffusers"
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.run.start",
        stage="run.start",
        backend=backend_variant,
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
        backend=backend_variant,
        frames=int(plan.frames),
        width=int(plan.width),
        height=int(plan.height),
        steps=int(plan.steps),
    )

    yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing img2vid")

    if engine_id == "ltx2":
        yield from _run_ltx2_img2vid(
            engine=engine,
            comp=comp,
            request=request,
            plan=plan,
            start=start,
            logger=logger,
            telemetry_scope=telemetry_scope,
        )
        return

    if pipe is None and high_model is None and low_model is None:
        from apps.backend.runtime.families.wan22.config import build_wan22_gguf_run_config
        from apps.backend.runtime.families.wan22 import wan22 as gguf

        extras = dict(plan.extras) if isinstance(plan.extras, dict) else {}
        temporal_opts = _parse_img2vid_temporal_options(extras, total_frames=plan.frames)
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="temporal_options.complete",
            stage_name="temporal_options",
            backend="gguf",
            temporal_mode=str(temporal_opts.mode),
        )

        cfg = None
        frames: list[Any] | None = None

        if temporal_opts.mode == "solo":
            cfg = build_wan22_gguf_run_config(
                request=request,
                device=getattr(comp, "device", None),
                dtype=getattr(comp, "dtype", "fp16"),
                logger=logger,
            )

            stream_gguf = gguf.stream_img2vid_single if cfg.single is not None else gguf.stream_img2vid
            for ev in stream_gguf(cfg, logger=logger):
                if not isinstance(ev, dict):
                    raise RuntimeError(f"WAN22 GGUF: invalid stream event type: {type(ev)}")
                if ev.get("type") == "progress":
                    pe = _yield_wan22_gguf_progress(ev)
                    if pe is not None:
                        yield pe
                    continue
                if ev.get("type") == "result":
                    raw_frames = ev.get("frames", [])
                    if raw_frames is None:
                        frames = []
                    elif isinstance(raw_frames, list):
                        frames = raw_frames
                    elif isinstance(raw_frames, tuple):
                        frames = list(raw_frames)
                    else:
                        raise RuntimeError(
                            "WAN22 GGUF: invalid result payload for 'frames' "
                            f"(expected sequence, got {type(raw_frames).__name__})"
                        )
                    break
                raise RuntimeError(f"WAN22 GGUF: unknown stream event type: {ev.get('type')!r}")
        elif temporal_opts.mode == "sliding":
            cfg = build_wan22_gguf_run_config(
                request=request,
                device=getattr(comp, "device", None),
                dtype=getattr(comp, "dtype", "fp16"),
                logger=logger,
            )
            if cfg.single is not None:
                raise NotImplementedError("WAN22 5B img2vid temporal mode 'sliding' not yet implemented")
            if temporal_opts.sliding is None:
                raise RuntimeError("img2vid_mode='sliding' selected but sliding options are missing.")
            sliding_opts = temporal_opts.sliding
            if logger:
                logger.info(
                    "[img2vid] sliding mode enabled: window=%d stride=%d commit=%d anchor_alpha=%.3f reset_anchor_to_base=%s seed_mode=%s buffer_mode=%s",
                    sliding_opts.window_frames,
                    sliding_opts.window_stride,
                    sliding_opts.window_commit_frames,
                    sliding_opts.anchor_alpha,
                    bool(sliding_opts.reset_anchor_to_base),
                    sliding_opts.chunk_seed_mode,
                    sliding_opts.chunk_buffer_mode,
                )
                if str(sliding_opts.chunk_seed_mode) != "fixed":
                    logger.warning(
                        "[img2vid] sliding mode continuity risk: chunk_seed_mode=%s can cause per-window temporal drift; "
                        "prefer 'fixed' for stable motion.",
                        str(sliding_opts.chunk_seed_mode),
                    )

            for ev in gguf.stream_img2vid_sliding_window(
                cfg,
                window_frames=int(sliding_opts.window_frames),
                window_stride=int(sliding_opts.window_stride),
                window_commit_frames=int(sliding_opts.window_commit_frames),
                anchor_alpha=float(sliding_opts.anchor_alpha),
                reset_anchor_to_base=bool(sliding_opts.reset_anchor_to_base),
                chunk_seed_mode=str(sliding_opts.chunk_seed_mode),
                chunk_buffer_mode=str(sliding_opts.chunk_buffer_mode),
                logger=logger,
            ):
                if not isinstance(ev, dict):
                    raise RuntimeError(f"WAN22 GGUF: invalid stream event type: {type(ev)}")
                if ev.get("type") == "progress":
                    pe = _yield_wan22_gguf_progress(ev)
                    if pe is not None:
                        yield pe
                    continue
                if ev.get("type") == "result":
                    raw_frames = ev.get("frames", [])
                    if raw_frames is None:
                        frames = []
                    elif isinstance(raw_frames, list):
                        frames = raw_frames
                    elif isinstance(raw_frames, tuple):
                        frames = list(raw_frames)
                    else:
                        raise RuntimeError(
                            "WAN22 GGUF: invalid result payload for 'frames' "
                            f"(expected sequence, got {type(raw_frames).__name__})"
                        )
                    break
                raise RuntimeError(f"WAN22 GGUF: unknown stream event type: {ev.get('type')!r}")
        elif temporal_opts.mode == "svi2":
            cfg = build_wan22_gguf_run_config(
                request=request,
                device=getattr(comp, "device", None),
                dtype=getattr(comp, "dtype", "fp16"),
                logger=logger,
            )
            if cfg.single is not None:
                raise NotImplementedError("WAN22 5B img2vid temporal mode 'svi2' not yet implemented")
            if temporal_opts.svi2 is None:
                raise RuntimeError("img2vid_mode='svi2' selected but svi2 options are missing.")
            svi_opts = temporal_opts.svi2
            if logger:
                logger.info(
                    "[img2vid] svi2 mode enabled: window=%d stride=%d commit=%d anchor_alpha=%.3f reset_anchor_to_base=%s seed_mode=%s buffer_mode=%s",
                    svi_opts.window_frames,
                    svi_opts.window_stride,
                    svi_opts.window_commit_frames,
                    svi_opts.anchor_alpha,
                    bool(svi_opts.reset_anchor_to_base),
                    svi_opts.chunk_seed_mode,
                    svi_opts.chunk_buffer_mode,
                )
                if str(svi_opts.chunk_seed_mode) == "fixed":
                    logger.warning(
                        "[img2vid] svi2 mode continuity risk: chunk_seed_mode=%s can lock per-window motion diversity; "
                        "prefer 'increment' or 'random' for long-form variation.",
                        str(svi_opts.chunk_seed_mode),
                    )

            for ev in gguf.stream_img2vid_svi2(
                cfg,
                window_frames=int(svi_opts.window_frames),
                window_stride=int(svi_opts.window_stride),
                window_commit_frames=int(svi_opts.window_commit_frames),
                anchor_alpha=float(svi_opts.anchor_alpha),
                chunk_seed_mode=str(svi_opts.chunk_seed_mode),
                chunk_buffer_mode=str(svi_opts.chunk_buffer_mode),
                logger=logger,
                ):
                if not isinstance(ev, dict):
                    raise RuntimeError(f"WAN22 GGUF: invalid stream event type: {type(ev)}")
                if ev.get("type") == "progress":
                    pe = _yield_wan22_gguf_progress(ev)
                    if pe is not None:
                        yield pe
                    continue
                if ev.get("type") == "result":
                    raw_frames = ev.get("frames", [])
                    if raw_frames is None:
                        frames = []
                    elif isinstance(raw_frames, list):
                        frames = raw_frames
                    elif isinstance(raw_frames, tuple):
                        frames = list(raw_frames)
                    else:
                        raise RuntimeError(
                            "WAN22 GGUF: invalid result payload for 'frames' "
                            f"(expected sequence, got {type(raw_frames).__name__})"
                        )
                    break
                raise RuntimeError(f"WAN22 GGUF: unknown stream event type: {ev.get('type')!r}")
        elif temporal_opts.mode == "svi2_pro":
            cfg = build_wan22_gguf_run_config(
                request=request,
                device=getattr(comp, "device", None),
                dtype=getattr(comp, "dtype", "fp16"),
                logger=logger,
            )
            if cfg.single is not None:
                raise NotImplementedError("WAN22 5B img2vid temporal mode 'svi2_pro' not yet implemented")
            if temporal_opts.svi2_pro is None:
                raise RuntimeError("img2vid_mode='svi2_pro' selected but svi2_pro options are missing.")
            svi_opts = temporal_opts.svi2_pro
            if logger:
                logger.info(
                    "[img2vid] svi2_pro mode enabled: window=%d stride=%d commit=%d anchor_alpha=%.3f reset_anchor_to_base=%s seed_mode=%s buffer_mode=%s",
                    svi_opts.window_frames,
                    svi_opts.window_stride,
                    svi_opts.window_commit_frames,
                    svi_opts.anchor_alpha,
                    bool(svi_opts.reset_anchor_to_base),
                    svi_opts.chunk_seed_mode,
                    svi_opts.chunk_buffer_mode,
                )
                if str(svi_opts.chunk_seed_mode) == "fixed":
                    logger.warning(
                        "[img2vid] svi2_pro mode continuity risk: chunk_seed_mode=%s can lock per-window motion diversity; "
                        "prefer 'increment' or 'random' for long-form variation.",
                        str(svi_opts.chunk_seed_mode),
                    )

            for ev in gguf.stream_img2vid_svi2_pro(
                cfg,
                window_frames=int(svi_opts.window_frames),
                window_stride=int(svi_opts.window_stride),
                window_commit_frames=int(svi_opts.window_commit_frames),
                anchor_alpha=float(svi_opts.anchor_alpha),
                chunk_seed_mode=str(svi_opts.chunk_seed_mode),
                chunk_buffer_mode=str(svi_opts.chunk_buffer_mode),
                logger=logger,
            ):
                if not isinstance(ev, dict):
                    raise RuntimeError(f"WAN22 GGUF: invalid stream event type: {type(ev)}")
                if ev.get("type") == "progress":
                    pe = _yield_wan22_gguf_progress(ev)
                    if pe is not None:
                        yield pe
                    continue
                if ev.get("type") == "result":
                    raw_frames = ev.get("frames", [])
                    if raw_frames is None:
                        frames = []
                    elif isinstance(raw_frames, list):
                        frames = raw_frames
                    elif isinstance(raw_frames, tuple):
                        frames = list(raw_frames)
                    else:
                        raise RuntimeError(
                            "WAN22 GGUF: invalid result payload for 'frames' "
                            f"(expected sequence, got {type(raw_frames).__name__})"
                        )
                    break
                raise RuntimeError(f"WAN22 GGUF: unknown stream event type: {ev.get('type')!r}")
        else:
            raise RuntimeError(f"Unsupported img2vid_mode: {temporal_opts.mode!r}")

        if not frames:
            raise RuntimeError("WAN22 GGUF: produced no frames")
        if cfg is None:
            raise RuntimeError("WAN22 GGUF: runtime config resolution failed (cfg is None).")
        generated_frame_count = int(len(frames))
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="generation.complete",
            stage_name="generation",
            backend="gguf",
            temporal_mode=str(temporal_opts.mode),
            frame_count=int(generated_frame_count),
        )
        vfi_options = read_video_interpolation_options(plan.extras)
        base_video_options = prepare_base_snapshot_video_options(
            getattr(request, "video_options", None),
            task="img2vid",
            interpolation_options=vfi_options,
        )
        base_video_meta: Any = None
        if base_video_options is not None:
            base_video_meta = export_video(engine, frames, plan, base_video_options, task="img2vid")
            if isinstance(base_video_meta, Mapping):
                base_rel_path = str(base_video_meta.get("rel_path") or "").strip()
                if base_rel_path:
                    logger.info(
                        "img2vid: base snapshot exported before post-process: %s",
                        base_rel_path,
                    )

        if vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1:
            yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
        frames, vfi_opts = apply_video_interpolation(frames, options=vfi_options, logger_=logger)
        interpolated_frame_count = int(len(frames))
        plan.fps = resolve_video_output_fps(plan.fps, vfi_opts)
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="interpolation.complete",
            stage_name="interpolation",
            backend="gguf",
            temporal_mode=str(temporal_opts.mode),
            interpolation_enabled=bool(vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1),
            output_fps=int(plan.fps),
            frame_count=int(interpolated_frame_count),
        )

        video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None), task="img2vid")
        video_saved = parse_bool_value(
            video_meta.get("saved") if isinstance(video_meta, Mapping) else None,
            field="video_meta.saved",
            default=False,
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="export.complete",
            stage_name="export",
            backend="gguf",
            temporal_mode=str(temporal_opts.mode),
            video_saved=video_saved,
            final_frame_count=int(len(frames)),
        )
        export_frame_count: int | None = None
        if isinstance(video_meta, Mapping):
            raw_export_frame_count = video_meta.get("frames", video_meta.get("frame_count"))
            if raw_export_frame_count is not None:
                try:
                    export_frame_count = int(raw_export_frame_count)
                except Exception:  # noqa: BLE001 - metadata remains optional for diagnostics
                    export_frame_count = None

        @dataclass(frozen=True)
        class _SamplerOutcome:
            sampler_in: str | None
            scheduler_in: str | None
            sampler_effective: str | None
            scheduler_effective: str | None
            warnings: tuple[str, ...] = ()

        extra_meta: dict[str, Any] = dict(plan.extras) if isinstance(plan.extras, dict) else {}
        if vfi_opts is not None:
            extra_meta["video_interpolation"] = vfi_opts
        if base_video_meta is not None:
            extra_meta["video_base_snapshot"] = base_video_meta
        extra_meta["video_request_vs_effective_snapshot"] = build_video_request_effective_snapshot(
            request=request,
            plan=plan,
            video_meta=video_meta,
            interpolation_options=vfi_options,
            interpolation_meta=vfi_opts,
            base_video_meta=base_video_meta,
            audio_source_kind="none",
            final_frame_count=len(frames),
        )
        extra_meta["frame_counts"] = {
            "requested": int(getattr(request, "num_frames", plan.frames) or plan.frames),
            "generated": int(generated_frame_count),
            "after_interpolation": int(interpolated_frame_count),
            "after_export": (int(export_frame_count) if export_frame_count is not None else None),
        }
        if cfg.low is not None:
            extra_meta["sampler_low"] = {
                "sampler_in": cfg.low.sampler,
                "scheduler_in": cfg.low.scheduler,
                "sampler": cfg.low.sampler,
                "scheduler": cfg.low.scheduler,
            }

        primary_stage = cfg.single if cfg.single is not None else cfg.high

        elapsed = time.perf_counter() - start
        result = build_video_result(
            engine,
            frames,
            plan,
            _SamplerOutcome(
                sampler_in=getattr(request, "sampler", None),
                scheduler_in=getattr(request, "scheduler", None),
                sampler_effective=(
                    primary_stage.sampler
                    if primary_stage is not None
                    else getattr(request, "sampler", None)
                ),
                scheduler_effective=(
                    primary_stage.scheduler
                    if primary_stage is not None
                    else getattr(request, "scheduler", None)
                ),
            ),
            elapsed=elapsed,
            task="img2vid",
            extra=extra_meta,
            video_meta=video_meta,
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.run.complete",
            stage="run.complete",
            backend="gguf",
            temporal_mode=str(temporal_opts.mode),
            total_pipeline_ms=max(0.0, float(elapsed) * 1000.0),
            final_frame_count=int(len(frames)),
            video_saved=video_saved,
        )

        yield ResultEvent(
            payload=_build_result_payload(
                engine=engine,
                result=result,
                plan=plan,
                request=request,
                video_meta=video_meta,
            )
        )
        return

    apply_engine_loras(engine, logger)

    active_pipe_hi = high_model or pipe
    if active_pipe_hi is None:
        raise RuntimeError("img2vid requires a Diffusers pipeline (single or per-stage)")

    extras = dict(plan.extras)
    wan_high_cfg = extras.get("wan_high")
    wan_hi_opts = WanStageOptions.from_mapping(wan_high_cfg) if isinstance(wan_high_cfg, dict) else None
    high_prompt = str(getattr(request, "prompt", None) or "").strip()
    if not high_prompt:
        raise RuntimeError("img2vid requires a non-empty request.prompt.")
    high_negative_prompt = str(getattr(request, "negative_prompt", "") or "").strip()
    if wan_hi_opts and wan_hi_opts.loras:
        apply_wan_stage_loras(
            pipe=active_pipe_hi,
            stage_loras=wan_hi_opts.loras,
            logger_=logger,
            stage_label="high",
        )

    outcome_hi = configure_sampler(active_pipe_hi, plan, logger)

    yield ProgressEvent(stage="run_high", percent=5.0, message="Stage 1 (High Noise)")
    frames = _run_stage(
        active_pipe_hi,
        plan,
        prompt=high_prompt,
        negative_prompt=high_negative_prompt,
        init_image=getattr(request, "init_image", None),
    )
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.stage.complete",
        stage="run_high.complete",
        stage_name="run_high",
        backend="diffusers",
        frame_count=int(len(frames)),
    )

    active_pipe_lo = low_model or pipe
    outcome_lo = None

    if active_pipe_lo is not None and frames:
        wan_low_cfg = extras.get("wan_low")
        wan_opts = WanStageOptions.from_mapping(wan_low_cfg) if isinstance(wan_low_cfg, dict) else None
        if wan_opts is None or wan_opts.prompt is None:
            raise RuntimeError("img2vid requires extras.wan_low.prompt to be set.")
        low_prompt = str(wan_opts.prompt).strip()
        if not low_prompt:
            raise RuntimeError("img2vid requires a non-empty low-stage prompt.")
        low_negative_prompt = (
            str(wan_opts.negative_prompt).strip()
            if wan_opts and wan_opts.negative_prompt is not None
            else str(getattr(request, "negative_prompt", "") or "").strip()
        )
        if wan_opts and wan_opts.loras:
            apply_wan_stage_loras(
                pipe=active_pipe_lo,
                stage_loras=wan_opts.loras,
                logger_=logger,
                stage_label="low",
                use_transformer_2=active_pipe_lo is pipe,
            )

        outcome_lo = configure_sampler(active_pipe_lo, plan, logger)
        yield ProgressEvent(stage="run_low", percent=50.0, message="Stage 2 (Low Noise)")
        frames = _run_stage(
            active_pipe_lo,
            plan,
            prompt=low_prompt,
            negative_prompt=low_negative_prompt,
            init_image=frames[-1],
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="run_low.complete",
            stage_name="run_low",
            backend="diffusers",
            frame_count=int(len(frames)),
        )
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.stage.complete",
        stage="generation.complete",
        stage_name="generation",
        backend="diffusers",
        low_stage_enabled=bool(outcome_lo is not None),
        frame_count=int(len(frames)),
    )
    vfi_options = read_video_interpolation_options(extras)
    base_video_options = prepare_base_snapshot_video_options(
        getattr(request, "video_options", None),
        task="img2vid",
        interpolation_options=vfi_options,
    )
    base_video_meta: Any = None
    if base_video_options is not None:
        base_video_meta = export_video(engine, frames, plan, base_video_options, task="img2vid")
        if isinstance(base_video_meta, Mapping):
            base_rel_path = str(base_video_meta.get("rel_path") or "").strip()
            if base_rel_path:
                logger.info(
                    "img2vid: base snapshot exported before post-process: %s",
                    base_rel_path,
                )

    if vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1:
        yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
    frames, vfi_opts = apply_video_interpolation(frames, options=vfi_options, logger_=logger)
    plan.fps = resolve_video_output_fps(plan.fps, vfi_opts)
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.stage.complete",
        stage="interpolation.complete",
        stage_name="interpolation",
        backend="diffusers",
        interpolation_enabled=bool(vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1),
        output_fps=int(plan.fps),
        frame_count=int(len(frames)),
    )

    video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None), task="img2vid")
    video_saved = parse_bool_value(
        video_meta.get("saved") if isinstance(video_meta, Mapping) else None,
        field="video_meta.saved",
        default=False,
    )
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.stage.complete",
        stage="export.complete",
        stage_name="export",
        backend="diffusers",
        video_saved=video_saved,
        final_frame_count=int(len(frames)),
    )

    extra_meta: dict[str, Any] = dict(extras) if isinstance(extras, dict) else {}
    if vfi_opts is not None:
        extra_meta["video_interpolation"] = vfi_opts
    if base_video_meta is not None:
        extra_meta["video_base_snapshot"] = base_video_meta
    extra_meta["video_request_vs_effective_snapshot"] = build_video_request_effective_snapshot(
        request=request,
        plan=plan,
        video_meta=video_meta,
        interpolation_options=vfi_options,
        interpolation_meta=vfi_opts,
        base_video_meta=base_video_meta,
        audio_source_kind="none",
        final_frame_count=len(frames),
    )
    if outcome_lo is not None:
        extra_meta["sampler_low"] = {
            "sampler_in": getattr(outcome_lo, "sampler_in", None),
            "scheduler_in": getattr(outcome_lo, "scheduler_in", None),
            "sampler": getattr(outcome_lo, "sampler_effective", None),
            "scheduler": getattr(outcome_lo, "scheduler_effective", None),
        }

    elapsed = time.perf_counter() - start
    result = build_video_result(
        engine,
        frames,
        plan,
        outcome_hi,
        elapsed=elapsed,
        task="img2vid",
        extra=extra_meta,
        video_meta=video_meta,
    )
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.run.complete",
        stage="run.complete",
        backend="diffusers",
        total_pipeline_ms=max(0.0, float(elapsed) * 1000.0),
        final_frame_count=int(len(frames)),
        video_saved=video_saved,
    )

    yield ResultEvent(
        payload=_build_result_payload(
            engine=engine,
            result=result,
            plan=plan,
            request=request,
            video_meta=video_meta,
        )
    )

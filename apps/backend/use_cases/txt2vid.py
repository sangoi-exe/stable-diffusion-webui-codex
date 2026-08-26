"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical txt2vid orchestration for backend video engines.
Runs the selected video execution path (active WAN22 Diffusers/GGUF lanes plus the native LTX2 branch), including the
truthful LTX2 `executionProfile` stage flow (`distilled` and `one_stage` execute the one-stage native lane; `two_stage`
runs `stage1_sampling -> latent_upsample -> stage2_refine -> decode`), applies optional interpolation,
exports the resulting video, and yields progress/result events.
WAN22 keeps exact stage ownership truthful across runtimes: GGUF 5B runs the single-stage lane from `extras.wan_single`,
while dual-stage 14B keeps top-level prompt/negative on the request owner plus selector-only `extras.wan_high` and explicit
second-stage `extras.wan_low`. The native LTX2 branch consumes a local
`Ltx2RunResult` (`frames + AudioExportAsset + metadata`) and owns cleanup of generated temp audio after export.

Symbols (top-level; keep in sync; no ghosts):
- `_build_pipeline_telemetry_scope` (function): Creates a mutable task-scoped telemetry context owner for txt2vid run/stage events.
- `_emit_pipeline_event` (function): Emits canonical structured pipeline telemetry events (`pipeline.*`) for txt2vid.
- `_build_result_payload` (function): Builds the final ResultEvent payload (video export descriptor + optional frames) and attaches warnings.
- `_cleanup_owned_audio_asset` (function): Deletes owned temporary generated-audio artifacts after LTX2 export completes or fails.
- `_ltx_execution_profile` (function): Reads the normalized LTX execution profile from the shared `VideoPlan`.
- `_run_ltx2_txt2vid` (function): Runs the native LTX2 txt2vid branch, including explicit `two_stage` stage orchestration, and threads generated audio through the shared export seam.
- `_run_pipeline` (function): Runs a Diffusers txt2vid pipeline and returns generated frames.
- `run_txt2vid` (function): Orchestrates txt2vid generation and yields an `InferenceEvent` stream.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterator, Optional

from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent, Txt2VidRequest
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
        default_mode="txt2vid",
        require_mode=True,
    )
    emit_backend_event(
        event,
        logger="backend.use_cases.txt2vid",
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
    request: Txt2VidRequest,
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


def _run_ltx2_txt2vid(
    *,
    engine: Any,
    comp: Any,
    request: Txt2VidRequest,
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
            task="txt2vid",
        )
        apply_engine_loras(engine, logger)
        if _ltx_execution_profile(plan) == "two_stage":
            geometry = build_ltx2_two_stage_geometry(plan)
            request_generator = comp.build_request_generator(request=request)
            yield ProgressEvent(stage="stage1_sampling", percent=5.0, message="Running LTX2 stage 1 sampling")
            stage1_result = comp.sample_txt2vid_stage(
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
            stage2_result = comp.refine_txt2vid_two_stage(
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
                pipeline_name="ltx2_native_txt2vid_two_stage",
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
            yield ProgressEvent(stage="run", percent=5.0, message="Running LTX2 txt2vid")
            runtime_result = comp.run_txt2vid(
                request=request,
                plan=plan,
                generated_audio_export_policy=generated_audio_export_policy,
            )
        if not isinstance(runtime_result, Ltx2RunResult):
            raise RuntimeError(
                "LTX2 txt2vid runtime must return `Ltx2RunResult`; "
                f"got {type(runtime_result).__name__}."
            )

        frames = list(runtime_result.frames)
        audio_asset = runtime_result.audio_asset
        runtime_meta = dict(runtime_result.metadata)
        audio_source_kind = "generated" if audio_asset is not None else "none"

        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="generation.complete",
            stage_name="generation",
            backend="ltx2",
            frame_count=int(len(frames)),
            has_audio=bool(audio_asset is not None),
        )
        vfi_options = read_video_interpolation_options(plan.extras)
        base_video_options = prepare_base_snapshot_video_options(
            getattr(request, "video_options", None),
            task="txt2vid",
            interpolation_options=vfi_options,
        )
        base_video_meta: Any = None
        if base_video_options is not None:
            base_video_meta = export_video(
                engine,
                frames,
                plan,
                base_video_options,
                task="txt2vid",
                audio_asset=audio_asset,
            )
            if isinstance(base_video_meta, Mapping):
                base_rel_path = str(base_video_meta.get("rel_path") or "").strip()
                if base_rel_path and logger is not None:
                    logger.info(
                        "txt2vid: base snapshot exported before post-process: %s",
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
            backend="ltx2",
            interpolation_enabled=bool(vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1),
            output_fps=int(plan.fps),
            frame_count=int(len(frames)),
        )

        video_meta = export_video(
            engine,
            frames,
            plan,
            getattr(request, "video_options", None),
            task="txt2vid",
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

        extra_meta: dict[str, Any] = dict(plan.extras)
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
            task="txt2vid",
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
        _cleanup_owned_audio_asset(audio_asset, logger=logger, task="txt2vid")


def _run_pipeline(
    pipe: Any,
    plan: VideoPlan,
    request: Txt2VidRequest,
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
) -> list[Any]:
    prompt_text = str(prompt if prompt is not None else request.prompt or "").strip()
    if not prompt_text:
        raise RuntimeError("txt2vid requires a non-empty prompt.")
    negative_prompt_text = (
        str(negative_prompt).strip()
        if negative_prompt is not None
        else str(getattr(request, "negative_prompt", None) or "").strip()
    )
    import torch

    with torch.inference_mode():
        output = pipe(
            prompt=prompt_text,
            negative_prompt=negative_prompt_text,
            num_frames=plan.frames,
            num_inference_steps=plan.steps,
            height=plan.height,
            width=plan.width,
            guidance_scale=plan.guidance_scale,
        )
    if hasattr(output, "frames"):
        return list(output.frames[0])
    if hasattr(output, "images"):
        return list(output.images)
    raise RuntimeError("txt2vid pipeline returned no frames")

def run_txt2vid(
    *,
    engine,
    comp,
    request: Txt2VidRequest,
) -> Iterator[InferenceEvent]:
    logger = getattr(engine, "_logger", None)
    telemetry_scope = _build_pipeline_telemetry_scope(mode="txt2vid")
    engine_id = str(getattr(engine, "engine_id", "") or "").strip().lower()
    plan = build_ltx2_video_plan(request) if engine_id == "ltx2" else build_video_plan(request)
    start = time.perf_counter()
    if engine_id == "ltx2":
        pipe = None
        backend_variant = "ltx2"
    else:
        pipe = getattr(comp, "pipeline", None)
        backend_variant = "gguf" if pipe is None else "diffusers"

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

    yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing txt2vid")

    if engine_id == "ltx2":
        yield from _run_ltx2_txt2vid(
            engine=engine,
            comp=comp,
            request=request,
            plan=plan,
            start=start,
            logger=logger,
            telemetry_scope=telemetry_scope,
        )
        return

    if pipe is None:
        from apps.backend.runtime.families.wan22.config import build_wan22_gguf_run_config
        from apps.backend.runtime.families.wan22 import wan22 as gguf

        cfg = build_wan22_gguf_run_config(
            request=request,
            device=getattr(comp, "device", None),
            dtype=getattr(comp, "dtype", "fp16"),
            logger=logger,
        )

        stream_gguf = gguf.stream_txt2vid_single if cfg.single is not None else gguf.stream_txt2vid

        frames: list[Any] | None = None
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

        if not frames:
            raise RuntimeError("WAN22 GGUF: produced no frames")
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.stage.complete",
            stage="generation.complete",
            stage_name="generation",
            backend="gguf",
            frame_count=int(len(frames)),
        )
        vfi_options = read_video_interpolation_options(plan.extras)
        base_video_options = prepare_base_snapshot_video_options(
            getattr(request, "video_options", None),
            task="txt2vid",
            interpolation_options=vfi_options,
        )
        base_video_meta: Any = None
        if base_video_options is not None:
            base_video_meta = export_video(engine, frames, plan, base_video_options, task="txt2vid")
            if isinstance(base_video_meta, Mapping):
                base_rel_path = str(base_video_meta.get("rel_path") or "").strip()
                if base_rel_path:
                    logger.info(
                        "txt2vid: base snapshot exported before post-process: %s",
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
            backend="gguf",
            interpolation_enabled=bool(vfi_options is not None and vfi_options.enabled and (vfi_options.times or 0) > 1),
            output_fps=int(plan.fps),
            frame_count=int(len(frames)),
        )

        video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None), task="txt2vid")
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
            video_saved=video_saved,
            final_frame_count=int(len(frames)),
        )

        @dataclass(frozen=True)
        class _SamplerOutcome:
            sampler_in: str | None
            scheduler_in: str | None
            sampler_effective: str | None
            scheduler_effective: str | None
            warnings: tuple[str, ...] = ()

        extra_meta: dict[str, Any] = dict(plan.extras)
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
            task="txt2vid",
            extra=extra_meta,
            video_meta=video_meta,
        )
        _emit_pipeline_event(
            telemetry_scope,
            "pipeline.run.complete",
            stage="run.complete",
            backend="gguf",
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

    extras = dict(plan.extras)
    wan_high_cfg = extras.get("wan_high")
    wan_hi_opts = WanStageOptions.from_mapping(wan_high_cfg) if isinstance(wan_high_cfg, dict) else None
    prompt_text = str(getattr(request, "prompt", None) or "").strip()
    if not prompt_text:
        raise RuntimeError("txt2vid requires a non-empty request.prompt.")
    negative_prompt_text = str(getattr(request, "negative_prompt", None) or "").strip()
    if wan_hi_opts and wan_hi_opts.loras:
        apply_wan_stage_loras(
            pipe=pipe,
            stage_loras=wan_hi_opts.loras,
            logger_=logger,
            stage_label="high",
        )

    apply_engine_loras(engine, logger)

    sampler_outcome = configure_sampler(pipe, plan, logger)

    yield ProgressEvent(stage="run", percent=5.0, message="Running pipeline")
    frames = _run_pipeline(
        pipe,
        plan,
        request,
        prompt=prompt_text,
        negative_prompt=negative_prompt_text,
    )
    _emit_pipeline_event(
        telemetry_scope,
        "pipeline.stage.complete",
        stage="generation.complete",
        stage_name="generation",
        backend="diffusers",
        frame_count=int(len(frames)),
    )
    vfi_options = read_video_interpolation_options(plan.extras)
    base_video_options = prepare_base_snapshot_video_options(
        getattr(request, "video_options", None),
        task="txt2vid",
        interpolation_options=vfi_options,
    )
    base_video_meta: Any = None
    if base_video_options is not None:
        base_video_meta = export_video(engine, frames, plan, base_video_options, task="txt2vid")
        if isinstance(base_video_meta, Mapping):
            base_rel_path = str(base_video_meta.get("rel_path") or "").strip()
            if base_rel_path:
                logger.info(
                    "txt2vid: base snapshot exported before post-process: %s",
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

    video_meta = export_video(engine, frames, plan, getattr(request, "video_options", None), task="txt2vid")
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

    extra_meta: dict[str, Any] = dict(plan.extras)
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

    elapsed = time.perf_counter() - start
    result = build_video_result(
        engine,
        frames,
        plan,
        sampler_outcome,
        elapsed=elapsed,
        task="txt2vid",
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

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared helpers for Codex video generation pipelines.
Builds generic `VideoPlan`/`VideoResult` helpers plus the strict LTX2 video-plan seam, applies LoRAs, configures sampler/scheduler,
explicitly rejects native-only sampler variants unsupported by the diffusers video bridge, preflights and applies stage-scoped WAN
Diffusers LoRAs, resolves generated-audio export policy before video runs, and assembles export metadata.

Symbols (top-level; keep in sync; no ghosts):
- `logger` (constant): Module logger used by video pipeline helpers.
- `AudioExportAsset` (dataclass): Pre-export audio asset descriptor used by shared video export helpers.
- `GeneratedAudioExportPolicy` (dataclass): Normalized policy for mux-capable generated-audio export decisions.
- `build_video_plan` (function): Normalizes request attributes into a `VideoPlan`.
- `build_ltx2_video_plan` (function): Builds a strict LTX2 `VideoPlan` that requires explicit router-owned fields instead of generic fallbacks.
- `Ltx2TwoStageGeometry` (dataclass): Execution-only geometry/scalar contract for the explicit LTX2 `two_stage` profile.
- `build_ltx2_two_stage_geometry` (function): Derives the fixed stage-1/stage-2 execution geometry from final public LTX2 output dimensions.
- `apply_engine_loras` (function): Applies globally selected LoRAs to the engine (when supported).
- `apply_wan_stage_loras` (function): Preflights and applies ordered WAN Diffusers stage LoRAs to the selected transformer owner.
- `configure_sampler` (function): Applies sampler/scheduler configuration to a component given a `VideoPlan`.
- `read_video_interpolation_options` (function): Parses `extras.video_interpolation` into typed interpolation options when present.
- `apply_video_interpolation` (function): Applies the shared interpolation stage and returns `(frames_out, interpolation_metadata)`.
- `resolve_video_output_fps` (function): Computes output fps from request/base fps and interpolation metadata.
- `resolve_generated_audio_export_policy` (function): Validates generated-audio export intent before heavy video generation work.
- `export_video` (function): Exports a frame sequence to a video file according to request options and a task label (stable output dir).
- `prepare_base_snapshot_video_options` (function): Builds a fail-loud snapshot export options payload for base-video persistence before post-processing.
- `build_video_request_effective_snapshot` (function): Builds an immutable request-vs-effective execution snapshot for shared backend video metadata.
- `assemble_video_metadata` (function): Builds a metadata dict describing the generated video.
- `build_video_result` (function): Returns a `VideoResult` bundle for API/UI consumers.
- `__all__` (constant): Explicit export list for the module.
"""

from __future__ import annotations
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import safetensors.torch as sf

from apps.backend.core.params.video import VideoInterpolationOptions
from apps.backend.core.strict_values import parse_bool_value
from apps.backend.patchers.lora_state_dict import load_lora
from apps.backend.runtime.adapters.lora import model_lora_keys_unet
from apps.backend.runtime.adapters.lora.preflight import (
    build_standard_shape_patch_dict_from_shape_map,
    format_shape_compatibility_samples,
    shapeify_patch_dict,
    validate_shape_patch_dict,
)
from apps.backend.runtime.adapters.lora import selections as lora_selections
from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_tensor_shapes
from apps.backend.runtime.model_registry.capabilities import ENGINE_SURFACES, semantic_engine_for_engine_id
from apps.backend.engines.util.schedulers import apply_sampler_scheduler, SamplerKind
from apps.backend.runtime.processing.datatypes import VideoPlan, VideoResult
from apps.backend.video.export.ffmpeg_exporter import resolve_video_export_container
from apps.backend.video.interpolation import maybe_interpolate

logger = get_backend_logger(__name__)
_LTX2_INTERNAL_EXTRA_KEYS = frozenset(
    {
        "ltx_two_stage_distilled_lora_path",
        "ltx_two_stage_spatial_upsampler_path",
    }
)
_LTX2_TWO_STAGE_STAGE2_SIGMAS = (0.909375, 0.725, 0.421875)
_LTX2_TWO_STAGE_STAGE2_GUIDANCE_SCALE = 1.0


@dataclass(slots=True, frozen=True)
class AudioExportAsset:
    """Pre-export audio asset descriptor for mux-capable video workflows."""

    path: str
    owned_temp: bool
    sample_rate_hz: int | None = None
    channels: int | None = None


@dataclass(slots=True, frozen=True)
class GeneratedAudioExportPolicy:
    """Normalized mux policy for generated-audio video workflows."""

    save_output: bool
    format: str
    container: str | None
    materialize_audio_asset: bool


@dataclass(slots=True, frozen=True)
class Ltx2TwoStageGeometry:
    """Execution-only geometry/scalars for the explicit LTX2 `two_stage` profile."""

    final_width: int
    final_height: int
    stage1_width: int
    stage1_height: int
    stage2_sigmas: tuple[float, ...]
    stage2_noise_scale: float
    stage2_guidance_scale: float


def build_video_plan(request: Any) -> VideoPlan:
    """Normalize request attributes into a ``VideoPlan``."""

    extras_raw = getattr(request, "extras", {}) or {}
    extras: dict[str, Any]
    if isinstance(extras_raw, Mapping):
        extras = dict(extras_raw)
    else:
        extras = {}

    return VideoPlan(
        sampler_name=getattr(request, "sampler", None),
        scheduler_name=getattr(request, "scheduler", None),
        steps=int(getattr(request, "steps", 30) or 30),
        frames=int(getattr(request, "num_frames", 16) or 16),
        fps=int(getattr(request, "fps", 24) or 24),
        width=int(getattr(request, "width", 768) or 768),
        height=int(getattr(request, "height", 432) or 432),
        guidance_scale=getattr(request, "guidance_scale", None),
        extras=extras,
    )


def build_ltx2_video_plan(request: Any) -> VideoPlan:
    """Build a strict `VideoPlan` for the LTX2 execution path."""

    extras_raw = getattr(request, "extras", {}) or {}
    extras: dict[str, Any]
    if isinstance(extras_raw, Mapping):
        extras = dict(extras_raw)
    else:
        extras = {}
    for key in _LTX2_INTERNAL_EXTRA_KEYS:
        extras.pop(key, None)

    required_ints = {
        "steps": getattr(request, "steps", None),
        "num_frames": getattr(request, "num_frames", None),
        "fps": getattr(request, "fps", None),
        "width": getattr(request, "width", None),
        "height": getattr(request, "height", None),
    }
    normalized: dict[str, int] = {}
    for field_name, raw_value in required_ints.items():
        if raw_value is None:
            raise RuntimeError(f"LTX2 video request missing required '{field_name}'.")
        value = int(raw_value)
        if value <= 0:
            raise RuntimeError(f"LTX2 video request field '{field_name}' must be > 0, got {value}.")
        normalized[field_name] = value

    return VideoPlan(
        sampler_name=getattr(request, "sampler", None),
        scheduler_name=getattr(request, "scheduler", None),
        steps=normalized["steps"],
        frames=normalized["num_frames"],
        fps=normalized["fps"],
        width=normalized["width"],
        height=normalized["height"],
        guidance_scale=getattr(request, "guidance_scale", None),
        extras=extras,
    )


def build_ltx2_two_stage_geometry(plan: VideoPlan) -> Ltx2TwoStageGeometry:
    """Derive the execution-only two-stage LTX2 geometry from final public output dimensions."""

    final_width = int(plan.width)
    final_height = int(plan.height)
    if final_width <= 0 or final_height <= 0:
        raise RuntimeError(
            f"LTX2 two_stage requires positive final width/height; got {final_width}x{final_height}."
        )
    if final_width % 64 != 0 or final_height % 64 != 0:
        raise RuntimeError(
            "LTX2 two_stage requires final width/height divisible by 64 because stage 1 runs at half resolution; "
            f"got {final_width}x{final_height}."
        )
    return Ltx2TwoStageGeometry(
        final_width=final_width,
        final_height=final_height,
        stage1_width=final_width // 2,
        stage1_height=final_height // 2,
        stage2_sigmas=_LTX2_TWO_STAGE_STAGE2_SIGMAS,
        stage2_noise_scale=float(_LTX2_TWO_STAGE_STAGE2_SIGMAS[0]),
        stage2_guidance_scale=_LTX2_TWO_STAGE_STAGE2_GUIDANCE_SCALE,
    )


def _collect_target_shape_by_key(model: Any) -> dict[str, tuple[int, ...]]:
    state_dict = model.state_dict()
    return {str(key): tuple(int(dim) for dim in tensor.shape) for key, tensor in state_dict.items()}


def _resolve_wan_stage_lora_owner(
    *,
    pipe: Any,
    stage_label: str,
    use_transformer_2: bool,
) -> tuple[str, Any, bool]:
    if use_transformer_2:
        transformer_2 = getattr(pipe, "transformer_2", None)
        if transformer_2 is None:
            raise RuntimeError(
                f"{stage_label} stage LoRA requested the low-stage transformer_2 owner, but the pipeline does not expose it."
            )
        return "transformer_2", transformer_2, True
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise RuntimeError(f"{stage_label} stage LoRA requires a pipeline with a non-null transformer module.")
    return "transformer", transformer, False


def _preflight_wan_stage_lora_file(
    *,
    path: str,
    to_load: Mapping[str, Any],
    target_shapes: Mapping[str, tuple[int, ...]],
    target_label: str,
) -> dict[Any, tuple]:
    suffix = Path(path).suffix.lower()
    if suffix not in {".safetensor", ".safetensors"}:
        raise RuntimeError(f"{target_label} stage LoRA path must be a .safetensors file: {path}")

    shape_map = read_safetensors_tensor_shapes(Path(path))
    header_result = build_standard_shape_patch_dict_from_shape_map(shape_map, to_load=to_load)
    if not header_result.requires_materialized_preflight and not header_result.shape_patch_dict:
        raise RuntimeError(
            "WAN diffusers stage LoRA key layout mismatch: no compatible layers were found for "
            f"'{path}' on the active {target_label}."
        )
    if header_result.shape_patch_dict and not header_result.requires_materialized_preflight:
        header_summary = validate_shape_patch_dict(
            header_result.shape_patch_dict,
            target_shape_by_key=target_shapes,
        )
        if header_summary.mismatches:
            raise RuntimeError(
                "WAN diffusers stage LoRA structural preflight failed for '{path}' on {label}. "
                "shape_compatible_targets={compatible}/{total}. samples={samples}".format(
                    path=path,
                    label=target_label,
                    compatible=header_summary.compatible_targets,
                    total=header_summary.total_targets,
                    samples=format_shape_compatibility_samples(header_summary),
                )
            )

    try:
        tensor_map = sf.load_file(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load WAN diffusers stage LoRA '{path}': {exc}") from exc

    patch_dict, _unused = load_lora(tensor_map, to_load=dict(to_load))
    if not patch_dict:
        raise RuntimeError(
            "WAN diffusers stage LoRA key layout mismatch: no compatible layers were found for "
            f"'{path}' on the active {target_label}."
        )
    materialized_summary = validate_shape_patch_dict(
        shapeify_patch_dict(patch_dict),
        target_shape_by_key=target_shapes,
    )
    if materialized_summary.mismatches:
        raise RuntimeError(
            "WAN diffusers stage LoRA structural preflight failed for '{path}' on {label}. "
            "shape_compatible_targets={compatible}/{total}. samples={samples}".format(
                path=path,
                label=target_label,
                compatible=materialized_summary.compatible_targets,
                total=materialized_summary.total_targets,
                samples=format_shape_compatibility_samples(materialized_summary),
            )
        )
    return patch_dict


def apply_wan_stage_loras(
    *,
    pipe: Any,
    stage_loras: tuple[tuple[str, float], ...],
    logger_: logging.Logger | None = None,
    stage_label: str,
    use_transformer_2: bool = False,
) -> None:
    target_name, target_module, load_into_transformer_2 = _resolve_wan_stage_lora_owner(
        pipe=pipe,
        stage_label=stage_label,
        use_transformer_2=use_transformer_2,
    )
    to_load = model_lora_keys_unet(target_module)
    target_shapes = _collect_target_shape_by_key(target_module)
    prepared: list[tuple[str, float, str]] = []
    total_stage_loras = len(stage_loras)

    for index, (lora_path, lora_weight) in enumerate(stage_loras):
        _preflight_wan_stage_lora_file(
            path=lora_path,
            to_load=to_load,
            target_shapes=target_shapes,
            target_label=target_name,
        )
        adapter_name = f"wan_{stage_label}_stage_lora_{index}"
        if logger_:
            logger_.info(
                "[wan] loading %s-stage LoRA %d/%d: %s (weight=%s adapter=%s target=%s)",
                stage_label,
                index + 1,
                total_stage_loras,
                lora_path,
                lora_weight,
                adapter_name,
                target_name,
            )
        prepared.append((lora_path, float(lora_weight), adapter_name))

    if not stage_loras:
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()  # type: ignore[attr-defined]
        return
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(f"{stage_label} stage LoRA requires a pipeline with 'load_lora_weights'.")
    if not hasattr(pipe, "set_adapters"):
        raise RuntimeError(f"{stage_label} stage LoRA requires a pipeline with 'set_adapters' for multi-LoRA support.")
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()  # type: ignore[attr-defined]

    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    try:
        for lora_path, lora_weight, adapter_name in prepared:
            kwargs = {"adapter_name": adapter_name}
            if load_into_transformer_2:
                kwargs["load_into_transformer_2"] = True
            pipe.load_lora_weights(lora_path, **kwargs)  # type: ignore[attr-defined]
            adapter_names.append(adapter_name)
            adapter_weights.append(float(lora_weight))
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)  # type: ignore[attr-defined]
    except Exception as exc:
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()  # type: ignore[attr-defined]
        raise RuntimeError(
            f"{stage_label} stage LoRA apply failed after successful preflight on target '{target_name}'."
        ) from exc


def apply_engine_loras(engine: Any, logger_: logging.Logger | None = None) -> Any | None:
    """Apply globally selected LoRAs to the engine, returning stats when available."""

    # Lazy import to keep pipeline stage module dependency-light for non-LoRA users.
    from apps.backend.patchers.lora_apply import apply_loras_to_engine

    try:
        selections = list(lora_selections.get_selections())
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        raise RuntimeError(f"Failed to fetch LoRA selections: {exc}") from exc

    engine_id = str(getattr(engine, "engine_id", "") or "").strip()
    supports_lora: bool | None = None
    if engine_id:
        try:
            semantic_engine = semantic_engine_for_engine_id(engine_id)
        except KeyError:
            semantic_engine = None
        else:
            supports_lora = bool(ENGINE_SURFACES[semantic_engine].supports_lora)

    if supports_lora is False:
        if selections:
            raise RuntimeError(
                f"Video pipeline LoRA selections are unsupported for engine '{engine_id}'. "
                "Remove LoRA selections for this request."
            )
        return None

    if supports_lora is None and selections:
        label = engine_id or type(engine).__name__
        raise RuntimeError(
            "Video pipeline LoRA selections require a capability-registered engine with "
            f"`supports_lora=true`; could not validate engine '{label}'."
        )

    if not selections:
        if supports_lora is not True:
            return None
        # Empty-selection runs must still clear any stale LoRA state from prior requests.
        return apply_loras_to_engine(engine, [])

    stats = apply_loras_to_engine(engine, selections)
    if logger_:
        emit_backend_message(
            "[native] applied LoRA(s)",
            logger=getattr(logger_, "name", __name__),
            files=getattr(stats, "files", len(selections)),
            params_touched=getattr(stats, "params_touched", 0),
        )
    return stats


def configure_sampler(component: Any, plan: VideoPlan, logger_: logging.Logger | None = None) -> Any:
    """Apply sampler/scheduler selection on a Diffusers pipeline component."""

    sampler_name = plan.sampler_name or "euler"
    scheduler_name = plan.scheduler_name or "simple"
    sampler_kind = SamplerKind.from_string(sampler_name)
    if sampler_kind is SamplerKind.UNI_PC_BH2:
        raise RuntimeError(
            "Video scheduler bridge does not implement sampler 'uni-pc bh2'; "
            "use 'uni-pc' for this execution path."
        )
    outcome = apply_sampler_scheduler(
        component,
        sampler_kind,
        scheduler_name,
    )
    for warning in outcome.warnings:
        if logger_:
            emit_backend_message(
                "video sampler warning",
                logger=getattr(logger_, "name", __name__),
                level=logging.WARNING,
                warning=warning,
            )
        else:  # pragma: no cover - fallback logging
            emit_backend_message(
                "video sampler warning",
                logger=__name__,
                level=logging.WARNING,
                warning=warning,
            )
    return outcome


def read_video_interpolation_options(extras: Mapping[str, Any] | None) -> VideoInterpolationOptions | None:
    if not isinstance(extras, Mapping):
        return None
    cfg = extras.get("video_interpolation")
    if not isinstance(cfg, Mapping):
        return None
    model_raw = cfg.get("model")
    if model_raw is None:
        model = None
    else:
        if not isinstance(model_raw, str):
            raise RuntimeError(
                "video_interpolation.model must be a string when provided "
                f"(got {type(model_raw).__name__})."
            )
        model = model_raw.strip() or None

    times_raw = cfg.get("times")
    if times_raw is None:
        times = None
    else:
        if isinstance(times_raw, bool) or not isinstance(times_raw, int):
            raise RuntimeError(
                "video_interpolation.times must be an integer when provided "
                f"(got {type(times_raw).__name__})."
            )
        times = int(times_raw)
        if times < 2:
            raise RuntimeError(f"video_interpolation.times must be >= 2 when provided (got {times}).")

    return VideoInterpolationOptions(
        enabled=parse_bool_value(cfg.get("enabled"), field="video_interpolation.enabled", default=False),
        model=model,
        times=times,
    )


def apply_video_interpolation(
    frames: Sequence[Any],
    *,
    options: VideoInterpolationOptions | None,
    logger_: logging.Logger | None = None,
) -> tuple[list[Any], dict[str, Any] | None]:
    frames_list = frames if isinstance(frames, list) else list(frames)
    if options is None:
        return frames_list, None

    opts = options.as_dict()
    if options.enabled and (options.times or 0) > 1:
        out_frames, meta = maybe_interpolate(
            frames_list,
            enabled=options.enabled,
            model=options.model,
            times=options.times or 2,
            logger=logger_ if logger_ is not None else logger,
        )
        out_list = out_frames if isinstance(out_frames, list) else list(out_frames)
        return out_list, {**opts, "result": meta}

    return frames_list, opts


def resolve_video_output_fps(base_fps: int, interpolation_meta: Mapping[str, Any] | None) -> int:
    fps_base = int(base_fps) if int(base_fps) > 0 else 1
    if not isinstance(interpolation_meta, Mapping):
        return fps_base

    result = interpolation_meta.get("result")
    if not isinstance(result, Mapping) or not bool(result.get("applied", False)):
        return fps_base

    raw_times = interpolation_meta.get("times")
    try:
        times = int(raw_times) if raw_times is not None else 1
    except Exception:
        times = 1
    if times <= 1:
        return fps_base
    return fps_base * times


def resolve_generated_audio_export_policy(video_options: Any, *, task: str) -> GeneratedAudioExportPolicy:
    options: dict[str, Any] = dict(video_options) if isinstance(video_options, Mapping) else {}
    save_output = parse_bool_value(
        options.get("save_output"),
        field="video_options.save_output",
        default=False,
    )
    format_value = str(options.get("format") or "video/h264-mp4").strip() or "video/h264-mp4"
    if not save_output:
        return GeneratedAudioExportPolicy(
            save_output=False,
            format=format_value,
            container=None,
            materialize_audio_asset=False,
        )

    try:
        container, _codec_kind = resolve_video_export_container(format_value)
    except Exception as exc:
        raise RuntimeError(
            f"{task}: generated-audio video export requires a supported video_options.format when save_output=true; "
            f"got {format_value!r} ({exc})."
        ) from exc
    if container not in {"mp4", "webm"}:
        raise RuntimeError(
            f"{task}: generated-audio video export requires mp4 or webm when video_options.save_output=true; "
            f"got {format_value!r}."
        )
    return GeneratedAudioExportPolicy(
        save_output=True,
        format=format_value,
        container=container,
        materialize_audio_asset=True,
    )


def export_video(
    engine: Any,
    frames: Sequence[Any],
    plan: VideoPlan,
    video_options: Any,
    *,
    task: str,
    audio_asset: AudioExportAsset | None = None,
) -> Any:
    audio_policy = (
        resolve_generated_audio_export_policy(video_options, task=task)
        if audio_asset is not None
        else None
    )
    save_output = (
        bool(audio_policy.save_output)
        if audio_policy is not None
        else parse_bool_value(
            video_options.get("save_output") if isinstance(video_options, Mapping) else None,
            field="video_options.save_output",
            default=False,
        )
    )
    normalized_audio_source = str(audio_asset.path).strip() if audio_asset is not None else ""
    if normalized_audio_source and not save_output:
        raise RuntimeError(
            f"{task}: audio-bearing video result requires video_options.save_output=true to preserve audio output."
        )
    if not hasattr(engine, "_maybe_export_video"):
        if save_output:
            raise RuntimeError(
                f"{task}: video export requested (save_output=true), but engine does not implement _maybe_export_video."
            )
        return None

    video_meta = engine._maybe_export_video(  # type: ignore[attr-defined]
        frames,
        fps=plan.fps,
        options=video_options,
        task=task,
        audio_source_path=normalized_audio_source,
    )
    if save_output:
        saved = parse_bool_value(
            video_meta.get("saved") if isinstance(video_meta, Mapping) else None,
            field="video_meta.saved",
            default=False,
        )
        if not saved:
            reason = ""
            if isinstance(video_meta, Mapping):
                reason = str(video_meta.get("reason") or "").strip()
            raise RuntimeError(
                f"{task}: video export failed with save_output=true"
                + (f" ({reason})" if reason else "")
            )
    return video_meta


def prepare_base_snapshot_video_options(
    video_options: Any,
    *,
    task: str,
    interpolation_options: VideoInterpolationOptions | None,
) -> dict[str, Any] | None:
    save_output = parse_bool_value(
        video_options.get("save_output") if isinstance(video_options, Mapping) else None,
        field="video_options.save_output",
        default=False,
    )
    if not save_output:
        return None

    interpolation_enabled = bool(
        interpolation_options is not None
        and interpolation_options.enabled
        and int(interpolation_options.times or 0) > 1
    )
    if not interpolation_enabled:
        return None

    normalized_options: dict[str, Any] = dict(video_options) if isinstance(video_options, Mapping) else {}
    base_prefix = str(normalized_options.get("filename_prefix") or task or "video").strip() or "video"
    normalized_options["filename_prefix"] = f"{base_prefix}_base"
    normalized_options["save_output"] = True
    return normalized_options


def _snapshot_clone(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _snapshot_clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_clone(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _export_meta_field(meta: Any, *, key: str) -> Any:
    if isinstance(meta, Mapping):
        return meta.get(key)
    return getattr(meta, key, None)


def _normalized_export_meta(meta: Any) -> dict[str, Any] | None:
    if meta is None:
        return None
    if isinstance(meta, Mapping):
        payload: dict[str, Any] = dict(meta)
    else:
        payload = {
            "saved": getattr(meta, "saved", None),
            "rel_path": getattr(meta, "rel_path", None),
            "mime": getattr(meta, "mime", None),
            "reason": getattr(meta, "reason", None),
            "fps": getattr(meta, "fps", None),
            "frames": getattr(meta, "frame_count", None),
            "has_audio": getattr(meta, "has_audio", None),
        }
    if payload.get("frames") is None and payload.get("frame_count") is not None:
        payload["frames"] = payload.get("frame_count")
    payload.pop("frame_count", None)
    return _snapshot_clone(payload)


def _container_supports_audio(format_value: Any) -> bool:
    normalized = str(format_value or "").strip().lower()
    if normalized in {"video/gif", "image/gif", "gif"}:
        return False
    return True


def build_video_request_effective_snapshot(
    *,
    request: Any,
    plan: VideoPlan,
    video_meta: Any,
    interpolation_options: VideoInterpolationOptions | None,
    interpolation_meta: Mapping[str, Any] | None,
    base_video_meta: Any = None,
    audio_source_kind: str = "none",
    final_frame_count: int,
) -> dict[str, Any]:
    """Build an immutable snapshot of requested vs effective shared video execution settings."""

    normalized_audio_source_kind = str(audio_source_kind or "none").strip().lower() or "none"
    if normalized_audio_source_kind not in {"none", "input", "generated"}:
        raise RuntimeError(
            "audio_source_kind must be 'none', 'input', or 'generated'."
        )

    raw_video_options = getattr(request, "video_options", None)
    video_options: dict[str, Any] = dict(raw_video_options) if isinstance(raw_video_options, Mapping) else {}
    extras_raw = getattr(request, "extras", {})
    extras: dict[str, Any] = dict(extras_raw) if isinstance(extras_raw, Mapping) else {}

    requested_return_frames = parse_bool_value(
        extras.get("video_return_frames"),
        field="extras.video_return_frames",
        default=False,
    )
    requested_save_output = parse_bool_value(
        video_options.get("save_output"),
        field="video_options.save_output",
        default=False,
    )
    requested_save_metadata = parse_bool_value(
        video_options.get("save_metadata"),
        field="video_options.save_metadata",
        default=False,
    )
    requested_trim_to_audio = parse_bool_value(
        video_options.get("trim_to_audio"),
        field="video_options.trim_to_audio",
        default=False,
    )
    requested_pingpong = parse_bool_value(
        video_options.get("pingpong"),
        field="video_options.pingpong",
        default=False,
    )

    requested_interpolation_enabled = bool(interpolation_options is not None and interpolation_options.enabled)
    requested_interpolation_times = (
        int(interpolation_options.times)
        if interpolation_options is not None and interpolation_options.times is not None
        else None
    )
    requested_interpolation_toggle = bool(
        requested_interpolation_enabled and int(requested_interpolation_times or 0) > 1
    )
    requested_base_snapshot = bool(requested_save_output and requested_interpolation_toggle)

    video_saved = parse_bool_value(
        _export_meta_field(video_meta, key="saved"),
        field="video_meta.saved",
        default=False,
    )
    export_failed = bool(requested_save_output and not video_saved)
    effective_return_frames = bool(requested_return_frames or (not requested_save_output) or export_failed)

    interpolation_result_raw = (
        interpolation_meta.get("result")
        if isinstance(interpolation_meta, Mapping)
        else None
    )
    interpolation_applied = parse_bool_value(
        interpolation_result_raw.get("applied") if isinstance(interpolation_result_raw, Mapping) else None,
        field="video_interpolation.result.applied",
        default=False,
    )

    base_snapshot_saved = parse_bool_value(
        _export_meta_field(base_video_meta, key="saved"),
        field="video_base_snapshot.saved",
        default=False,
    )

    format_effective = str(video_options.get("format") or "video/h264-mp4")
    pix_fmt_effective = str(video_options.get("pix_fmt") or "yuv420p")
    crf_effective = int(video_options.get("crf", 23) or 23)
    loop_count_effective = int(video_options.get("loop_count", 0) or 0)
    trim_to_audio_effective = bool(
        requested_trim_to_audio
        and requested_save_output
        and video_saved
        and normalized_audio_source_kind != "none"
        and _container_supports_audio(format_effective)
    )
    save_metadata_effective = bool(
        requested_save_metadata
        and requested_save_output
        and video_saved
    )
    pingpong_effective = bool(
        requested_pingpong
        and requested_save_output
        and video_saved
    )

    requested_snapshot = {
        "video_options": {
            "format": video_options.get("format"),
            "pix_fmt": video_options.get("pix_fmt"),
            "crf": video_options.get("crf"),
            "loop_count": video_options.get("loop_count"),
            "pingpong": requested_pingpong,
            "save_output": requested_save_output,
            "save_metadata": requested_save_metadata,
            "trim_to_audio": requested_trim_to_audio,
        },
        "video_return_frames": requested_return_frames,
        "video_interpolation": (
            interpolation_options.as_dict() if interpolation_options is not None else {"enabled": False}
        ),
        "video_base_snapshot": {"requested": requested_base_snapshot},
        "input_geometry": {
            "width": int(getattr(request, "width", plan.width) or plan.width),
            "height": int(getattr(request, "height", plan.height) or plan.height),
            "fps": int(getattr(request, "fps", plan.fps) or plan.fps),
            "frames": int(getattr(request, "num_frames", final_frame_count) or final_frame_count),
        },
        "audio_source_kind": normalized_audio_source_kind,
    }

    effective_has_audio = bool(
        video_saved
        and _container_supports_audio(format_effective)
        and parse_bool_value(
            _export_meta_field(video_meta, key="has_audio"),
            field="video_meta.has_audio",
            default=False,
        )
    )
    effective_snapshot = {
        "video_options": {
            "format": format_effective,
            "pix_fmt": pix_fmt_effective,
            "crf": crf_effective,
            "loop_count": loop_count_effective,
            "pingpong": pingpong_effective,
            "save_output": bool(video_saved),
            "save_metadata": save_metadata_effective,
            "trim_to_audio": trim_to_audio_effective,
        },
        "video_return_frames": effective_return_frames,
        "video_interpolation": (
            _snapshot_clone(interpolation_meta)
            if isinstance(interpolation_meta, Mapping)
            else {"enabled": requested_interpolation_enabled, "result": {"applied": interpolation_applied}}
        ),
        "video_base_snapshot": _normalized_export_meta(base_video_meta),
        "video_export": _normalized_export_meta(video_meta),
        "output_geometry": {
            "width": int(plan.width),
            "height": int(plan.height),
            "fps": int(plan.fps),
            "frames": int(final_frame_count),
        },
        "audio_source_kind": normalized_audio_source_kind if effective_has_audio else "none",
        "has_audio": effective_has_audio,
    }

    toggle_effective_map = {
        "video_return_frames": {
            "requested": requested_return_frames,
            "effective": effective_return_frames,
        },
        "video_save_output": {
            "requested": requested_save_output,
            "effective": bool(video_saved),
        },
        "video_save_metadata": {
            "requested": requested_save_metadata,
            "effective": save_metadata_effective,
        },
        "video_trim_to_audio": {
            "requested": requested_trim_to_audio,
            "effective": trim_to_audio_effective,
        },
        "video_pingpong": {
            "requested": requested_pingpong,
            "effective": pingpong_effective,
        },
        "video_interpolation_enabled": {
            "requested": requested_interpolation_toggle,
            "effective": interpolation_applied,
        },
        "video_base_snapshot": {
            "requested": requested_base_snapshot,
            "effective": base_snapshot_saved,
        },
        "video_audio_output": {
            "requested": normalized_audio_source_kind != "none",
            "effective": effective_has_audio,
        },
    }

    return {
        "request_snapshot": _snapshot_clone(requested_snapshot),
        "effective_snapshot": _snapshot_clone(effective_snapshot),
        "toggle_effective_map": _snapshot_clone(toggle_effective_map),
    }


def assemble_video_metadata(
    engine: Any,
    plan: VideoPlan,
    sampler_outcome: Any,
    *,
    elapsed: float,
    frame_count: int,
    task: str,
    extra: Mapping[str, Any] | None = None,
    video_meta: Any = None,
) -> dict[str, Any]:
    engine_dispatch = str(getattr(engine, "engine_id", "unknown"))
    engine_label = engine_dispatch
    if isinstance(extra, Mapping):
        raw_variant = extra.get("wan_engine_variant")
        if isinstance(raw_variant, str) and raw_variant.strip():
            engine_label = raw_variant.strip()

    metadata: dict[str, Any] = {
        "engine": engine_label,
        "task": task,
        "elapsed": round(elapsed, 3),
        "frames": frame_count,
        "fps": plan.fps,
        "width": plan.width,
        "height": plan.height,
        "steps": plan.steps,
        "sampler_in": getattr(sampler_outcome, "sampler_in", None),
        "scheduler_in": getattr(sampler_outcome, "scheduler_in", None),
        "sampler": getattr(sampler_outcome, "sampler_effective", None),
        "scheduler": getattr(sampler_outcome, "scheduler_effective", None),
    }
    if plan.guidance_scale is not None:
        metadata["guidance_scale"] = float(plan.guidance_scale)
    if extra:
        metadata.update(dict(extra))
    if engine_label != engine_dispatch:
        metadata["engine_dispatch"] = engine_dispatch
    if video_meta is not None:
        metadata["video_export"] = video_meta
    return metadata


def build_video_result(
    engine: Any,
    frames: Sequence[Any],
    plan: VideoPlan,
    sampler_outcome: Any,
    *,
    elapsed: float,
    task: str,
    extra: Mapping[str, Any] | None = None,
    video_meta: Any = None,
) -> VideoResult:
    frames_list = frames if isinstance(frames, list) else list(frames)
    metadata = assemble_video_metadata(
        engine,
        plan,
        sampler_outcome,
        elapsed=elapsed,
        frame_count=len(frames_list),
        task=task,
        extra=extra,
        video_meta=video_meta,
    )
    return VideoResult(frames=frames_list, metadata=metadata, video_meta=video_meta)


__all__ = [
    "AudioExportAsset",
    "GeneratedAudioExportPolicy",
    "Ltx2TwoStageGeometry",
    "apply_engine_loras",
    "apply_wan_stage_loras",
    "build_ltx2_two_stage_geometry",
    "build_ltx2_video_plan",
    "build_video_plan",
    "configure_sampler",
    "read_video_interpolation_options",
    "apply_video_interpolation",
    "resolve_video_output_fps",
    "resolve_generated_audio_export_policy",
    "export_video",
    "prepare_base_snapshot_video_options",
    "build_video_request_effective_snapshot",
    "assemble_video_metadata",
    "build_video_result",
]

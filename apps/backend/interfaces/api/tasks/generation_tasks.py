"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared task orchestration helpers for generation endpoints.
Centralizes image encoding, engine-options building, and the common task worker loop (status/progress/result/end/error) so routers stay thin, preserving rich `ProgressEvent` metadata (`message`/`data`) in streamed progress payloads.
Uses the shared inference gate when `CODEX_SINGLE_FLIGHT=1` and always marks tasks finished via `TaskEntry.mark_finished` (stream termination + cleanup).
Inference-gate wait cancellation is mode-agnostic (both `immediate` and `after_current` cancel before start); once running, only immediate mode interrupts in-flight orchestration.
When `CODEX_TRACE_CONTRACT=1`, emits prompt-redacted contract-trace JSONL events (`prompt_hash` only) for prepare/run/progress/result/error/end stages.

Symbols (top-level; keep in sync; no ghosts):
- `encode_images` (function): Encode PIL images to base64 PNG payloads, optionally injecting PNG text metadata.
- `build_engine_options` (function): Build `engine_options` dict from request extras, explicit worker engine key, and options snapshot (TE/VAE selectors, external VAE requirements, no-VAE contracts, explicit checkpoint selectors, Z-Image/Qwen Image/Lens variants, core streaming).
- `resolve_request_smart_flags` (function): Parse/validate per-request smart flags (`smart_offload`/`smart_fallback`/`smart_cache`) as strict booleans.
- `force_runtime_memory_cleanup` (function): Best-effort runtime cleanup used on worker error paths (orchestrator cache + memory manager + CUDA cache).
- `_format_parameters_infotext` (function): Serializes generation `info` dicts into A1111-compatible infotext for PNG `parameters`.
- `_build_png_metadata` (function): Builds PNG text chunks (`parameters` + provenance) for saved/API-encoded images.
- `_PreparedImageExecutionResult` (dataclass): Encoded terminal image result produced by one prepared txt2img/img2img execution.
- `_execute_prepared_image_request` (function): Runs one prepared txt2img/img2img request through the orchestrator and returns the encoded result payload.
- `run_image_task` (function): Run a generic image task worker (txt2img/img2img) using a `prepare(payload)` callback and orchestrator event stream.
- `run_image_automation_task` (function): Run a backend-owned image automation worker around the canonical txt2img/img2img request owners.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import base64
import contextlib
import gc
import io
import json
import logging
import math
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional

from apps.backend.interfaces.api.inference_gate import acquire_inference_gate, release_inference_gate, single_flight_enabled
from apps.backend.interfaces.api.public_errors import (
    build_cancelled_task_error,
    build_missing_result_task_error,
    build_public_task_error,
)
from apps.backend.interfaces.api.task_registry import TaskCancelMode, TaskEntry, unregister_task
from apps.backend.core.strict_values import parse_bool_value
from apps.backend.runtime.diagnostics.contract_trace import error_meta
from apps.backend.runtime.diagnostics.contract_trace import emit_event as emit_contract_trace
from apps.backend.runtime.diagnostics.contract_trace import hash_request_prompt
from apps.backend.runtime.diagnostics.fallback_state import fallback_used as fallback_state_used
from apps.backend.runtime.diagnostics.fallback_state import reset_fallback_state
from apps.backend.runtime.families.lens.config import LENS_ENGINE_ID, LENS_EXTRAS_KEY, LENS_VARIANT_KEY, require_lens_variant
from apps.backend.runtime.families.qwen_image.config import QWEN_IMAGE_VARIANT_KEY, require_qwen_image_variant
from apps.backend.runtime.load_authority import (
    LoadAuthorityStage,
    coordinator_load_permit,
)

logger = get_backend_logger("backend.api.tasks.generation")


def encode_images(images: Any, *, metadata: Optional[Mapping[str, str]] = None) -> list[dict[str, str]]:  # type: ignore[no-untyped-def]
    encoded: list[dict[str, str]] = []
    for img in images or []:
        if img is None:
            continue
        buf = io.BytesIO()
        pnginfo = None
        use_metadata = False
        try:
            from PIL import PngImagePlugin  # type: ignore

            def _add_text(key: object, value: object) -> None:
                nonlocal pnginfo, use_metadata
                if not isinstance(key, str) or not isinstance(value, str):
                    return
                if not value:
                    return
                if pnginfo is None:
                    pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text(key, value)
                use_metadata = True

            info_items = getattr(img, "info", None)
            if isinstance(info_items, dict):
                for key, value in info_items.items():
                    _add_text(key, value)
            if metadata:
                for key, value in metadata.items():
                    _add_text(key, value)
        except Exception:
            pnginfo = None
            use_metadata = False

        img.save(buf, format="PNG", pnginfo=(pnginfo if use_metadata else None))
        encoded.append(
            {
                "format": "png",
                "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            }
        )
    return encoded


def build_engine_options(*, req: Any, engine_key: str, opts_snapshot: Callable[[], Any]) -> dict[str, object]:
    engine_options: dict[str, object] = {}
    extras = getattr(req, "extras", {}) or {}
    engine_id = str(engine_key or "").strip().lower()
    if not engine_id:
        raise RuntimeError("engine_key required for engine option contract resolution.")

    checkpoint_core_only_from_extras = extras.get("checkpoint_core_only")
    if checkpoint_core_only_from_extras is None:
        raise RuntimeError("Missing extras.checkpoint_core_only.")
    if not isinstance(checkpoint_core_only_from_extras, bool):
        raise RuntimeError("extras.checkpoint_core_only must be a boolean.")
    engine_options["checkpoint_core_only"] = checkpoint_core_only_from_extras

    from apps.backend.core.contracts.asset_requirements import contract_for_request

    try:
        asset_contract = contract_for_request(
            engine_id=engine_id,
            checkpoint_core_only=checkpoint_core_only_from_extras,
        )
        uses_vae = bool(asset_contract.uses_vae)
        requires_vae = bool(asset_contract.requires_vae)
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Failed to resolve asset contract for engine '{engine_id}'.") from exc

    te_override = extras.get("text_encoder_override")
    if isinstance(te_override, dict):
        engine_options["text_encoder_override"] = dict(te_override)

    vae_path_from_extras = extras.get("vae_path")
    if isinstance(vae_path_from_extras, str) and vae_path_from_extras.strip():
        engine_options["vae_path"] = vae_path_from_extras.strip()
    vae_source_from_extras = extras.get("vae_source")
    vae_sha_from_extras = extras.get("vae_sha")
    has_vae_sha = isinstance(vae_sha_from_extras, str) and bool(vae_sha_from_extras.strip())
    if uses_vae:
        if vae_source_from_extras is None:
            raise RuntimeError("Missing extras.vae_source.")
        if not isinstance(vae_source_from_extras, str) or not vae_source_from_extras.strip():
            raise RuntimeError("extras.vae_source must be 'built_in' or 'external'.")
        normalized_vae_source = vae_source_from_extras.strip().lower()
        if normalized_vae_source not in {"built_in", "external"}:
            raise RuntimeError("extras.vae_source must be 'built_in' or 'external'.")
        engine_options["vae_source"] = normalized_vae_source
        if normalized_vae_source == "built_in":
            if "vae_path" in engine_options:
                raise RuntimeError("extras.vae_source='built_in' does not allow extras.vae_path.")
            if has_vae_sha:
                raise RuntimeError("extras.vae_source='built_in' does not allow extras.vae_sha.")
            if requires_vae:
                raise RuntimeError(f"Engine '{engine_id}' requires an external VAE; extras.vae_source must be 'external'.")
        else:
            if "vae_path" not in engine_options:
                raise RuntimeError("extras.vae_source='external' requires extras.vae_path.")
    else:
        if vae_source_from_extras is not None:
            raise RuntimeError(f"Engine '{engine_id}' does not use extras.vae_source.")
        if "vae_path" in engine_options:
            raise RuntimeError(f"Engine '{engine_id}' does not use extras.vae_path.")
        if has_vae_sha:
            raise RuntimeError(f"Engine '{engine_id}' does not use extras.vae_sha.")

    model_format_from_extras = extras.get("model_format")
    if model_format_from_extras is None:
        raise RuntimeError("Missing extras.model_format.")
    if not isinstance(model_format_from_extras, str) or not model_format_from_extras.strip():
        raise RuntimeError("extras.model_format must be one of: checkpoint, diffusers, gguf.")
    normalized_model_format = model_format_from_extras.strip().lower()
    if normalized_model_format not in {"checkpoint", "diffusers", "gguf"}:
        raise RuntimeError("extras.model_format must be one of: checkpoint, diffusers, gguf.")
    engine_options["model_format"] = normalized_model_format

    tenc_path_from_extras = extras.get("tenc_path")
    if isinstance(tenc_path_from_extras, str) and tenc_path_from_extras.strip():
        engine_options["tenc_path"] = tenc_path_from_extras.strip()
    elif isinstance(tenc_path_from_extras, list):
        resolved: list[str] = []
        for item in tenc_path_from_extras:
            if isinstance(item, str) and item.strip():
                resolved.append(item.strip())
        if resolved:
            engine_options["tenc_path"] = resolved

    engine_options["tenc_source"] = (
        "external" if ("tenc_path" in engine_options or "text_encoder_override" in engine_options) else "built_in"
    )

    zimage_variant = extras.get("zimage_variant")
    if isinstance(zimage_variant, str) and zimage_variant.strip():
        engine_options["zimage_variant"] = zimage_variant.strip()

    qwen_image_variant = extras.get(QWEN_IMAGE_VARIANT_KEY)
    if isinstance(qwen_image_variant, str) and qwen_image_variant.strip():
        engine_options[QWEN_IMAGE_VARIANT_KEY] = require_qwen_image_variant(
            qwen_image_variant,
            context=f"extras.{QWEN_IMAGE_VARIANT_KEY}",
        )

    for lens_alias_key in ("lens_variant", "lensVariant", "variant", "engine_options"):
        if lens_alias_key in extras:
            raise RuntimeError(
                "Lens variant must use the nested owner extras.lens.variant; "
                f"unsupported alias key: extras.{lens_alias_key}."
            )
    lens_payload = extras.get(LENS_EXTRAS_KEY)
    if lens_payload is not None:
        if engine_id != LENS_ENGINE_ID:
            raise RuntimeError("extras.lens is only valid for engine 'lens'.")
        if not isinstance(lens_payload, Mapping):
            raise RuntimeError("extras.lens must be an object.")
        unexpected_lens_keys = sorted(str(key) for key in lens_payload.keys() if key != "variant")
        if unexpected_lens_keys:
            raise RuntimeError(
                "extras.lens supports only 'variant'; unexpected keys: "
                + ", ".join(unexpected_lens_keys)
                + "."
            )
        engine_options[LENS_VARIANT_KEY] = require_lens_variant(
            lens_payload.get("variant"),
            context="extras.lens.variant",
        )

    # Pass streaming option from settings to engine (no model-part fallbacks).
    snap = opts_snapshot()
    core_streaming_enabled = parse_bool_value(
        getattr(snap, "codex_core_streaming", None),
        field="options.codex_core_streaming",
        default=False,
    )
    if core_streaming_enabled:
        engine_options["core_streaming_enabled"] = True

    return engine_options


def resolve_request_smart_flags(req: Any) -> tuple[bool, bool, bool]:
    values: dict[str, bool] = {}
    for field_name in ("smart_offload", "smart_fallback", "smart_cache"):
        field_value = getattr(req, field_name, False)
        if not isinstance(field_value, bool):
            raise RuntimeError(
                f"Invalid request field '{field_name}': expected boolean, got {type(field_value).__name__}."
            )
        values[field_name] = field_value
    return values["smart_offload"], values["smart_fallback"], values["smart_cache"]


def force_runtime_memory_cleanup(*, reason: str, orch: Any | None = None) -> None:
    cleanup_failures: list[str] = []

    clear_cache = getattr(orch, "clear_cache", None)
    if callable(clear_cache):
        try:
            clear_cache()
        except Exception as exc:
            cleanup_failures.append(f"orchestrator_cache:{exc}")
            logger.warning(
                "Failed to clear orchestrator cache during runtime cleanup (%s): %s",
                reason,
                exc,
                exc_info=False,
            )

    try:
        from apps.backend.runtime.memory import memory_management as memory_state
    except Exception as exc:
        cleanup_failures.append(f"memory_manager_import:{exc}")
        logger.warning(
            "Runtime memory-manager import failed during cleanup (%s): %s",
            reason,
            exc,
            exc_info=False,
        )
    else:
        try:
            with coordinator_load_permit(
                owner="api.tasks.generation.force_runtime_memory_cleanup",
                stage=LoadAuthorityStage.CLEANUP,
            ):
                memory_state.manager.unload_all_models()
        except Exception as exc:
            cleanup_failures.append(f"unload_all_models:{exc}")
            logger.warning(
                "Runtime unload_all_models failed during cleanup (%s): %s",
                reason,
                exc,
                exc_info=False,
            )
        try:
            memory_state.manager.soft_empty_cache(force=True)
        except Exception as exc:
            cleanup_failures.append(f"soft_empty_cache:{exc}")
            logger.warning(
                "Runtime soft_empty_cache failed during cleanup (%s): %s",
                reason,
                exc,
                exc_info=False,
            )

    with contextlib.suppress(Exception):
        gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        cleanup_failures.append(f"torch_cache:{exc}")
        logger.warning(
            "Torch cache cleanup failed during runtime cleanup (%s): %s",
            reason,
            exc,
            exc_info=False,
        )

    if cleanup_failures:
        detail = "; ".join(cleanup_failures[:3])
        if len(cleanup_failures) > 3:
            detail = f"{detail}; ... (+{len(cleanup_failures) - 3} more)"
        raise RuntimeError(f"Runtime memory cleanup failed ({reason}): {detail}")

    logger.info("Runtime memory cleanup completed (%s).", reason)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("+"):
            text = text[1:]
        if text.startswith("-"):
            return None
        if not text.isdigit():
            return None
        try:
            return int(text)
        except Exception:
            return None
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            out = float(text)
        except Exception:
            return None
        if not math.isfinite(out):
            return None
        return out
    return None


def _format_number(value: object) -> str | None:
    integer = _as_int(value)
    if integer is not None:
        return str(integer)
    number = _as_float(value)
    if number is None:
        return None
    return f"{number:g}"


def _format_parameters_infotext(info_obj: object) -> str:
    if not isinstance(info_obj, dict):
        return _as_text(info_obj)

    info = dict(info_obj)
    lines: list[str] = []
    kv_entries: list[str] = []
    seen_keys: set[str] = set()

    def add_kv(label: str, value: object) -> None:
        if not isinstance(label, str):
            return
        key = label.strip()
        if not key:
            return
        text = _as_text(value)
        if not text:
            return
        normalized = key.lower()
        if normalized in seen_keys:
            return
        seen_keys.add(normalized)
        kv_entries.append(f"{key}: {text}")

    prompt = _as_text(info.get("prompt", ""))
    if prompt:
        lines.append(prompt)
    negative_prompt = _as_text(info.get("negative_prompt", ""))
    if negative_prompt:
        lines.append(f"Negative prompt: {negative_prompt}")

    steps = _as_int(info.get("steps"))
    if steps is not None and steps >= 0:
        add_kv("Steps", steps)

    sampler = _as_text(info.get("sampler", ""))
    if sampler:
        add_kv("Sampler", sampler)

    scheduler = _as_text(info.get("scheduler", ""))
    if scheduler:
        add_kv("Schedule type", scheduler)

    cfg_scale = _format_number(info.get("guidance_scale"))
    if cfg_scale is not None:
        add_kv("CFG scale", cfg_scale)

    seed = _as_int(info.get("seed"))
    if seed is not None:
        add_kv("Seed", seed)

    width = _as_int(info.get("width"))
    height = _as_int(info.get("height"))
    if width is not None and height is not None and width > 0 and height > 0:
        add_kv("Size", f"{width}x{height}")

    model_hash = _as_text(info.get("model_hash", ""))
    if model_hash:
        add_kv("Model hash", model_hash)

    model_name = _as_text(info.get("model", ""))
    if model_name:
        add_kv("Model", model_name)

    vae_name = _as_text(info.get("vae", ""))
    if vae_name:
        add_kv("VAE", vae_name)

    clip_skip = _as_int(info.get("clip_skip"))
    if clip_skip is not None and clip_skip >= 0:
        add_kv("Clip skip", clip_skip)

    denoise = _format_number(info.get("denoising_strength"))
    if denoise is not None:
        add_kv("Denoising strength", denoise)

    rng = _as_text(info.get("rng", ""))
    if rng:
        add_kv("RNG", rng)

    extra = info.get("extra")
    if isinstance(extra, dict):
        for raw_key, raw_value in extra.items():
            if raw_value is None:
                continue
            key_text = _as_text(raw_key)
            if not key_text:
                continue
            if isinstance(raw_value, (dict, list)):
                try:
                    value_text = json.dumps(raw_value, ensure_ascii=False)
                except Exception:
                    value_text = _as_text(raw_value)
            else:
                value_text = _as_text(raw_value)
                if "," in value_text or "\n" in value_text or "\r" in value_text:
                    value_text = json.dumps(value_text, ensure_ascii=False)
            if not value_text:
                continue
            add_kv(key_text, value_text)

    if kv_entries:
        lines.append(", ".join(kv_entries))

    return "\n".join(lines).strip()


def _build_png_metadata(info_obj: object, *, generation_provenance: Mapping[str, str]) -> dict[str, str]:
    metadata: dict[str, str] = {}

    parameters = _format_parameters_infotext(info_obj)
    if parameters:
        metadata["parameters"] = parameters

    for key, value in generation_provenance.items():
        key_text = str(key).strip()
        value_text = str(value).strip()
        if not key_text or not value_text:
            continue
        metadata.setdefault(key_text, value_text)
    return metadata


@dataclass(frozen=True)
class _PreparedImageExecutionResult:
    result: dict[str, Any] | None
    cancelled_immediate: bool


def _merge_trace_meta(*parts: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        merged.update(part)
    return merged


def _execute_prepared_image_request(
    *,
    task_id: str,
    mode: str,
    task_type: Any,
    req: Any,
    engine_key: str,
    model_ref: str | None,
    device: str,
    entry: TaskEntry,
    orch: Any,
    live_preview: Any,
    opts_get: Callable[..., Any],
    opts_snapshot: Callable[[], Any],
    generation_provenance: Mapping[str, str],
    save_generated_images: Callable[..., Any],
    push: Callable[[dict[str, Any]], None],
    storage_dtype: object,
    compute_dtype: object,
    fallback_enabled: bool,
    prompt_hash_value: str,
    smart_offload: bool,
    smart_fallback: bool,
    smart_cache: bool,
    trace_meta: Mapping[str, Any] | None = None,
    on_immediate_cancel: Callable[[], None] | None = None,
) -> _PreparedImageExecutionResult:
    preview_cfg = live_preview.build_task_config(opts_get)
    entry.last_preview_id_sent = 0
    engine_options = build_engine_options(req=req, engine_key=engine_key, opts_snapshot=opts_snapshot)

    from apps.backend.core.requests import ProgressEvent, ResultEvent
    from apps.backend.core.state import state as backend_state
    from apps.backend.runtime.memory.smart_offload import smart_runtime_overrides

    def _fallback_used_now() -> bool:
        return bool(fallback_enabled and fallback_state_used())

    cancelled_immediate = False
    result: dict[str, Any] | None = None
    expected_progress_owner_token = f"task:{task_id}"
    backend_state.clear_progress_snapshot()
    with preview_cfg.runtime_overrides(), smart_runtime_overrides(
        smart_offload=smart_offload,
        smart_fallback=smart_fallback,
        smart_cache=smart_cache,
    ):
        for ev in orch.run(
            task_type,
            engine_key,
            req,
            model_ref=model_ref,
            engine_options=engine_options,
        ):
            if entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE:
                if not cancelled_immediate and callable(on_immediate_cancel):
                    on_immediate_cancel()
                cancelled_immediate = True
                continue

            if isinstance(ev, ProgressEvent):
                evt: dict[str, Any] = {
                    "type": "progress",
                    "stage": ev.stage,
                    "percent": ev.percent,
                    "step": ev.step,
                    "total_steps": ev.total_steps,
                    "eta_seconds": ev.eta_seconds,
                }
                if ev.message is not None:
                    evt["message"] = str(ev.message)
                if ev.data:
                    evt["data"] = dict(ev.data)
                live_preview.maybe_attach_to_progress_event(
                    evt,
                    entry,
                    config=preview_cfg,
                    expected_owner_token=expected_progress_owner_token,
                )
                push(evt)
                emit_contract_trace(
                    task_id=task_id,
                    mode=mode,
                    stage=str(ev.stage or "progress"),
                    action="progress",
                    component="orchestrator",
                    device=device,
                    storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                    compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                    strict=True,
                    fallback_enabled=fallback_enabled,
                    fallback_used=_fallback_used_now(),
                    prompt_hash_value=prompt_hash_value,
                    meta=_merge_trace_meta(
                        {
                            "step": ev.step,
                            "total_steps": ev.total_steps,
                            "percent": ev.percent,
                        },
                        trace_meta,
                    ),
                )
                continue

            if not isinstance(ev, ResultEvent):
                continue

            payload_obj = ev.payload or {}
            info_raw = payload_obj.get("info", "{}")
            try:
                info_obj = json.loads(info_raw)
            except Exception:
                info_obj = info_raw
            info_dict = info_obj if isinstance(info_obj, dict) else None
            if isinstance(info_obj, dict):
                for key, value in generation_provenance.items():
                    info_obj.setdefault(key, value)
            png_metadata = _build_png_metadata(info_obj, generation_provenance=generation_provenance)

            if parse_bool_value(
                opts_get("samples_save", True),
                field="options.samples_save",
                default=True,
            ):
                save_generated_images(
                    payload_obj.get("images", []),
                    task=task_type,
                    info=info_dict,
                    metadata=png_metadata,
                )

            result = {
                "images": encode_images(payload_obj.get("images", []), metadata=png_metadata),
                "info": info_obj,
            }
            emit_contract_trace(
                task_id=task_id,
                mode=mode,
                stage="result",
                action="emit",
                component="orchestrator",
                device=device,
                storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                strict=True,
                fallback_enabled=fallback_enabled,
                fallback_used=_fallback_used_now(),
                prompt_hash_value=prompt_hash_value,
                meta=_merge_trace_meta(
                    {"image_count": len(payload_obj.get("images", []) or [])},
                    trace_meta,
                ),
            )

    if not cancelled_immediate and result is None:
        raise RuntimeError("prepared image execution completed without result payload")
    return _PreparedImageExecutionResult(result=result, cancelled_immediate=cancelled_immediate)


def run_image_task(
    *,
    task_id: str,
    payload: dict[str, Any],
    entry: TaskEntry,
    device: str,
    task_type: Any,
    prepare: Callable[[dict[str, Any]], tuple[Any, str, Optional[str]]],
    orch: Any,
    ensure_default_engines_registered: Callable[[], None],
    live_preview: Any,
    opts_get: Callable[..., Any],
    opts_snapshot: Callable[[], Any],
    generation_provenance: Mapping[str, str],
    save_generated_images: Callable[..., Any],
) -> None:
    def push(event: dict[str, Any]) -> None:
        entry.push_event(event)

    push({"type": "status", "stage": "queued"})
    try:
        ensure_default_engines_registered()
        req, engine_key, model_ref = prepare(payload)
    except Exception as err:
        emit_contract_trace(
            task_id=task_id,
            mode=str(getattr(task_type, "value", "unknown")),
            stage="prepare",
            action="error",
            component="router",
            device=device,
            strict=True,
            fallback_enabled=False,
            fallback_used=False,
            prompt_hash_value="",
            meta=error_meta(err),
        )
        entry.error = build_public_task_error(err)
        entry.mark_finished(success=False)
        unregister_task(task_id)
        raise

    mode = str(getattr(task_type, "value", "unknown"))
    prompt_hash_value = hash_request_prompt(req)
    smart_offload, smart_fallback, smart_cache = resolve_request_smart_flags(req)
    fallback_enabled = smart_fallback
    storage_dtype = getattr(req, "core_dtype", None)
    compute_dtype = getattr(req, "core_compute_dtype", None)
    single_flight = single_flight_enabled()

    def _fallback_used_now() -> bool:
        return bool(fallback_enabled and fallback_state_used())

    emit_contract_trace(
        task_id=task_id,
        mode=mode,
        stage="prepare",
        action="ready",
        component="router",
        device=device,
        storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
        compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
        strict=True,
        fallback_enabled=fallback_enabled,
        fallback_used=_fallback_used_now(),
        prompt_hash_value=prompt_hash_value,
        meta={"engine_key": engine_key, "single_flight_enabled": single_flight},
    )

    def worker() -> None:
        acquired = False
        success = False
        reset_fallback_state()
        try:
            if single_flight:
                push({"type": "status", "stage": "waiting_for_inference"})
                emit_contract_trace(
                    task_id=task_id,
                    mode=mode,
                    stage="waiting_for_inference",
                    action="wait",
                    component="inference_gate",
                    device=device,
                    storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                    compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                    strict=True,
                    fallback_enabled=fallback_enabled,
                    fallback_used=_fallback_used_now(),
                    prompt_hash_value=prompt_hash_value,
                    meta={"single_flight_enabled": single_flight},
                )

            acquired = acquire_inference_gate(
                should_cancel=lambda: bool(entry.cancel_requested),
            )
            if not acquired:
                entry.error = build_cancelled_task_error()
                emit_contract_trace(
                    task_id=task_id,
                    mode=mode,
                    stage="inference_gate",
                    action="cancelled",
                    component="inference_gate",
                    device=device,
                    storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                    compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                    strict=True,
                    fallback_enabled=fallback_enabled,
                    fallback_used=_fallback_used_now(),
                    prompt_hash_value=prompt_hash_value,
                    meta={"single_flight_enabled": single_flight},
                )
                return

            push({"type": "status", "stage": "running"})
            from apps.backend.interfaces.api.device_selection import apply_primary_device

            apply_primary_device(device)
            emit_contract_trace(
                task_id=task_id,
                mode=mode,
                stage="running",
                action="start",
                component="orchestrator",
                device=device,
                storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                strict=True,
                fallback_enabled=fallback_enabled,
                fallback_used=_fallback_used_now(),
                prompt_hash_value=prompt_hash_value,
                meta={"single_flight_enabled": single_flight},
            )

            prepared_result = _execute_prepared_image_request(
                task_id=task_id,
                mode=mode,
                task_type=task_type,
                req=req,
                engine_key=engine_key,
                model_ref=model_ref,
                device=device,
                entry=entry,
                orch=orch,
                live_preview=live_preview,
                opts_get=opts_get,
                opts_snapshot=opts_snapshot,
                generation_provenance=generation_provenance,
                save_generated_images=save_generated_images,
                push=push,
                storage_dtype=storage_dtype,
                compute_dtype=compute_dtype,
                fallback_enabled=fallback_enabled,
                prompt_hash_value=prompt_hash_value,
                smart_offload=smart_offload,
                smart_fallback=smart_fallback,
                smart_cache=smart_cache,
                on_immediate_cancel=lambda: setattr(entry, "error", build_cancelled_task_error()),
            )
            if prepared_result.result is not None:
                entry.result = {"status": "completed", "result": prepared_result.result}
            success = not prepared_result.cancelled_immediate
        except Exception as err:  # pragma: no cover - surfaces runtime errors
            engine_execution_error = False
            try:
                from apps.backend.core.exceptions import EngineExecutionError, EngineLoadError

                engine_execution_error = isinstance(err, (EngineExecutionError, EngineLoadError))
            except Exception:
                pass

            if not engine_execution_error:
                try:
                    from apps.backend.runtime.diagnostics.exception_hook import dump_exception as _dump_exc

                    _dump_exc(type(err), err, err.__traceback__, where="generation_image_worker", context={"task_id": task_id})
                except Exception:
                    pass

            cleanup_err: Exception | None = None
            try:
                force_runtime_memory_cleanup(
                    reason=f"{mode}:worker_error",
                    orch=orch,
                )
            except Exception as cleanup_exc:
                cleanup_err = cleanup_exc
                logger.error(
                    "Runtime memory cleanup failed after worker error (task_id=%s mode=%s): %s",
                    task_id,
                    mode,
                    cleanup_exc,
                    exc_info=False,
                )
            if cleanup_err is not None:
                err = RuntimeError(f"{err} [runtime_cleanup_error: {cleanup_err}]")
            entry.error = build_public_task_error(err)
            fallback_used = _fallback_used_now() or (fallback_enabled and ("fallback" in str(err).lower()))
            emit_contract_trace(
                task_id=task_id,
                mode=mode,
                stage="error",
                action="error",
                component="orchestrator",
                device=device,
                storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                strict=True,
                fallback_enabled=fallback_enabled,
                fallback_used=fallback_used,
                prompt_hash_value=prompt_hash_value,
                meta=error_meta(err),
            )
            success = False
        finally:
            if success:
                result_obj = entry.result.get("result") if isinstance(entry.result, dict) else None
                if not isinstance(result_obj, dict):
                    invariant_err = RuntimeError("task completed without result payload")
                    entry.error = build_missing_result_task_error()
                    success = False
                    emit_contract_trace(
                        task_id=task_id,
                        mode=mode,
                        stage="error",
                        action="error",
                        component="task",
                        device=device,
                        storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                        compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                        strict=True,
                        fallback_enabled=fallback_enabled,
                        fallback_used=_fallback_used_now(),
                        prompt_hash_value=prompt_hash_value,
                        meta=error_meta(invariant_err),
                    )
            entry.mark_finished(success=success)
            entry.schedule_cleanup(task_id)
            emit_contract_trace(
                task_id=task_id,
                mode=mode,
                stage="end",
                action="finish",
                component="task",
                device=device,
                storage_dtype=(str(storage_dtype) if storage_dtype is not None else None),
                compute_dtype=(str(compute_dtype) if compute_dtype is not None else None),
                strict=True,
                fallback_enabled=fallback_enabled,
                fallback_used=_fallback_used_now(),
                prompt_hash_value=prompt_hash_value,
                meta={"success": success},
            )
            if acquired:
                try:
                    release_inference_gate()
                except Exception as exc:
                    logger.warning(
                        "inference gate release failed in generation_image_worker (task_id=%s): %s",
                        task_id,
                        exc,
                        exc_info=False,
                    )

    threading.Thread(target=worker, name=f"{task_type.value}-task-{task_id}", daemon=True).start()


def run_image_automation_task(
    *,
    task_id: str,
    request: Any,
    entry: TaskEntry,
    device: str,
    prepare_txt2img: Callable[[dict[str, Any]], tuple[Any, str, Optional[str]]],
    prepare_img2img: Callable[[dict[str, Any]], tuple[Any, str, Optional[str]]],
    orch: Any,
    ensure_default_engines_registered: Callable[[], None],
    live_preview: Any,
    opts_get: Callable[..., Any],
    opts_snapshot: Callable[[], Any],
    generation_provenance: Mapping[str, str],
    save_generated_images: Callable[..., Any],
) -> None:
    from apps.backend.core.engine_interface import TaskType
    from apps.backend.core.requests import ImageAutomationRequest
    from apps.backend.use_cases.image_automation import (
        ImageAutomationImmediateCancel,
        run_image_automation,
    )

    if not isinstance(request, ImageAutomationRequest):
        raise TypeError("run_image_automation_task requires ImageAutomationRequest")

    mode = str(request.mode or "").strip()
    if mode == "txt2img":
        task_type = TaskType.TXT2IMG
        prepare_iteration = prepare_txt2img
    elif mode == "img2img":
        task_type = TaskType.IMG2IMG
        prepare_iteration = prepare_img2img
    else:
        raise ValueError(f"Unsupported image automation mode {request.mode!r}.")

    def push(event: dict[str, Any]) -> None:
        entry.push_event(event)

    push({"type": "status", "stage": "queued"})
    try:
        ensure_default_engines_registered()
    except Exception as err:
        emit_contract_trace(
            task_id=task_id,
            mode=f"{mode}_automation",
            stage="prepare",
            action="error",
            component="router",
            device=device,
            strict=True,
            fallback_enabled=False,
            fallback_used=False,
            prompt_hash_value="",
            meta=error_meta(err),
        )
        entry.error = build_public_task_error(err)
        entry.mark_finished(success=False)
        unregister_task(task_id)
        raise

    prompt_hash_value = hash_request_prompt(SimpleNamespace(**dict(request.template or {})))
    single_flight = single_flight_enabled()

    emit_contract_trace(
        task_id=task_id,
        mode=f"{mode}_automation",
        stage="prepare",
        action="ready",
        component="router",
        device=device,
        strict=True,
        fallback_enabled=False,
        fallback_used=False,
        prompt_hash_value=prompt_hash_value,
        meta={
            "mode": mode,
            "loop_mode": request.loop.mode,
            "loop_count": request.loop.count,
            "single_flight_enabled": single_flight,
        },
    )

    def worker() -> None:
        acquired = False
        success = False
        reset_fallback_state()
        try:
            if single_flight:
                push({"type": "status", "stage": "waiting_for_inference"})
                emit_contract_trace(
                    task_id=task_id,
                    mode=f"{mode}_automation",
                    stage="waiting_for_inference",
                    action="wait",
                    component="inference_gate",
                    device=device,
                    strict=True,
                    fallback_enabled=False,
                    fallback_used=False,
                    prompt_hash_value=prompt_hash_value,
                    meta={"single_flight_enabled": single_flight},
                )

            acquired = acquire_inference_gate(
                should_cancel=lambda: bool(entry.cancel_requested),
            )
            if not acquired:
                entry.error = build_cancelled_task_error()
                emit_contract_trace(
                    task_id=task_id,
                    mode=f"{mode}_automation",
                    stage="inference_gate",
                    action="cancelled",
                    component="inference_gate",
                    device=device,
                    strict=True,
                    fallback_enabled=False,
                    fallback_used=False,
                    prompt_hash_value=prompt_hash_value,
                    meta={"cancel_mode": str(entry.cancel_mode.value)},
                )
                return

            push({"type": "status", "stage": "running"})
            from apps.backend.interfaces.api.device_selection import apply_primary_device

            apply_primary_device(device)
            emit_contract_trace(
                task_id=task_id,
                mode=f"{mode}_automation",
                stage="run",
                action="start",
                component="task",
                device=device,
                strict=True,
                fallback_enabled=False,
                fallback_used=False,
                prompt_hash_value=prompt_hash_value,
                meta={"mode": mode, "loop_mode": request.loop.mode},
            )

            iteration_counter = 0

            def execute_iteration(iteration_payload: dict[str, Any]) -> dict[str, Any]:
                nonlocal iteration_counter
                iteration_counter += 1
                req, engine_key, model_ref = prepare_iteration(iteration_payload)
                smart_offload, smart_fallback, smart_cache = resolve_request_smart_flags(req)
                prepared = _execute_prepared_image_request(
                    task_id=task_id,
                    mode=mode,
                    task_type=task_type,
                    req=req,
                    engine_key=engine_key,
                    model_ref=model_ref,
                    device=device,
                    entry=entry,
                    orch=orch,
                    live_preview=live_preview,
                    opts_get=opts_get,
                    opts_snapshot=opts_snapshot,
                    generation_provenance=generation_provenance,
                    save_generated_images=save_generated_images,
                    push=push,
                    storage_dtype=getattr(req, "core_dtype", None),
                    compute_dtype=getattr(req, "core_compute_dtype", None),
                    fallback_enabled=smart_fallback,
                    prompt_hash_value=hash_request_prompt(req),
                    smart_offload=smart_offload,
                    smart_fallback=smart_fallback,
                    smart_cache=smart_cache,
                    trace_meta={
                        "automation_iteration_index": iteration_counter,
                        "automation_loop_mode": request.loop.mode,
                    },
                )
                if prepared.cancelled_immediate:
                    raise ImageAutomationImmediateCancel("cancelled")
                if not isinstance(prepared.result, dict):
                    raise RuntimeError("automation iteration completed without result payload")
                return prepared.result

            automation_result = run_image_automation(
                request,
                execute_iteration=execute_iteration,
                emit_iteration=push,
                emit_progress=push,
                cancel_snapshot=lambda: (bool(entry.cancel_requested), entry.cancel_mode),
            )
            entry.flush_pending_callbacks()
            final_result = dict(automation_result.last_result)
            gallery_images = entry.recoverable_automation_gallery_images()
            raw_info = final_result.get("info")
            if isinstance(raw_info, dict):
                info_payload = dict(raw_info)
                info_payload["last_iteration_info"] = dict(raw_info)
            else:
                info_payload = {"last_iteration_info": raw_info}
            info_payload["automation_summary"] = dict(automation_result.automation_summary)
            final_result["info"] = info_payload
            entry.result = {"status": "completed", "result": final_result}
            if gallery_images:
                entry.result["automation_gallery_images"] = gallery_images
            success = True
        except ImageAutomationImmediateCancel:
            entry.error = build_cancelled_task_error()
            success = False
        except Exception as err:  # pragma: no cover - runtime surfaces
            cleanup_err: Exception | None = None
            try:
                force_runtime_memory_cleanup(
                    reason=f"{mode}_automation:worker_error",
                    orch=orch,
                )
            except Exception as cleanup_exc:
                cleanup_err = cleanup_exc
                logger.error(
                    "Runtime memory cleanup failed after automation worker error (task_id=%s mode=%s): %s",
                    task_id,
                    mode,
                    cleanup_exc,
                    exc_info=False,
                )
            if cleanup_err is not None:
                err = RuntimeError(f"{err} [runtime_cleanup_error: {cleanup_err}]")
            entry.error = build_public_task_error(err)
            emit_contract_trace(
                task_id=task_id,
                mode=f"{mode}_automation",
                stage="error",
                action="error",
                component="orchestrator",
                device=device,
                strict=True,
                fallback_enabled=False,
                fallback_used=False,
                prompt_hash_value=prompt_hash_value,
                meta=error_meta(err),
            )
            success = False
        finally:
            if success:
                result_obj = entry.result.get("result") if isinstance(entry.result, dict) else None
                if not isinstance(result_obj, dict):
                    invariant_err = RuntimeError("automation task completed without result payload")
                    entry.error = build_missing_result_task_error("automation task")
                    success = False
                    emit_contract_trace(
                        task_id=task_id,
                        mode=f"{mode}_automation",
                        stage="error",
                        action="error",
                        component="task",
                        device=device,
                        strict=True,
                        fallback_enabled=False,
                        fallback_used=False,
                        prompt_hash_value=prompt_hash_value,
                        meta=error_meta(invariant_err),
                    )
            entry.mark_finished(success=success)
            entry.schedule_cleanup(task_id)
            emit_contract_trace(
                task_id=task_id,
                mode=f"{mode}_automation",
                stage="end",
                action="finish",
                component="task",
                device=device,
                strict=True,
                fallback_enabled=False,
                fallback_used=False,
                prompt_hash_value=prompt_hash_value,
                meta={"success": success},
            )
            if acquired:
                try:
                    release_inference_gate()
                except Exception as exc:
                    logger.warning(
                        "inference gate release failed in automation worker (task_id=%s): %s",
                        task_id,
                        exc,
                        exc_info=False,
                    )

    threading.Thread(target=worker, name=f"{mode}-automation-task-{task_id}", daemon=True).start()

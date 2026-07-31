"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Image-to-image use case orchestration and canonical streaming wrapper (init image + optional hires pass).
Builds prompt/sampling plans from `CodexProcessingImg2Img`, routes exact Qwen Image Edit-2511 through its native
conditioning/reference/denoise/decode runtime before classic sampling, prepares classic init-image bundles/latents,
dispatches classic img2img conditioning by executable family, optionally routes SDXL img2img/inpaint through native
SUPIR mode, and optionally performs a hires second pass with family-specific continuation semantics.
Masked img2img (“inpaint”) uses Forge/A1111 “Only masked” semantics and supports optional ADetailer-style multi-region passes for disconnected masks.
Exact-engine SDXL `fooocus_inpaint` and `brushnet` stay on request-scoped family helper seams while the shared masked stage remains generic-only; the canonical sampling stage now enters those exact-engine sessions after LoRA activation instead of mutating only the pre-sampling active snapshot.
The hires pass init is prepared via the global family-dispatched hires-fix stage (`apps/backend/runtime/pipeline_stages/hires_fix.py`).
When configured, the hires second pass parses LoRA tags from the hires prompt/negative prompt pair, inherits base/request LoRAs when the hires
prompt omits them, and resolves explicit hires request overrides by deriving a dedicated `SamplingPlan` for the hires pass, including sampler-specific
ER-SDE options when the hires sampler selects ER-SDE.
When smart offload is enabled, keeps required text-encoder patchers loaded across cond+uncond and unloads them after conditioning.
The wrapper executes sampling + decode + post-cleanup inside the same worker-thread envelope so model residency/offload policies remain single-owner per job.
Worker-thread smart runtime overrides are propagated through `_image_streaming._run_inference_worker(...)`, the wrapper seeds a per-run progress-owner token before pre-sampling VAE encode begins, and decode/cleanup hooks run under a `finally` contract.
Base img2img/inpaint now use the shared proportional denoise-step contract, while hires continuations opt into the internal fixed-step seam explicitly.

Symbols (top-level; keep in sync; no ghosts):
- `_resolve_img2img_variant` (function): Decide which img2img variant to run (classic, Flux Kontext, or Qwen Image Edit).
- `_resolve_classic_img2img_backend` (function): Decide whether classic img2img uses SD-style image conditioning or flow-family zero-conditioning fallback.
- `_resolve_requested_exact_inpaint_mode` (function): Classify whether the current request asks for an exact SDXL inpaint runtime.
- `_validate_exact_inpaint_runtime_state` (function): Fail loud when exact SDXL inpaint reaches runtime without its required masked bundle or alongside SUPIR.
- `_resolve_inpaint_sampling_session_factory` (function): Resolve the optional request-scoped exact-inpaint sampling session factory for the active masked bundle.
- `_install_inpaint_sampling_session_factory` (function): Install/remove the exact-inpaint sampling session factory around non-SUPIR sampling callsites.
- `_resolve_hires_target_dimensions` (function): Resolve the truthful hires target size for the active engine, including zimage `%16` snapping.
- `_conditioning_cache_hit_metadata` (function): Build standardized `GenerationResult.metadata` cache-hit payload for img2img wrappers.
- `_smart_cache_buckets_for_engine` (function): Resolve Smart Cache metric buckets used by each engine family.
- `_smart_cache_bucket_snapshot` (function): Snapshot hit/miss counters for selected Smart Cache buckets.
- `_smart_cache_call_hit` (function): Compute whether a conditioning call was a pure cache hit from bucket deltas.
- `_resolve_hires_execution` (function): Resolves the hires second-pass execution scalars directly from `processing.hires`.
- `_build_hr_prompt_context` (function): Builds the hires prompt context while inheriting base/request LoRAs unless hires tags override them.
- `_run_hires_pass` (function): Runs the hires second pass by reconditioning and resampling from the base samples (init prepared via global hires-fix stage; supports `init_latent` and Kontext `image_latents` continuation modes).
- `_compute_conditioning_payload` (function): Ensure (cond/uncond) conditioning exists for a prompt context.
- `_generate_kontext_img2img` (function): Flux Kontext img2img implementation (init image as `image_latents`, no denoise schedule).
- `_derive_seeds` (function): Normalizes seed/subseed inputs from processing config.
- `_generate_qwen_image_img2img` (function): Orchestrate exact single-image Edit-2511 runtime stages before classic sampling.
- `generate_img2img` (function): Canonical img2img implementation; selects the variant and executes sampling.
- `run_img2img` (function): Canonical img2img mode wrapper (progress polling + decode + result events) used by engines/orchestrator.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import gc
import contextlib
import math
import threading
from dataclasses import replace
from typing import Any, Callable, Iterator, Mapping, Sequence

import logging
import torch

from apps.backend.core.rng import ImageRNG
from apps.backend.runtime.diagnostics.pipeline_debug import log as pipeline_log
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.smart_offload import (
    get_smart_cache_stats,
    smart_cache_enabled,
    smart_offload_enabled,
)
from apps.backend.runtime.memory.smart_offload_invariants import (
    enforce_smart_offload_pre_conditioning_residency,
    enforce_smart_offload_text_encoders_off,
)
from apps.backend.runtime.logging import emit_backend_event
from apps.backend.runtime.processing.conditioners import (
    decode_latent_batch,
    img2img_conditioning,
)
from apps.backend.runtime.processing.datatypes import (
    ConditioningPayload,
    GenerationResult,
    PromptContext,
    SamplingPlan,
)
from apps.backend.runtime.processing.models import CodexProcessingImg2Img
from apps.backend.runtime.pipeline_stages.image_init import prepare_init_bundle
from apps.backend.runtime.pipeline_stages.image_io import latents_to_pil, pil_to_tensor
from apps.backend.runtime.pipeline_stages.hires_fix import (
    prepare_hires_latents_and_conditioning,
    resolve_hires_family_strategy,
    resolve_pipeline_telemetry_context,
    resolve_zimage_hires_pixel_multiple,
)
from apps.backend.runtime.pipeline_stages.masked_img2img import (
    apply_inpaint_full_res_composite,
    compute_mask_connected_component_bboxes,
    prepare_masked_img2img_bundle,
    resolve_mask_enforcer_hooks,
)
from apps.backend.runtime.pipeline_stages.prompt_context import (
    apply_prompt_context,
    build_hires_prompt_context,
    build_prompt_context,
)
from apps.backend.runtime.pipeline_stages.sampling_execute import execute_sampling
from apps.backend.runtime.pipeline_stages.sampling_plan import (
    build_sampling_plan,
    ensure_sampler_and_rng,
    resolve_er_sde_options_for_sampler,
    resolve_sampler_scheduler_override,
)
from apps.backend.runtime.pipeline_stages.scripts import run_process_scripts
from apps.backend.runtime.families.qwen_image.config import (
    QWEN_IMAGE_EDIT_VARIANT,
    QWEN_IMAGE_ENGINE_ID,
    QWEN_IMAGE_PUBLIC_SAMPLER,
    QWEN_IMAGE_PUBLIC_SCHEDULER,
    qwen_image_edit_vae_dimensions,
)
from apps.backend.runtime.families.supir.runtime import run_supir_img2img
from apps.backend.runtime.families.sd.brushnet import apply_brushnet_for_sampling
from apps.backend.runtime.families.sd.fooocus_inpaint import apply_fooocus_inpaint_for_sampling
from apps.backend.runtime.sampling.driver import CodexSampler
from PIL import Image

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

logger = get_backend_logger(__name__)

_KONTEXT_MULTIPLE_OF = 16
_CLASSIC_SD_IMG2IMG_ENGINES = {"sd15", "sd20", "sdxl", "sdxl_refiner", "sd35"}
_CLASSIC_FLOW_IMG2IMG_ENGINES = {"flux1", "flux1_chroma", "zimage", "anima"}
_CLASSIC_FLOW_MASKED_IMG2IMG_ENGINES = {"zimage"}
_GENERIC_POST_SAMPLE_BLEND_INPAINT_MODE = "post_sample_blend"
_FOOOCUS_INPAINT_MODE = "fooocus_inpaint"
_BRUSHNET_INPAINT_MODE = "brushnet"
_EXACT_INPAINT_SAMPLING_SESSION_FACTORY_ATTR = "_codex_exact_inpaint_sampling_session_factory"

# Recommended resolutions from upstream diffusers FluxKontextPipeline.
_PREFERRED_KONTEXT_RESOLUTIONS: list[tuple[int, int]] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def _resolve_img2img_variant(processing: CodexProcessingImg2Img) -> str:
    engine_id = str(getattr(getattr(processing, "sd_model", None), "engine_id", "") or "")
    if engine_id == QWEN_IMAGE_ENGINE_ID:
        return QWEN_IMAGE_ENGINE_ID
    return "kontext" if engine_id == "flux1_kontext" else "classic"


def _resolve_classic_img2img_backend(engine_id: str, *, has_mask: bool) -> str:
    normalized_engine_id = str(engine_id or "").strip()
    if normalized_engine_id == "":
        raise RuntimeError("Classic img2img backend resolution requires a non-empty engine id.")

    if normalized_engine_id in _CLASSIC_SD_IMG2IMG_ENGINES:
        return "sd"
    if normalized_engine_id in _CLASSIC_FLOW_IMG2IMG_ENGINES:
        if has_mask and normalized_engine_id not in _CLASSIC_FLOW_MASKED_IMG2IMG_ENGINES:
            raise NotImplementedError(f"masking is not supported for engine '{normalized_engine_id}' img2img yet")
        return "flow"
    raise NotImplementedError(
        f"Classic img2img backend is not implemented for engine '{normalized_engine_id}'."
    )


def _resolve_shared_inpaint_mode(requested_mode: str | None) -> str:
    normalized = str(requested_mode or "").strip()
    if normalized in {_FOOOCUS_INPAINT_MODE, _BRUSHNET_INPAINT_MODE}:
        return _GENERIC_POST_SAMPLE_BLEND_INPAINT_MODE
    return normalized


def _resolve_requested_exact_inpaint_mode(processing: CodexProcessingImg2Img) -> str | None:
    requested_mode = str(getattr(processing, "inpaint_mode", "") or "").strip()
    if requested_mode in {_FOOOCUS_INPAINT_MODE, _BRUSHNET_INPAINT_MODE}:
        return requested_mode
    return None


def _validate_exact_inpaint_runtime_state(
    *,
    processing: CodexProcessingImg2Img,
    masked_bundle,
) -> str | None:
    requested_mode = _resolve_requested_exact_inpaint_mode(processing)
    if requested_mode is None:
        return None
    if masked_bundle is None:
        raise RuntimeError(
            f"Exact img2img inpaint mode '{requested_mode}' cannot run without a masked img2img bundle."
        )
    if getattr(processing, "supir", None) is not None:
        raise RuntimeError(f"Exact img2img inpaint mode '{requested_mode}' cannot be combined with SUPIR.")
    return requested_mode


def _resolve_inpaint_sampling_session_factory(
    *,
    processing: CodexProcessingImg2Img,
    masked_bundle,
) -> Callable[[], contextlib.AbstractContextManager[object]] | None:
    requested_mode = _validate_exact_inpaint_runtime_state(processing=processing, masked_bundle=masked_bundle)
    if requested_mode is None:
        return None
    if requested_mode == _FOOOCUS_INPAINT_MODE:
        return lambda: apply_fooocus_inpaint_for_sampling(processing=processing, masked_bundle=masked_bundle)
    if requested_mode == _BRUSHNET_INPAINT_MODE:
        return lambda: apply_brushnet_for_sampling(processing=processing, masked_bundle=masked_bundle)
    return None


@contextlib.contextmanager
def _install_inpaint_sampling_session_factory(
    *,
    processing: CodexProcessingImg2Img,
    masked_bundle,
) -> Iterator[None]:
    session_factory = _resolve_inpaint_sampling_session_factory(processing=processing, masked_bundle=masked_bundle)
    if session_factory is None:
        yield
        return
    if hasattr(processing, _EXACT_INPAINT_SAMPLING_SESSION_FACTORY_ATTR):
        raise RuntimeError(
            f"CodexProcessingImg2Img already owns {_EXACT_INPAINT_SAMPLING_SESSION_FACTORY_ATTR}; "
            "exact inpaint session ownership must stay single-owner."
        )
    setattr(processing, _EXACT_INPAINT_SAMPLING_SESSION_FACTORY_ATTR, session_factory)
    try:
        yield
    finally:
        delattr(processing, _EXACT_INPAINT_SAMPLING_SESSION_FACTORY_ATTR)


def _resolve_hires_target_dimensions(
    hi_cfg: Any,
    *,
    engine_id: str,
    base_width: int,
    base_height: int,
) -> tuple[int, int]:
    target_width, target_height = hi_cfg.resolve_target_dimensions(
        base_width=int(base_width),
        base_height=int(base_height),
    )
    zimage_pixel_multiple = resolve_zimage_hires_pixel_multiple(engine_id)
    if zimage_pixel_multiple is None:
        return int(target_width), int(target_height)
    return (
        _floor_multiple(int(target_width), multiple_of=zimage_pixel_multiple),
        _floor_multiple(int(target_height), multiple_of=zimage_pixel_multiple),
    )


def _floor_multiple(value: int, *, multiple_of: int) -> int:
    if multiple_of <= 0:
        raise ValueError("multiple_of must be positive")
    if value <= 0:
        raise ValueError("value must be positive")
    floored = (int(value) // multiple_of) * multiple_of
    return max(multiple_of, floored)


def _pick_preferred_kontext_resolution(image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid init_image size")
    aspect = float(width) / float(height)
    _, best_w, best_h = min((abs(aspect - w / h), w, h) for w, h in _PREFERRED_KONTEXT_RESOLUTIONS)
    return int(best_w), int(best_h)


def _conditioning_cache_hit_metadata(processing: CodexProcessingImg2Img) -> dict[str, object]:
    return {"conditioning_cache_hit": bool(getattr(processing, "_codex_conditioning_cache_hit", False))}


def _merge_generation_metadata(
    processing: CodexProcessingImg2Img,
    result: GenerationResult | None = None,
) -> dict[str, object]:
    metadata = _conditioning_cache_hit_metadata(processing)
    if result is None:
        return metadata
    if not isinstance(result.metadata, dict):
        raise RuntimeError(
            f"img2img pipeline received GenerationResult.metadata as {type(result.metadata).__name__}; expected dict."
        )
    metadata.update(result.metadata)
    return metadata


def _decoded_output_to_images(decoded: Any, *, task_label: str) -> list[Image.Image]:
    if isinstance(decoded, torch.Tensor):
        return latents_to_pil(decoded)
    if isinstance(decoded, list) and all(isinstance(image, Image.Image) for image in decoded):
        return decoded
    raise RuntimeError(
        f"{task_label} returned decoded output as {type(decoded).__name__}; expected torch.Tensor or list[PIL.Image.Image]."
    )


def _emit_pipeline_event(
    processing: CodexProcessingImg2Img,
    event: str,
    *,
    stage: str,
    **fields: object,
) -> None:
    telemetry = resolve_pipeline_telemetry_context(
        processing,
        default_mode="img2img",
        require_mode=True,
    )
    emit_backend_event(
        event,
        logger=logger.name,
        mode=telemetry.mode,
        stage=stage,
        correlation_id=telemetry.correlation_id,
        correlation_source=telemetry.correlation_source,
        task_id=telemetry.task_id,
        **fields,
    )


def _smart_cache_buckets_for_engine(sd_model: Any) -> tuple[str, ...]:
    engine_id = str(getattr(sd_model, "engine_id", "") or "")
    if engine_id == "sdxl":
        return ("sdxl.base.text", "sdxl.base.embed")
    if engine_id == "sdxl_refiner":
        return ("sdxl.refiner.text", "sdxl.refiner.embed")
    if engine_id in {"flux1", "flux1_kontext", "flux1_chroma"}:
        return ("flux.conditioning",)
    if engine_id == "zimage":
        return ("zimage.conditioning",)
    if engine_id == "anima":
        return ("anima.conditioning",)
    return ()


def _smart_cache_bucket_snapshot(bucket_names: Sequence[str]) -> dict[str, tuple[int, int]]:
    stats = get_smart_cache_stats()
    snapshot: dict[str, tuple[int, int]] = {}
    for name in bucket_names:
        bucket = stats.get(str(name), {})
        hits = int(bucket.get("hits", 0) or 0)
        misses = int(bucket.get("misses", 0) or 0)
        snapshot[str(name)] = (hits, misses)
    return snapshot


def _smart_cache_call_hit(bucket_names: Sequence[str], before: Mapping[str, tuple[int, int]]) -> bool:
    if not bucket_names:
        return False
    after = _smart_cache_bucket_snapshot(bucket_names)
    hit_delta = 0
    miss_delta = 0
    for name in bucket_names:
        before_hits, before_misses = before.get(str(name), (0, 0))
        after_hits, after_misses = after.get(str(name), (0, 0))
        hit_delta += max(0, int(after_hits) - int(before_hits))
        miss_delta += max(0, int(after_misses) - int(before_misses))
    return miss_delta == 0 and hit_delta > 0


def _compute_conditioning_payload(
    processing: CodexProcessingImg2Img,
    prompt_context: PromptContext,
    prompts: Sequence[str],
    conditioning: Any,
    unconditional_conditioning: Any,
) -> ConditioningPayload:
    cond = conditioning
    uncond = unconditional_conditioning

    sd_model = getattr(processing, "sd_model", None)
    if sd_model is None or not hasattr(sd_model, "get_learned_conditioning"):
        raise RuntimeError("img2img requires processing.sd_model with get_learned_conditioning")

    enforce_smart_offload_pre_conditioning_residency(sd_model, stage="img2img.conditioning")

    uses_distilled_cfg = bool(getattr(sd_model, "use_distilled_cfg_scale", False))
    smart_flag = getattr(processing, "smart_cache", None)
    cache_enabled = bool(smart_flag) if smart_flag is not None else smart_cache_enabled()
    bucket_names = _smart_cache_buckets_for_engine(sd_model) if cache_enabled else ()
    cache_observations: list[bool] = []
    setattr(processing, "_codex_conditioning_cache_hit", False)

    text_encoder_patchers: list[tuple[str, object]] = []
    needs_conditioning = cond is None or (uncond is None and not uses_distilled_cfg)
    if needs_conditioning and smart_offload_enabled():
        codex_objects = getattr(sd_model, "codex_objects", None)
        text_encoders = getattr(codex_objects, "text_encoders", None) if codex_objects is not None else None
        if isinstance(text_encoders, dict):
            for name, entry in text_encoders.items():
                if entry is None:
                    continue
                try:
                    patcher = entry.patcher
                except AttributeError as exc:
                    raise RuntimeError(
                        "img2img conditioning requires TextEncoderHandle entries "
                        f"(missing .patcher for text_encoders['{name}'])."
                    ) from exc
                if patcher is None:
                    raise RuntimeError(
                        "img2img conditioning requires TextEncoderHandle with non-null patcher "
                        f"for text_encoders['{name}']."
                    )
                text_encoder_patchers.append((str(name), patcher))
            for name, patcher in text_encoder_patchers:
                if memory_management.manager.is_model_loaded(patcher):
                    continue
                pipeline_log(f"[img2img.conditioning] smart_offload: loading text encoder '{name}' patcher for stage")
                memory_management.manager.load_model(patcher)

    try:
        if cond is None:
            texts = list(prompt_context.prompts or [getattr(processing, "prompt", "")])
            before_stats = _smart_cache_bucket_snapshot(bucket_names) if cache_enabled else None
            if hasattr(sd_model, "_prepare_prompt_wrappers"):
                wrapped = sd_model._prepare_prompt_wrappers(texts, processing, is_negative=False)
                cond = sd_model.get_learned_conditioning(wrapped)
            else:
                cond = sd_model.get_learned_conditioning(texts)
            if cache_enabled and before_stats is not None:
                cache_observations.append(_smart_cache_call_hit(bucket_names, before_stats))
            if cond is None:
                raise RuntimeError("Failed to build conditioning for img2img; get_learned_conditioning returned None.")

        if uncond is None and not uses_distilled_cfg:
            negatives = list(prompt_context.negative_prompts or [getattr(processing, "negative_prompt", "")])
            before_stats = _smart_cache_bucket_snapshot(bucket_names) if cache_enabled else None
            if hasattr(sd_model, "_prepare_prompt_wrappers"):
                wrapped = sd_model._prepare_prompt_wrappers(negatives, processing, is_negative=True)
                uncond = sd_model.get_learned_conditioning(wrapped)
            else:
                uncond = sd_model.get_learned_conditioning(negatives)
            if cache_enabled and before_stats is not None:
                cache_observations.append(_smart_cache_call_hit(bucket_names, before_stats))
    finally:
        if smart_offload_enabled():
            if text_encoder_patchers:
                pipeline_log("[img2img.conditioning] smart_offload: unloading text encoders after stage")
            enforce_smart_offload_text_encoders_off(sd_model, stage="img2img.conditioning(post)")

    if not needs_conditioning:
        stage_cache_hit = True
    elif cache_enabled and cache_observations:
        stage_cache_hit = all(cache_observations)
    else:
        stage_cache_hit = False
    setattr(processing, "_codex_conditioning_cache_hit", bool(stage_cache_hit))

    return ConditioningPayload(conditioning=cond, unconditional=uncond)


def _generate_kontext_img2img(
    processing: CodexProcessingImg2Img,
    conditioning: Any,
    unconditional_conditioning: Any,
    prompts: Sequence[str],
    *,
    seeds: Sequence[int] | None,
    subseeds: Sequence[int] | None,
    subseed_strength: float | None,
) -> torch.Tensor:
    if getattr(processing, "init_image", None) is None:
        raise ValueError("img2img requires processing.init_image")

    prompt_context = build_prompt_context(processing, prompts)
    apply_prompt_context(processing, prompt_context)

    overrides = getattr(processing, "override_settings", {})
    auto_resize = True
    if isinstance(overrides, dict) and "kontext_auto_resize" in overrides:
        auto_resize = bool(overrides.get("kontext_auto_resize"))

    init_image: Image.Image = processing.init_image.convert("RGB")
    if auto_resize:
        target_width, target_height = _pick_preferred_kontext_resolution(init_image)
    else:
        target_width, target_height = int(processing.width), int(processing.height)

    target_width = _floor_multiple(target_width, multiple_of=_KONTEXT_MULTIPLE_OF)
    target_height = _floor_multiple(target_height, multiple_of=_KONTEXT_MULTIPLE_OF)

    if init_image.size != (target_width, target_height):
        init_image = init_image.resize((target_width, target_height), _RESAMPLE_LANCZOS)
    processing.init_image = init_image
    processing.width = target_width
    processing.height = target_height

    # Kontext does not use denoise strength / init_latent.
    if hasattr(processing, "denoising_strength"):
        try:
            denoise = float(getattr(processing, "denoising_strength", 0.0) or 0.0)
        except Exception:
            denoise = None
        if denoise not in (None, 0.0, 1.0):
            logger.warning("[kontext] denoising_strength is ignored (got=%s)", denoise)

    seed_list, subseed_list, subseed_value = _derive_seeds(processing)
    if seeds is not None:
        seed_list = list(seeds)
    if subseeds is not None:
        subseed_list = list(subseeds)
    if subseed_strength is not None:
        subseed_value = float(subseed_strength)

    plan = build_sampling_plan(processing, seed_list, subseed_list, subseed_value)
    rng = ensure_sampler_and_rng(processing, plan)

    processing.seeds = list(plan.seeds)
    processing.subseeds = list(plan.subseeds)
    processing.guidance_scale = plan.guidance_scale
    processing.cfg_scale = plan.guidance_scale
    processing.steps = plan.steps
    processing.prepare_prompt_data()

    run_process_scripts(processing)

    bundle = prepare_init_bundle(processing)
    image_latents = bundle.latents
    payload = _compute_conditioning_payload(processing, prompt_context, prompts, conditioning, unconditional_conditioning)
    if not isinstance(payload.conditioning, dict):
        raise TypeError(
            "kontext requires dict conditioning (crossattn/vector) to pass image_latents; "
            f"got {type(payload.conditioning).__name__}"
        )
    payload.conditioning["image_latents"] = image_latents
    if isinstance(payload.unconditional, dict):
        payload.unconditional["image_latents"] = image_latents

    hires_execution = _resolve_hires_execution(processing, emit_plan_event=True) if processing.hires.enabled else None
    _emit_pipeline_event(
        processing,
        "pipeline.stage.complete",
        stage="prepare.complete",
        stage_name="prepare",
        variant="kontext",
        hires_enabled=bool(hires_execution is not None),
        image_latents_shape=tuple(int(dim) for dim in image_latents.shape),
    )

    samples = execute_sampling(
        processing,
        plan,
        payload,
        prompt_context,
        prompt_context.loras,
        rng=rng,
        init_latent=None,
        start_at_step=0,
        allow_txt2img_conditioning_fallback=False,
        img2img_fix_steps=False,
    )
    _emit_pipeline_event(
        processing,
        "pipeline.stage.complete",
        stage="base_sampling.complete",
        stage_name="base_sampling",
        variant="kontext",
        samples_shape=tuple(int(dim) for dim in samples.shape),
    )

    if hires_execution is None:
        return samples

    upscaler_id, target_width, target_height, steps, denoise = hires_execution
    logger.info(
        "[kontext] running hires pass upscaler=%s target=%dx%d steps=%d denoise=%.4f",
        upscaler_id,
        int(target_width),
        int(target_height),
        int(steps),
        float(denoise),
    )
    return _run_hires_pass(
        processing,
        plan,
        samples,
        prompt_context,
        upscaler_id=upscaler_id,
        target_width=int(target_width),
        target_height=int(target_height),
        steps=int(steps),
        denoise=float(denoise),
    )


def _resolve_hires_execution(
    processing: CodexProcessingImg2Img,
    *,
    emit_plan_event: bool,
) -> tuple[str, int, int, int, float]:
    if not processing.hires.enabled:
        raise RuntimeError("Hires execution was requested but processing.hires.enabled is false.")
    model = getattr(processing, "sd_model", None)
    engine_id = str(getattr(model, "engine_id", "") or "").strip()
    if engine_id == "":
        raise RuntimeError("Hires is enabled but processing.sd_model.engine_id is unavailable.")
    hires_strategy = resolve_hires_family_strategy(engine_id)
    setattr(processing, "_codex_hires_strategy", hires_strategy)

    hi_cfg = processing.hires
    upscaler_id = hi_cfg.require_upscaler_id()
    from apps.backend.runtime.vision.upscalers.specs import LATENT_UPSCALE_MODES

    if upscaler_id not in LATENT_UPSCALE_MODES and not upscaler_id.startswith("spandrel:"):
        raise ValueError(
            f"Invalid 'hires.upscaler': {upscaler_id!r}. "
            "Expected a 'latent:*' or 'spandrel:*' upscaler id from GET /api/upscalers."
        )
    target_width, target_height = _resolve_hires_target_dimensions(
        hi_cfg,
        engine_id=engine_id,
        base_width=int(processing.width),
        base_height=int(processing.height),
    )
    steps = hi_cfg.resolve_second_pass_steps(base_steps=int(processing.steps))
    denoise = float(hi_cfg.denoise)
    if emit_plan_event:
        _emit_pipeline_event(
            processing,
            "pipeline.hires.plan",
            stage="hires.plan",
            engine_id=engine_id,
            strategy=hires_strategy,
            upscaler_id=upscaler_id,
            target_width=target_width,
            target_height=target_height,
            steps=int(steps),
            denoise=float(denoise),
        )
    return upscaler_id, int(target_width), int(target_height), int(steps), float(denoise)


def _build_hr_prompt_context(
    processing: CodexProcessingImg2Img, base_context: PromptContext
) -> PromptContext:
    hi_cfg = processing.hires
    return build_hires_prompt_context(
        prompt_seed=hi_cfg.prompt if hi_cfg.prompt else base_context.prompts,
        negative_seed=(
            [hi_cfg.negative_prompt]
            if hi_cfg.negative_prompt
            else list(base_context.negative_prompts)
        ),
        base_context=base_context,
    )


def _run_hires_pass(
    processing: CodexProcessingImg2Img,
    plan: SamplingPlan,
    base_samples: torch.Tensor,
    base_context: PromptContext,
    *,
    upscaler_id: str,
    target_width: int,
    target_height: int,
    steps: int,
    denoise: float,
) -> torch.Tensor:
    hi_cfg = processing.hires

    original = {
        "prompts": processing.prompts,
        "negative_prompts": getattr(processing, "negative_prompts", []),
        "width": processing.width,
        "height": processing.height,
        "guidance_scale": processing.guidance_scale,
        "steps": processing.steps,
        "denoising_strength": getattr(processing, "denoising_strength", 0.75),
        "sampler_name": getattr(processing, "sampler_name", None),
        "scheduler": getattr(processing, "scheduler", None),
        "sampler": getattr(processing, "sampler", None),
    }

    hi_prompt_context = _build_hr_prompt_context(processing, base_context)

    try:
        processing.prompts = hi_prompt_context.prompts
        processing.negative_prompts = hi_prompt_context.negative_prompts
        processing.width = target_width
        processing.height = target_height
        effective_target_width = int(processing.width)
        effective_target_height = int(processing.height)
        processing.guidance_scale = float(hi_cfg.cfg or processing.guidance_scale)
        processing.cfg_scale = processing.guidance_scale
        processing.steps = int(steps)
        processing.denoising_strength = denoise
        hires_runtime_plan = replace(
            plan,
            steps=int(processing.steps),
            guidance_scale=float(processing.guidance_scale),
            er_sde=None,
        )
        hires_sampler, hires_scheduler = resolve_sampler_scheduler_override(
            base_sampler=str(hires_runtime_plan.sampler_name or ""),
            base_scheduler=str(hires_runtime_plan.scheduler_name or ""),
            sampler_override=getattr(hi_cfg, "sampler_name", None),
            scheduler_override=getattr(hi_cfg, "scheduler", None),
        )
        hires_runtime_plan = replace(
            hires_runtime_plan,
            er_sde=resolve_er_sde_options_for_sampler(processing, hires_sampler),
        )
        processing.sampler_name = hires_sampler
        processing.scheduler = hires_scheduler
        processing.sampler = CodexSampler(processing.sd_model, algorithm=hires_sampler)
        processing._codex_effective_hires_sampling = {
            "sampler": hires_sampler,
            "scheduler": hires_scheduler,
            "steps": int(steps),
            "denoise": float(denoise),
            "width": int(effective_target_width),
            "height": int(effective_target_height),
        }
        processing.prepare_prompt_data()
        engine_id = str(getattr(getattr(processing, "sd_model", None), "engine_id", "") or "unknown")
        hires_strategy = str(getattr(processing, "_codex_hires_strategy", "unknown") or "unknown")
        _emit_pipeline_event(
            processing,
            "pipeline.hires.transition",
            stage="hires.transition",
            engine_id=engine_id,
            strategy=hires_strategy,
            upscaler_id=upscaler_id,
            target_width=effective_target_width,
            target_height=effective_target_height,
            steps=steps,
            denoise=denoise,
            sampler=hires_sampler,
            scheduler=hires_scheduler,
            base_samples_shape=tuple(int(dim) for dim in base_samples.shape),
        )

        hires_inputs = prepare_hires_latents_and_conditioning(
            processing,
            base_samples=base_samples,
            base_decoded=None,
            target_width=int(effective_target_width),
            target_height=int(effective_target_height),
            upscaler_id=upscaler_id,
            tile=getattr(hi_cfg, "tile", None),
        )
        latents = hires_inputs.latents
        image_conditioning = hires_inputs.image_conditioning
        continuation_mode = hires_inputs.continuation_mode

        hires_settings = hires_runtime_plan.noise_settings
        rng = ImageRNG(
            (latents.shape[1], latents.shape[2], latents.shape[3]),
            hires_runtime_plan.seeds,
            subseeds=hires_runtime_plan.subseeds,
            subseed_strength=hires_runtime_plan.subseed_strength,
            seed_resize_from_h=getattr(processing, "seed_resize_from_h", 0),
            seed_resize_from_w=getattr(processing, "seed_resize_from_w", 0),
            settings=hires_settings,
        )
        noise = rng.next().to(latents)

        start_index = 0
        denoise_strength: float | None = float(denoise)
        init_latent: torch.Tensor | None = latents
        sampling_image_conditioning = image_conditioning
        allow_txt2img_conditioning_fallback = True

        hr_plan = replace(
            hires_runtime_plan,
            sampler_name=hires_sampler,
            scheduler_name=hires_scheduler,
            steps=int(processing.steps),
            guidance_scale=float(processing.guidance_scale),
        )
        hires_payload = _compute_conditioning_payload(
            processing,
            hi_prompt_context,
            hi_prompt_context.prompts,
            conditioning=None,
            unconditional_conditioning=None,
        )
        if continuation_mode == "image_latents":
            if not isinstance(hires_payload.conditioning, dict):
                raise TypeError(
                    "Hires Kontext continuation requires dict conditioning to inject image_latents; "
                    f"got {type(hires_payload.conditioning).__name__}."
                )
            hires_payload.conditioning["image_latents"] = latents
            if isinstance(hires_payload.unconditional, dict):
                hires_payload.unconditional["image_latents"] = latents
            init_latent = None
            sampling_image_conditioning = None
            denoise_strength = None
            allow_txt2img_conditioning_fallback = False
            if not (math.isclose(float(denoise), 0.0) or math.isclose(float(denoise), 1.0)):
                logger.warning(
                    "[kontext] hires continuation ignores denoise schedule semantics (configured denoise=%.4f).",
                    float(denoise),
                )
        image_conditioning_shape = (
            tuple(int(dim) for dim in sampling_image_conditioning.shape)
            if isinstance(sampling_image_conditioning, torch.Tensor)
            else None
        )
        _emit_pipeline_event(
            processing,
            "pipeline.hires.inputs_ready",
            stage="hires.inputs_ready",
            engine_id=engine_id,
            strategy=hires_strategy,
            continuation_mode=continuation_mode,
            latents_shape=tuple(int(dim) for dim in latents.shape),
            init_latent_present=init_latent is not None,
            image_conditioning_shape=image_conditioning_shape,
            start_at_step=int(start_index),
            total_steps=int(processing.steps),
            allow_txt2img_conditioning_fallback=allow_txt2img_conditioning_fallback,
        )
        logger.info(
            "[hires] img2img continuation_mode=%s init_latent=%s image_conditioning=%s start_at_step=%d",
            continuation_mode,
            "set" if init_latent is not None else "none",
            "set" if isinstance(sampling_image_conditioning, torch.Tensor) else "none",
            int(start_index),
        )

        samples = execute_sampling(
            processing,
            hr_plan,
            hires_payload,
            hi_prompt_context,
            hi_prompt_context.loras,
            rng=rng,
            noise=noise,
            image_conditioning=sampling_image_conditioning,
            init_latent=init_latent,
            start_at_step=start_index,
            denoise_strength=denoise_strength,
            allow_txt2img_conditioning_fallback=allow_txt2img_conditioning_fallback,
            img2img_fix_steps=True,
        )
        _emit_pipeline_event(
            processing,
            "pipeline.stage.complete",
            stage="hires_sampling.complete",
            stage_name="hires_sampling",
            samples_shape=tuple(int(dim) for dim in samples.shape),
        )
        return samples
    finally:
        processing.prompts = original["prompts"]
        processing.negative_prompts = original["negative_prompts"]
        processing.width = original["width"]
        processing.height = original["height"]
        processing.guidance_scale = original["guidance_scale"]
        processing.cfg_scale = processing.guidance_scale
        processing.steps = original["steps"]
        processing.denoising_strength = original["denoising_strength"]
        processing.sampler_name = original["sampler_name"]
        processing.scheduler = original["scheduler"]
        processing.sampler = original["sampler"]
        processing.prepare_prompt_data()


def _derive_seeds(processing: CodexProcessingImg2Img) -> tuple[list[int], list[int], float]:
    seeds = list(getattr(processing, "seeds", []) or [])
    if not seeds:
        seeds = [int(getattr(processing, "seed", -1) or -1)]
    subseeds = list(getattr(processing, "subseeds", []) or [])
    strength = float(getattr(processing, "subseed_strength", 0.0) or 0.0)
    return seeds, subseeds, strength


def _generate_qwen_image_img2img(
    processing: CodexProcessingImg2Img,
    conditioning: object,
    unconditional_conditioning: object,
    prompts: Sequence[str],
    *,
    seeds: Sequence[int] | None,
    subseeds: Sequence[int] | None,
    subseed_strength: float | None,
) -> GenerationResult:
    if conditioning is not None or unconditional_conditioning is not None:
        raise RuntimeError(
            "Qwen Image prepare stage does not accept precomputed classic conditioning; "
            "multimodal conditioning is owned by QwenImageRuntime."
        )
    if processing.batch_total != 1:
        raise NotImplementedError("Qwen Image Edit-2511 supports exactly one image per img2img request.")
    if len(prompts) != 1:
        raise NotImplementedError("Qwen Image Edit-2511 supports exactly one prompt per img2img request.")
    if processing.has_mask() or getattr(processing, "inpaint_mode", None) is not None:
        raise NotImplementedError("Qwen Image Edit-2511 does not support masks or inpaint modes.")
    if bool(getattr(getattr(processing, "hires", None), "enabled", False)):
        raise NotImplementedError("Qwen Image Edit-2511 does not support HiRes.")
    if getattr(processing, "supir", None) is not None:
        raise NotImplementedError("Qwen Image Edit-2511 does not support SUPIR.")
    if getattr(processing, "ip_adapter", None) is not None:
        raise NotImplementedError("Qwen Image Edit-2511 does not support IP-Adapter.")
    if getattr(processing, "scripts", None) is not None or tuple(getattr(processing, "script_args", ()) or ()):
        raise NotImplementedError("Qwen Image Edit-2511 does not support script-owned img2img mutations.")
    if list(getattr(processing, "init_images", ()) or ()):
        raise NotImplementedError("Qwen Image Edit-2511 supports one canonical init_image, not init_images.")

    init_image = getattr(processing, "init_image", None)
    if not isinstance(init_image, Image.Image):
        raise TypeError(
            "Qwen Image Edit-2511 requires processing.init_image as PIL.Image.Image; "
            f"got {type(init_image).__name__}."
        )

    denoise_strength = float(getattr(processing, "denoising_strength", 0.0) or 0.0)
    if not math.isclose(denoise_strength, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise NotImplementedError(
            "Qwen Image Edit-2511 owns full native edit denoising and requires denoising_strength=1.0."
        )
    if getattr(processing, "image_cfg_scale", None) is not None:
        raise NotImplementedError("Qwen Image Edit-2511 does not support image_cfg_scale.")

    expected_width, expected_height = qwen_image_edit_vae_dimensions(*init_image.size)
    output_width = int(getattr(processing, "width", 0) or 0)
    output_height = int(getattr(processing, "height", 0) or 0)
    if (output_width, output_height) != (expected_width, expected_height):
        raise RuntimeError(
            "Qwen Image Edit-2511 output dimensions must be derived from the canonical VAE reference geometry; "
            f"got {output_width}x{output_height}, expected {expected_width}x{expected_height} "
            f"for source {init_image.width}x{init_image.height}."
        )

    prompt_context = build_prompt_context(processing, prompts)
    if prompt_context.loras:
        raise NotImplementedError("Qwen Image Edit-2511 does not support LoRA selections or prompt LoRA tags.")
    if prompt_context.clip_skip is not None:
        raise NotImplementedError("Qwen Image Edit-2511 does not support clip_skip.")
    apply_prompt_context(processing, prompt_context)

    seed_list, subseed_list, subseed_value = _derive_seeds(processing)
    if seeds is not None:
        seed_list = list(seeds)
    if subseeds is not None:
        subseed_list = list(subseeds)
    if subseed_strength is not None:
        subseed_value = float(subseed_strength)

    plan = build_sampling_plan(processing, seed_list, subseed_list, subseed_value)
    if str(plan.sampler_name or "").strip().lower() != QWEN_IMAGE_PUBLIC_SAMPLER:
        raise RuntimeError(
            f"Qwen Image Edit-2511 requires sampler {QWEN_IMAGE_PUBLIC_SAMPLER!r}; "
            f"got {plan.sampler_name!r}."
        )
    if str(plan.scheduler_name or "").strip().lower() != QWEN_IMAGE_PUBLIC_SCHEDULER:
        raise RuntimeError(
            f"Qwen Image Edit-2511 requires scheduler {QWEN_IMAGE_PUBLIC_SCHEDULER!r}; "
            f"got {plan.scheduler_name!r}."
        )
    if plan.steps < 2:
        raise RuntimeError(f"Qwen Image Edit-2511 requires at least 2 denoise steps; got {plan.steps}.")
    if len(plan.seeds) != 1:
        raise RuntimeError(f"Qwen Image Edit-2511 requires exactly one seed; got {len(plan.seeds)}.")
    if not math.isclose(float(plan.subseed_strength), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise NotImplementedError("Qwen Image Edit-2511 does not support subseed variation.")
    if any(int(value) != -1 for value in plan.subseeds):
        raise NotImplementedError("Qwen Image Edit-2511 does not support explicit subseeds.")

    processing.seeds = list(plan.seeds)
    processing.subseeds = list(plan.subseeds)
    processing.guidance_scale = plan.guidance_scale
    processing.cfg_scale = plan.guidance_scale
    processing.steps = plan.steps
    processing.prepare_prompt_data()

    progress_owner_token = str(getattr(processing, "_codex_progress_owner_token", "") or "").strip()
    if not progress_owner_token:
        raise RuntimeError("Qwen Image Edit-2511 prepare stage requires a non-empty progress owner token.")

    engine = getattr(processing, "sd_model", None)
    if str(getattr(engine, "engine_id", "") or "") != QWEN_IMAGE_ENGINE_ID:
        raise RuntimeError("Qwen Image Edit-2511 prepare stage requires the canonical qwen_image engine.")
    try:
        runtime = engine.qwen_image_runtime
    except Exception as exc:
        raise RuntimeError(f"Qwen Image Edit-2511 prepare stage could not resolve the loaded family runtime: {exc}") from exc
    if str(getattr(runtime, "variant", "") or "") != QWEN_IMAGE_EDIT_VARIANT:
        raise RuntimeError(
            "Qwen Image Edit-2511 prepare stage resolved an incompatible runtime variant: "
            f"{getattr(runtime, 'variant', None)!r}."
        )

    _emit_pipeline_event(
        processing,
        "pipeline.stage.complete",
        stage="prepare.complete",
        stage_name="prepare",
        engine_id=QWEN_IMAGE_ENGINE_ID,
        variant=QWEN_IMAGE_EDIT_VARIANT,
        hires_enabled=False,
        has_mask=False,
        output_width=output_width,
        output_height=output_height,
        steps=int(plan.steps),
        true_cfg_scale=float(plan.guidance_scale),
    )

    qwen_conditioning = runtime.encode_conditioning(
        prompt=processing.primary_prompt,
        negative_prompt=processing.primary_negative_prompt,
        image=init_image,
    )
    _emit_pipeline_event(
        processing,
        "pipeline.stage.complete",
        stage="conditioning.complete",
        stage_name="conditioning",
        condition_width=int(qwen_conditioning.condition_width),
        condition_height=int(qwen_conditioning.condition_height),
        positive_tokens=int(qwen_conditioning.positive_embeddings.shape[1]),
        negative_tokens=int(qwen_conditioning.negative_embeddings.shape[1]),
    )

    reference = runtime.encode_reference(image=init_image)
    _emit_pipeline_event(
        processing,
        "pipeline.stage.complete",
        stage="encode.complete",
        stage_name="encode",
        reference_shape=tuple(int(dimension) for dimension in reference.packed_latents.shape),
        reference_width=int(reference.grid.width),
        reference_height=int(reference.grid.height),
    )

    denoised = runtime.denoise(
        conditioning=qwen_conditioning,
        reference=reference,
        width=output_width,
        height=output_height,
        steps=plan.steps,
        seed=plan.seeds[0],
        progress_owner_token=progress_owner_token,
        true_cfg_scale=plan.guidance_scale,
        noise_settings=plan.noise_settings,
    )
    _emit_pipeline_event(
        processing,
        "pipeline.stage.complete",
        stage="base_sampling.complete",
        stage_name="base_sampling",
        samples_shape=tuple(int(dimension) for dimension in denoised.packed_latents.shape),
        seed=int(denoised.seed),
        steps=int(denoised.steps),
    )

    decoded = runtime.decode(denoised=denoised)
    _emit_pipeline_event(
        processing,
        "pipeline.stage.complete",
        stage="decode.complete",
        stage_name="decode",
        decoded_shape=tuple(int(dimension) for dimension in decoded.shape),
    )

    result_metadata = _conditioning_cache_hit_metadata(processing)
    result_metadata["qwen_image_variant"] = QWEN_IMAGE_EDIT_VARIANT
    return GenerationResult(
        samples=denoised.packed_latents,
        decoded=decoded,
        metadata=result_metadata,
    )


def generate_img2img(
    processing,
    conditioning,
    unconditional_conditioning,
    prompts: Sequence[str],
    *,
    seeds: Sequence[int] | None = None,
    subseeds: Sequence[int] | None = None,
    subseed_strength: float | None = None,
) -> GenerationResult:
    if not isinstance(processing, CodexProcessingImg2Img):
        raise TypeError("generate_img2img expects CodexProcessingImg2Img")
    setattr(processing, "_codex_pipeline_mode", "img2img")
    resolve_pipeline_telemetry_context(
        processing,
        default_mode="img2img",
        require_mode=True,
    )
    setattr(processing, "_codex_conditioning_cache_hit", False)
    variant = _resolve_img2img_variant(processing)
    engine_id = str(getattr(getattr(processing, "sd_model", None), "engine_id", "") or "")
    _emit_pipeline_event(
        processing,
        "pipeline.run.start",
        stage="run.start",
        variant=variant,
        engine_id=engine_id or "unknown",
    )

    if engine_id == QWEN_IMAGE_ENGINE_ID:
        result = _generate_qwen_image_img2img(
            processing,
            conditioning,
            unconditional_conditioning,
            prompts,
            seeds=seeds,
            subseeds=subseeds,
            subseed_strength=subseed_strength,
        )
        _emit_pipeline_event(
            processing,
            "pipeline.run.complete",
            stage="run.complete",
            variant=variant,
            hires_enabled=False,
            samples_shape=tuple(int(dimension) for dimension in result.samples.shape),
            decoded_shape=tuple(int(dimension) for dimension in result.decoded.shape),
        )
        return result

    if variant == "kontext":
        if processing.has_mask():
            raise NotImplementedError("masking is not supported for flux1_kontext img2img yet")
        samples = _generate_kontext_img2img(
            processing,
            conditioning,
            unconditional_conditioning,
            prompts,
            seeds=seeds,
            subseeds=subseeds,
            subseed_strength=subseed_strength,
        )
        _emit_pipeline_event(
            processing,
            "pipeline.run.complete",
            stage="run.complete",
            variant=variant,
            hires_enabled=bool(getattr(getattr(processing, "hires", None), "enabled", False)),
            samples_shape=tuple(int(dim) for dim in samples.shape),
        )
        return GenerationResult(samples=samples, decoded=None, metadata=_conditioning_cache_hit_metadata(processing))

    classic_backend = _resolve_classic_img2img_backend(engine_id, has_mask=processing.has_mask())
    prompt_context = build_prompt_context(processing, prompts)
    apply_prompt_context(processing, prompt_context)
    if processing.supir is not None and prompt_context.loras:
        raise NotImplementedError("SUPIR mode cannot be combined with LoRA selections in tranche 1")

    seed_list, subseed_list, subseed_value = _derive_seeds(processing)
    if seeds is not None:
        seed_list = list(seeds)
    if subseeds is not None:
        subseed_list = list(subseeds)
    if subseed_strength is not None:
        subseed_value = float(subseed_strength)

    plan = build_sampling_plan(processing, seed_list, subseed_list, subseed_value)

    rng = ensure_sampler_and_rng(processing, plan)

    processing.seeds = list(plan.seeds)
    processing.subseeds = list(plan.subseeds)
    processing.guidance_scale = plan.guidance_scale
    processing.cfg_scale = plan.guidance_scale
    processing.steps = plan.steps
    processing.prepare_prompt_data()

    run_process_scripts(processing)
    hires_execution = _resolve_hires_execution(processing, emit_plan_event=True) if processing.hires.enabled else None
    if processing.supir is not None and hires_execution is not None:
        raise NotImplementedError("SUPIR mode is not supported with HiRes")

    payload = _compute_conditioning_payload(
        processing,
        prompt_context,
        prompts,
        conditioning,
        unconditional_conditioning,
    )
    _emit_pipeline_event(
        processing,
        "pipeline.stage.complete",
        stage="prepare.complete",
        stage_name="prepare",
        hires_enabled=bool(hires_execution is not None),
        has_mask=bool(processing.has_mask()),
    )
    pre_denoiser_hook = None
    post_denoiser_hook = None
    post_step_hook = None
    post_sample_hook = None
    full_res_plan = None
    source_tensor = None
    if processing.has_mask():
        if bool(getattr(getattr(processing, "hires", None), "enabled", False)):
            raise NotImplementedError("HiRes is not supported for masked img2img yet")
        if processing.supir is not None:
            inpainting_fill = int(getattr(processing, "inpainting_fill", 0) or 0)
            if inpainting_fill in {2, 3}:
                raise NotImplementedError(
                    "SUPIR mode does not support masked img2img fill modes 'latent noise' or 'latent nothing' in tranche 1"
                )
        include_masked_image_conditioning = classic_backend == "sd"

        mask_region_split = bool(getattr(processing, "mask_region_split", False))
        if mask_region_split:
            invert_value = int(getattr(processing, "inpainting_mask_invert", 0) or 0)
            if invert_value != 0:
                raise ValueError("mask_region_split is not supported with inpainting_mask_invert=1")
            batch_total = int(getattr(processing, "batch_total", 1) or 1)
            if batch_total != 1:
                raise NotImplementedError("mask_region_split is only supported for batch_total=1")

            raw_mask = getattr(processing, "mask", None)
            if raw_mask is None:
                raise ValueError("mask_region_split requires a non-null mask")

            component_bboxes = compute_mask_connected_component_bboxes(
                raw_mask,
                round_mask=bool(getattr(processing, "mask_round", True)),
            )
            if len(component_bboxes) > 1:
                processing.update_extra_param("Mask regions", len(component_bboxes))
                current = processing.init_image.convert("RGB")

                def _region_mask_from_bbox(
                    *,
                    source_mask: Image.Image,
                    bbox: tuple[int, int, int, int],
                ) -> Image.Image:
                    x1, y1, x2, y2 = bbox
                    source = source_mask.convert("RGBA")
                    region = Image.new("RGBA", source.size, (0, 0, 0, 0))
                    region_crop = source.crop((x1, y1, x2, y2))
                    region.alpha_composite(region_crop, dest=(x1, y1))
                    return region

                last_samples: torch.Tensor | None = None
                for index, bbox in enumerate(component_bboxes):
                    processing.update_extra_param("Mask region", f"{index + 1}/{len(component_bboxes)}")
                    processing.init_image = current
                    processing.set_mask(_region_mask_from_bbox(source_mask=raw_mask, bbox=bbox))

                    requested_inpaint_mode = getattr(processing, "inpaint_mode", None)
                    shared_inpaint_mode = _resolve_shared_inpaint_mode(requested_inpaint_mode)
                    masked_bundle, enforcer = prepare_masked_img2img_bundle(
                        processing,
                        plan,
                        enforce_mode=shared_inpaint_mode,
                        include_image_conditioning=include_masked_image_conditioning,
                    )
                    if masked_bundle.full_res is None:
                        raise RuntimeError("mask_region_split requires an inpaint crop plan (internal bug)")

                    processing.init_latent = masked_bundle.init_latent
                    processing.image_conditioning = masked_bundle.image_conditioning
                    hooks = resolve_mask_enforcer_hooks(enforcer, enforce_mode=shared_inpaint_mode)
                    pre_denoiser_hook = hooks.pre_denoiser
                    post_denoiser_hook = hooks.post_denoiser
                    post_step_hook = hooks.post_step
                    post_sample_hook = hooks.post_sample

                    init_latent = masked_bundle.init_latent
                    image_conditioning = masked_bundle.image_conditioning
                    source_tensor = masked_bundle.init_tensor
                    pass_rng = ensure_sampler_and_rng(processing, plan)
                    noise = pass_rng.next().to(init_latent)
                    denoise_strength = float(getattr(processing, "denoising_strength", 0.5) or 0.5)
                    _validate_exact_inpaint_runtime_state(processing=processing, masked_bundle=masked_bundle)

                    runtime_result = None
                    if processing.supir is not None:
                        runtime_result = run_supir_img2img(
                            processing,
                            plan=plan,
                            payload=payload,
                            prompt_context=prompt_context,
                            rng=pass_rng,
                            noise=noise,
                            source_tensor=source_tensor,
                            pre_denoiser_hook=pre_denoiser_hook,
                            post_denoiser_hook=post_denoiser_hook,
                            post_step_hook=post_step_hook,
                            post_sample_hook=post_sample_hook,
                        )
                        samples = runtime_result.samples
                    else:
                        with _install_inpaint_sampling_session_factory(
                            processing=processing,
                            masked_bundle=masked_bundle,
                        ):
                            samples = execute_sampling(
                                processing,
                                plan,
                                payload,
                                prompt_context,
                                prompt_context.loras,
                                rng=pass_rng,
                                noise=noise,
                                image_conditioning=image_conditioning,
                                init_latent=init_latent,
                                start_at_step=0,
                                denoise_strength=denoise_strength,
                                img2img_fix_steps=False,
                                pre_denoiser_hook=pre_denoiser_hook,
                                post_denoiser_hook=post_denoiser_hook,
                                post_step_hook=post_step_hook,
                                post_sample_hook=post_sample_hook,
                            )

                    last_samples = samples
                    gc.collect()
                    memory_management.manager.soft_empty_cache(force=True)
                    if runtime_result is not None and runtime_result.decoded is not None:
                        patch_images = _decoded_output_to_images(
                            runtime_result.decoded,
                            task_label="img2img.mask_region_split",
                        )
                    else:
                        decoded = decode_latent_batch(
                            processing.sd_model,
                            samples,
                            target_device=memory_management.manager.cpu_device,
                            stage="img2img.mask_region_split.decode(pre)",
                        )
                        patch_images = latents_to_pil(decoded)
                    composited = apply_inpaint_full_res_composite(patch_images, plan=masked_bundle.full_res)
                    if len(composited) != 1:
                        raise RuntimeError(
                            "mask_region_split requires single-image batches (internal bug)"
                        )
                    current = composited[0]

                if last_samples is None:
                    raise RuntimeError("mask_region_split produced no passes (internal bug)")
                last_metadata = _merge_generation_metadata(processing, runtime_result)
                _emit_pipeline_event(
                    processing,
                    "pipeline.stage.complete",
                    stage="base_sampling.complete",
                    stage_name="base_sampling",
                    region_split=True,
                    region_count=int(len(component_bboxes)),
                    samples_shape=tuple(int(dim) for dim in last_samples.shape),
                )
                _emit_pipeline_event(
                    processing,
                    "pipeline.run.complete",
                    stage="run.complete",
                    hires_enabled=False,
                    region_split=True,
                    samples_shape=tuple(int(dim) for dim in last_samples.shape),
                )
                return GenerationResult(
                    samples=last_samples,
                    decoded=[current],
                    metadata=last_metadata,
                )

        requested_inpaint_mode = getattr(processing, "inpaint_mode", None)
        shared_inpaint_mode = _resolve_shared_inpaint_mode(requested_inpaint_mode)
        masked_bundle, enforcer = prepare_masked_img2img_bundle(
            processing,
            plan,
            enforce_mode=shared_inpaint_mode,
            include_image_conditioning=include_masked_image_conditioning,
        )
        processing.init_latent = masked_bundle.init_latent
        processing.image_conditioning = masked_bundle.image_conditioning
        full_res_plan = masked_bundle.full_res
        hooks = resolve_mask_enforcer_hooks(enforcer, enforce_mode=shared_inpaint_mode)
        pre_denoiser_hook = hooks.pre_denoiser
        post_denoiser_hook = hooks.post_denoiser
        post_step_hook = hooks.post_step
        post_sample_hook = hooks.post_sample
        init_latent = masked_bundle.init_latent
        image_conditioning = masked_bundle.image_conditioning
        source_tensor = masked_bundle.init_tensor
    else:
        bundle = prepare_init_bundle(processing)
        processing.init_latent = bundle.latents

        image_conditioning = None
        if classic_backend == "sd":
            image_conditioning = img2img_conditioning(
                processing.sd_model,
                bundle.tensor,
                bundle.latents,
                image_mask=bundle.mask,
                round_mask=getattr(processing, "mask_round", True),
            )
        processing.image_conditioning = image_conditioning
        init_latent = bundle.latents
        source_tensor = bundle.tensor

    if not torch.is_tensor(source_tensor):
        raise RuntimeError("img2img pipeline failed to resolve source_tensor for the canonical img2img owner.")

    noise = rng.next().to(init_latent)
    denoise_value = float(getattr(processing, "denoising_strength", 0.5) or 0.5)
    start_step = 0
    denoise_strength = denoise_value

    exact_masked_bundle = masked_bundle if processing.has_mask() else None
    _validate_exact_inpaint_runtime_state(processing=processing, masked_bundle=exact_masked_bundle)

    runtime_result = None
    if processing.supir is not None:
        runtime_result = run_supir_img2img(
            processing,
            plan=plan,
            payload=payload,
            prompt_context=prompt_context,
            rng=rng,
            noise=noise,
            source_tensor=source_tensor,
            pre_denoiser_hook=pre_denoiser_hook,
            post_denoiser_hook=post_denoiser_hook,
            post_step_hook=post_step_hook,
            post_sample_hook=post_sample_hook,
        )
        samples = runtime_result.samples
    else:
        with _install_inpaint_sampling_session_factory(
            processing=processing,
            masked_bundle=exact_masked_bundle,
        ):
            samples = execute_sampling(
                processing,
                plan,
                payload,
                prompt_context,
                prompt_context.loras,
                rng=rng,
                noise=noise,
                image_conditioning=image_conditioning,
                init_latent=init_latent,
                start_at_step=start_step,
                denoise_strength=denoise_strength,
                img2img_fix_steps=False,
                pre_denoiser_hook=pre_denoiser_hook,
                post_denoiser_hook=post_denoiser_hook,
                post_step_hook=post_step_hook,
                post_sample_hook=post_sample_hook,
            )
    result_metadata = _merge_generation_metadata(processing, runtime_result)
    _emit_pipeline_event(
        processing,
        "pipeline.stage.complete",
        stage="base_sampling.complete",
        stage_name="base_sampling",
        has_mask=bool(processing.has_mask()),
        samples_shape=tuple(int(dim) for dim in samples.shape),
    )

    if full_res_plan is not None:
        gc.collect()
        memory_management.manager.soft_empty_cache(force=True)
        if runtime_result is not None and runtime_result.decoded is not None:
            images = _decoded_output_to_images(runtime_result.decoded, task_label="img2img.full_res")
        else:
            decoded = decode_latent_batch(
                processing.sd_model,
                samples,
                target_device=memory_management.manager.cpu_device,
                stage="img2img.full_res.decode(pre)",
            )
            images = latents_to_pil(decoded)
        composited = apply_inpaint_full_res_composite(images, plan=full_res_plan)
        _emit_pipeline_event(
            processing,
            "pipeline.run.complete",
            stage="run.complete",
            hires_enabled=False,
            full_res_masked=True,
            samples_shape=tuple(int(dim) for dim in samples.shape),
        )
        return GenerationResult(
            samples=samples,
            decoded=composited,
            metadata=result_metadata,
            decode_engine=getattr(runtime_result, "decode_engine", None) if runtime_result is not None else None,
        )

    if hires_execution is None:
        _emit_pipeline_event(
            processing,
            "pipeline.run.complete",
            stage="run.complete",
            hires_enabled=False,
            samples_shape=tuple(int(dim) for dim in samples.shape),
        )
        return GenerationResult(
            samples=samples,
            decoded=getattr(runtime_result, "decoded", None) if runtime_result is not None else None,
            metadata=result_metadata,
            decode_engine=getattr(runtime_result, "decode_engine", None) if runtime_result is not None else None,
        )

    upscaler_id, target_width, target_height, steps, denoise = hires_execution
    hires_samples = _run_hires_pass(
        processing,
        plan,
        samples,
        prompt_context,
        upscaler_id=upscaler_id,
        target_width=int(target_width),
        target_height=int(target_height),
        steps=int(steps),
        denoise=float(denoise),
    )
    _emit_pipeline_event(
        processing,
        "pipeline.run.complete",
        stage="run.complete",
        hires_enabled=True,
        samples_shape=tuple(int(dim) for dim in hires_samples.shape),
    )

    return GenerationResult(
        samples=hires_samples,
        decoded=None,
        metadata=result_metadata,
    )


def run_img2img(
    *,
    engine: Any,
    request: Any,
) -> Iterator["InferenceEvent"]:
    """Run img2img as a canonical event stream.

    This wrapper owns the mode-level concerns (seed defaults, progress polling, decode + result packaging).
    Engines should delegate here rather than implementing per-mode pipelines.
    """

    import json

    from apps.backend.core.requests import Img2ImgRequest, ResultEvent
    from apps.backend.engines.util.adapters import build_img2img_processing
    from apps.backend.runtime.text_processing import (
        clear_last_extra_generation_params,
        snapshot_last_extra_generation_params,
    )

    from ._image_streaming import (
        _build_common_info,
        _decode_generation_output,
        _ImageProgressProfile,
        _iter_image_progress_events,
        _resolve_seed_plan,
        _resolve_progress_owner_token,
        _run_inference_worker,
        _seed_progress_owner_token,
    )

    if not isinstance(request, Img2ImgRequest):
        raise TypeError("run_img2img expects Img2ImgRequest")

    engine.ensure_loaded()

    proc = build_img2img_processing(request)
    proc.sd_model = engine
    task_context = str(threading.current_thread().name or "").strip() or "unknown-thread"
    setattr(proc, "_codex_pipeline_mode", "img2img")
    task_id: str | None = None
    marker = "-task-"
    if marker in task_context:
        candidate = task_context.split(marker, 1)[1].strip()
        if candidate:
            task_id = candidate
    if task_id is not None:
        setattr(proc, "_codex_task_id", task_id)
        setattr(proc, "_codex_correlation_id", task_id)
        setattr(proc, "_codex_hires_correlation_id", task_id)
        setattr(proc, "_codex_correlation_source", "task_id")
    progress_owner_token = _resolve_progress_owner_token(task_context=task_context, task_id=task_id)
    setattr(proc, "_codex_progress_owner_token", progress_owner_token)
    _seed_progress_owner_token(progress_owner_token=progress_owner_token)

    base_seed, seeds, subseeds, subseed_strength = _resolve_seed_plan(
        seed=getattr(request, "seed", None),
        batch_total=proc.batch_total,
    )
    proc.seed = base_seed
    proc.seeds = list(seeds)
    proc.subseed = -1
    proc.subseeds = list(subseeds)

    prompts = list(getattr(proc, "prompts", []) or []) or [proc.prompt]
    smart_flags = {
        "smart_offload": bool(getattr(proc, "smart_offload", False)),
        "smart_fallback": bool(getattr(proc, "smart_fallback", False)),
        "smart_cache": bool(getattr(proc, "smart_cache", False)),
    }

    def _generate() -> dict[str, object]:
        import json
        import time

        cleanup_targets: list[Any] = [engine]
        sampling_start = 0.0
        sampling_end = 0.0
        active_decode_engine: Any = engine

        try:
            clear_last_extra_generation_params()
            sampling_start = time.perf_counter()
            output = generate_img2img(
                proc,
                conditioning=None,
                unconditional_conditioning=None,
                prompts=prompts,
                seeds=seeds,
                subseeds=subseeds,
                subseed_strength=subseed_strength,
            )
            sampling_end = time.perf_counter()

            output_decode_engine = getattr(output, "decode_engine", None)
            active_decode_engine = output_decode_engine if output_decode_engine is not None else getattr(proc, "sd_model", None)
            if active_decode_engine is None:
                active_decode_engine = engine
            if active_decode_engine is not None and not any(existing is active_decode_engine for existing in cleanup_targets):
                cleanup_targets.append(active_decode_engine)

            images, decode_ms = _decode_generation_output(engine=active_decode_engine, output=output, task_label="img2img")

            all_seeds = list(getattr(proc, "all_seeds", []) or []) or list(seeds)
            seed_value = int(all_seeds[0]) if all_seeds else int(base_seed)

            extra_params: dict[str, object] = {}
            try:
                extra_params.update(snapshot_last_extra_generation_params())
                extra_params.update(getattr(proc, "extra_generation_params", {}) or {})
            except Exception:  # noqa: BLE001
                extra_params = getattr(proc, "extra_generation_params", {}) or {}

            timings: dict[str, float] = {
                "sampling_ms": max(0.0, (sampling_end - sampling_start) * 1000.0),
                "decode_ms": float(decode_ms),
            }

            mode_info: dict[str, object] = {
                "denoise_strength": float(getattr(proc, "denoising_strength", 0.0) or 0.0),
            }
            if bool(getattr(getattr(proc, "hires", None), "enabled", False)):
                try:
                    mode_info["hires"] = getattr(proc, "hires", None).as_dict()
                except Exception:  # noqa: BLE001
                    pass
                effective_hires_sampling = getattr(proc, "_codex_effective_hires_sampling", None)
                if isinstance(effective_hires_sampling, dict) and effective_hires_sampling:
                    mode_info["effective_hires_sampling"] = dict(effective_hires_sampling)

            info = _build_common_info(
                engine_id=engine.engine_id,
                task="img2img",
                proc=proc,
                seed=seed_value,
                all_seeds=all_seeds,
                extra_params=extra_params,
                timings_ms=timings,
                mode_info=mode_info,
            )
            return {"images": images, "info": json.dumps(info)}
        finally:
            processing_model = getattr(proc, "sd_model", None)
            if processing_model is not None and not any(existing is processing_model for existing in cleanup_targets):
                cleanup_targets.append(processing_model)
            for cleanup_target in cleanup_targets:
                post_cleanup = getattr(cleanup_target, "_post_txt2img_cleanup", None)
                if callable(post_cleanup):
                    post_cleanup()

    done, outcome = _run_inference_worker(
        name=f"{engine.engine_id}-img2img-worker",
        fn=_generate,
        runtime_overrides=smart_flags,
    )

    yield from _iter_image_progress_events(
        done=done,
        outcome=outcome,
        progress_owner_token=progress_owner_token,
        profile=_ImageProgressProfile(
            encode_weight=10.0,
            sampling_weight=80.0,
            decode_weight=10.0,
        ),
    )

    if outcome.error is not None:
        raise outcome.error

    payload = outcome.output
    if not isinstance(payload, dict):
        raise RuntimeError(
            "img2img worker returned invalid payload type; expected dict with 'images' and 'info'. "
            f"Got {type(payload).__name__}."
        )
    images = payload.get("images")
    info = payload.get("info")
    if not isinstance(images, list):
        raise RuntimeError("img2img worker payload field 'images' must be list.")
    if not isinstance(info, str):
        raise RuntimeError("img2img worker payload field 'info' must be JSON string.")
    yield ResultEvent(payload={"images": images, "info": info})

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Stage-based txt2img pipeline orchestrator (sampling + first-pass swap-model + hi-res + optional refiner).
Coordinates prompt parsing, conditioning, sampling execution, generic first-pass model-swap execution, and optional refiner stages while producing images and metadata, with fail-loud conditioning guards that avoid embedding raw prompt text in raised errors.
Conditioning smart-cache entries are keyed by model/load identity plus wrapped prompt metadata and stored detached on CPU to avoid stale hits and cross-request GPU pinning.
The hires stage delegates family-dispatched init preparation and continuation semantics to the global hires-fix workflow stage (`apps/backend/runtime/pipeline_stages/hires_fix.py`).
When configured, the hires second pass parses LoRA-only prompt tags, inherits base/request LoRAs when the hires prompt omits them, resolves explicit hires request overrides by deriving a dedicated `SamplingPlan` for the hires pass, and calls the shared sampler with the internal fixed-step img2img continuation flag only from this hires seam.
Sampler-specific hires plan options such as ER-SDE stay attached to the derived hires plan rather than leaking through flat processing fields.
First-pass base decode before hires is now upscaler-aware (`latent:*` skips decode; pixel upscalers decode).
Exact Z-Image L2P runs stay inside this canonical txt2img runner with pre-conditioning unsupported-surface guards, effective prompt/optional-negative validation, pixel RGB sampling, canonical `BackendState` sampling-progress publication, and `GenerationResult.decoded` propagation that avoids VAE decode fallback.
Exact Lens runs stay parked but now have the same canonical runner-hook seam: the runner guards unsupported Lens txt2img surfaces before generic prompt application/conditioning/RNG construction and expects decoded RGB hook output.
When smart offload is enabled, keeps required text-encoder patchers loaded across cond+uncond and unloads them after conditioning.

Symbols (top-level; keep in sync; no ghosts):
- `PrepareState` (dataclass): Prepared per-run state (resolved engine + plans + prompt context) used across stages.
- `SamplingOutput` (dataclass): Sampling result container (latents/images + optional swap-model boundary bridge) passed between pipeline stages.
- `Txt2ImgPipelineRunner` (class): Main orchestrator; owns the stage pipeline (conditioning/sampling/first-pass swap/hires/refiner) and calls the runtime helpers
  (hires stage uses the global hires-fix stage for family-dispatched upscaling and continuation mode routing; integrates smart cache + pipeline tracing).
- `GenerationResult` (dataclass): Standardized output container for the runner (`samples` + optional `decoded`).
"""
# // tags: txt2img, pipeline, sdxl, hires, refiner

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import logging
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass, replace
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from apps.backend.core import devices
from apps.backend.core.state import state as backend_state
from apps.backend.core.rng import ImageRNG
from apps.backend.infra.config import args as backend_args
from apps.backend.runtime.diagnostics.pipeline_debug import log as pipeline_log, pipeline_trace
from apps.backend.engines.common.tensor_tree import detach_to_cpu, move_to_device
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.memory.smart_offload import (
    smart_cache_enabled,
    smart_offload_enabled,
    record_smart_cache_hit,
    record_smart_cache_miss,
)
from apps.backend.runtime.memory.smart_offload_invariants import (
    enforce_smart_offload_pre_conditioning_residency,
    enforce_smart_offload_text_encoders_off,
)
from apps.backend.runtime.families.lens.resolution import require_lens_dimensions
from apps.backend.runtime.families.lens.sampler import validate_lens_decoded_output
from apps.backend.runtime.processing.datatypes import (
    ConditioningPayload,
    GenerationResult,
    PromptContext,
    SamplingPlan,
)
from apps.backend.runtime.processing.conditioners import encode_image_batch
from apps.backend.runtime.processing.models import CodexProcessingTxt2Img
from apps.backend.runtime.pipeline_stages.image_io import maybe_decode_for_hr
from apps.backend.runtime.pipeline_stages.hires_fix import (
    prepare_hires_latents_and_conditioning,
    resolve_hires_family_strategy,
    resolve_pipeline_telemetry_context,
)
from apps.backend.runtime.pipeline_stages.prompt_context import (
    apply_prompt_context,
    build_hires_prompt_context,
    build_prompt_context,
)
from apps.backend.runtime.pipeline_stages.sampling_execute import execute_sampling, execute_sampling_result
from apps.backend.runtime.pipeline_stages.sampling_plan import (
    build_sampling_plan,
    ensure_sampler,
    ensure_sampler_and_rng,
    resolve_er_sde_options_for_sampler,
    resolve_sampler_scheduler_override,
)
from apps.backend.patchers.lora_apply import selection_hash_for_request
from apps.backend.runtime.logging import emit_backend_event
from apps.backend.runtime.pipeline_stages.scripts import collect_lora_selections, run_process_scripts
from apps.backend.runtime.sampling.driver import CodexSampler, SamplingBoundaryState
from apps.backend.runtime.text_processing.extra_nets import ExtraNetsParseError, parse_prompts
from apps.backend.core.engine_loader import EngineLoadOptions, load_engine as _load_engine
from apps.backend.use_cases.txt2img_pipeline.refiner import (
    GlobalRefinerStage,
    GlobalSwapModelStage,
    HiresRefinerStage,
    RefinerStage,
    resolve_swap_pointer_remaining_steps,
)


@dataclass(slots=True)
class PrepareState:
    """State captured after the preparation stage."""

    prompt_context: PromptContext
    hires_prompt_context: PromptContext | None
    sampling_plan: SamplingPlan
    rng: ImageRNG | None
    payload: ConditioningPayload
    init_latents: torch.Tensor | None
    init_decoded: torch.Tensor | None
    cond: object | None = None
    uncond: object | None = None


@dataclass(slots=True)
class SamplingOutput:
    """Result of executing a sampling stage."""

    samples: torch.Tensor
    decoded: torch.Tensor | None
    boundary_state: SamplingBoundaryState | None = None


class Txt2ImgPipelineRunner:
    """Orchestrates txt2img generation through well-defined stages."""

    def __init__(self) -> None:
        self._logger = get_backend_logger(__name__)
        # SDXL conditioning cache (prompt + dims → (cond, uncond))
        # shared across runs for this runner instance.
        self._conditioning_cache: dict[tuple, tuple[object, object | None]] = {}

    @classmethod
    def _freeze_cache_value(cls, value: Any, *, _depth: int = 0) -> object:
        """Convert arbitrary values to a deterministic, hashable cache token."""
        if _depth > 6:
            return ("depth_limit", type(value).__name__)
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            if type(value) is str:
                return value
            attrs = cls._freeze_known_attrs(value, _depth=_depth + 1)
            if attrs:
                return (
                    "string_subclass",
                    f"{type(value).__module__}.{type(value).__qualname__}",
                    str(value),
                    attrs,
                )
            return ("string_subclass", f"{type(value).__module__}.{type(value).__qualname__}", str(value))
        if isinstance(value, torch.Tensor):
            return (
                "tensor",
                tuple(int(dim) for dim in value.shape),
                str(value.dtype),
                str(value.device),
            )
        if isinstance(value, Mapping):
            items: list[tuple[str, object]] = []
            for key in sorted(value.keys(), key=lambda item: str(item)):
                items.append((str(key), cls._freeze_cache_value(value[key], _depth=_depth + 1)))
            return ("mapping", tuple(items))
        if isinstance(value, tuple):
            return ("tuple", tuple(cls._freeze_cache_value(item, _depth=_depth + 1) for item in value))
        if isinstance(value, list):
            attrs = cls._freeze_known_attrs(value, _depth=_depth + 1)
            return (
                "list",
                tuple(cls._freeze_cache_value(item, _depth=_depth + 1) for item in value),
                attrs,
            )
        if isinstance(value, set):
            return (
                "set",
                tuple(
                    sorted(
                        (cls._freeze_cache_value(item, _depth=_depth + 1) for item in value),
                        key=repr,
                    )
                ),
            )
        if is_dataclass(value):
            try:
                return (
                    "dataclass",
                    f"{type(value).__module__}.{type(value).__qualname__}",
                    cls._freeze_cache_value(asdict(value), _depth=_depth + 1),
                )
            except Exception:
                return ("dataclass_repr", f"{type(value).__module__}.{type(value).__qualname__}", repr(value))

        attrs = cls._freeze_known_attrs(value, _depth=_depth + 1)
        if attrs:
            return (
                "object",
                f"{type(value).__module__}.{type(value).__qualname__}",
                attrs,
                str(value),
            )
        return ("repr", f"{type(value).__module__}.{type(value).__qualname__}", repr(value))

    @classmethod
    def _freeze_known_attrs(cls, value: Any, *, _depth: int) -> tuple[tuple[str, object], ...]:
        attr_names = (
            "is_negative_prompt",
            "smart_cache",
            "distilled_cfg_scale",
            "cfg_scale",
            "width",
            "height",
            "target_width",
            "target_height",
            "crop_left",
            "crop_top",
            "tenc_source",
            "tenc_path",
            "vae_source",
            "vae_path",
            "core_streaming_enabled",
            "extras",
            "label",
            "family",
        )
        attrs: list[tuple[str, object]] = []
        for attr_name in attr_names:
            if not hasattr(value, attr_name):
                continue
            try:
                attr_value = getattr(value, attr_name)
            except Exception:
                continue
            if callable(attr_value):
                continue
            attrs.append((attr_name, cls._freeze_cache_value(attr_value, _depth=_depth + 1)))
        return tuple(sorted(attrs))

    @classmethod
    def _conditioning_model_identity(cls, sd_model: Any) -> tuple[object, ...]:
        model_ref = getattr(sd_model, "model_ref", None)
        if model_ref in (None, ""):
            model_ref = getattr(sd_model, "_current_model_ref", None)
        load_options = getattr(sd_model, "_load_options", None)
        lora_hash = getattr(sd_model, "current_lora_hash", None)

        codex_objects = None
        try:
            codex_objects = getattr(sd_model, "codex_objects", None)
        except Exception:
            codex_objects = None

        denoiser_obj = getattr(codex_objects, "denoiser", None) if codex_objects is not None else None
        denoiser_target = getattr(denoiser_obj, "patcher", denoiser_obj) if denoiser_obj is not None else None
        vae_obj = getattr(codex_objects, "vae", None) if codex_objects is not None else None
        vae_patcher = getattr(vae_obj, "patcher", vae_obj) if vae_obj is not None else None

        text_encoder_ids: tuple[tuple[str, int], ...] = ()
        if codex_objects is not None:
            text_encoders = getattr(codex_objects, "text_encoders", None)
            if isinstance(text_encoders, dict):
                pairs: list[tuple[str, int]] = []
                for name, entry in text_encoders.items():
                    if entry is None:
                        continue
                    try:
                        patcher = entry.patcher
                    except AttributeError as exc:
                        raise RuntimeError(
                            "txt2img conditioning identity requires TextEncoderHandle entries "
                            f"(missing .patcher for text_encoders['{name}'])."
                        ) from exc
                    if patcher is None:
                        raise RuntimeError(
                            "txt2img conditioning identity requires TextEncoderHandle with non-null patcher "
                            f"for text_encoders['{name}']."
                        )
                    pairs.append((str(name), int(id(patcher))))
                text_encoder_ids = tuple(sorted(pairs))

        return (
            f"{type(sd_model).__module__}.{type(sd_model).__qualname__}",
            str(getattr(sd_model, "engine_id", "") or ""),
            cls._freeze_cache_value(model_ref),
            cls._freeze_cache_value(load_options),
            cls._freeze_cache_value(lora_hash),
            int(id(denoiser_target)),
            int(id(vae_patcher)),
            text_encoder_ids,
        )

    @staticmethod
    def _conditioning_target_device() -> torch.device | str:
        get_device = getattr(memory_management.manager, "get_device", None)
        if callable(get_device):
            try:
                return get_device(DeviceRole.TEXT_ENCODER)
            except Exception:
                pass
        return "cpu"

    @staticmethod
    def _to_device_dtype_if_needed(
        tensor: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if tensor.device == device and tensor.dtype == dtype:
            return tensor
        return tensor.to(device=device, dtype=dtype)

    @staticmethod
    def _log_tensor_stats(label: str, tensor: torch.Tensor | None) -> None:
        logger = get_backend_logger(__name__)
        if tensor is None:
            logger.info("[sampling] %s: <none>", label)
            return
        with torch.no_grad():
            data = tensor.detach()
            try:
                stats_tensor = data if data.dtype in {torch.float32, torch.float64} else data.float()
            except Exception:
                stats_tensor = data
            mean = float(stats_tensor.mean().item())
            std = float(stats_tensor.std(unbiased=False).item())
            min_value = float(stats_tensor.min().item())
            max_value = float(stats_tensor.max().item())
        logger.info(
            "[sampling] %s: shape=%s dtype=%s device=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
            label,
            tuple(data.shape),
            data.dtype,
            data.device,
            min_value,
            max_value,
            mean,
            std,
        )

    def _compute_conditioning(self, processing: CodexProcessingTxt2Img, context: PromptContext):
        """Build cond/uncond using the engine's SDXL-aware helpers after prompt parsing and dimension overrides.

        Smart Cache is resolved per job: when ``processing.smart_cache`` is present it
        takes precedence over the global options snapshot, so callers can flip cache
        on/off for individual requests without touching Quicksettings.
        """
        sd_model = getattr(processing, "sd_model", None)
        if sd_model is None or not hasattr(sd_model, "get_learned_conditioning"):
            return None, None

        try:
            merged_loras = collect_lora_selections(getattr(context, "loras", ()) or ())
            setattr(sd_model, "current_lora_hash", selection_hash_for_request(merged_loras))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "txt2img conditioning requires deterministic LoRA selection identity before cache lookup."
            ) from exc

        setattr(processing, "_codex_conditioning_cache_hit", False)

        prompts = list(context.prompts or [getattr(processing, "prompt", "")])
        negative_prompts = list(context.negative_prompts or [getattr(processing, "negative_prompt", "")])
        uses_distilled_cfg = bool(getattr(sd_model, "use_distilled_cfg_scale", False))
        if hasattr(sd_model, "_prepare_prompt_wrappers"):
            prompts_payload = sd_model._prepare_prompt_wrappers(prompts, processing, is_negative=False)
            negative_payload = (
                None
                if uses_distilled_cfg
                else sd_model._prepare_prompt_wrappers(negative_prompts, processing, is_negative=True)
            )
        else:
            prompts_payload = prompts
            negative_payload = None if uses_distilled_cfg else negative_prompts

        smart_flag = getattr(processing, "smart_cache", None)
        cache_enabled = bool(smart_flag) if smart_flag is not None else smart_cache_enabled()
        key = None
        if cache_enabled:
            try:
                key = (
                    self._conditioning_model_identity(sd_model),
                    self._freeze_cache_value(prompts_payload),
                    self._freeze_cache_value(negative_payload),
                    context.clip_skip,
                )
            except Exception:
                key = None

        if cache_enabled and key is not None:
            cached = self._conditioning_cache.get(key)
            if cached is not None:
                record_smart_cache_hit("sdxl.runner.conditioning")
                setattr(processing, "_codex_conditioning_cache_hit", True)
                enforce_smart_offload_text_encoders_off(sd_model, stage="txt2img.conditioning(cache-hit)")
                target_device = self._conditioning_target_device()
                return move_to_device(cached, device=target_device)
            record_smart_cache_miss("sdxl.runner.conditioning")

        enforce_smart_offload_pre_conditioning_residency(sd_model, stage="txt2img.conditioning")

        text_encoder_patchers: list[tuple[str, object]] = []
        if smart_offload_enabled():
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
                            "txt2img conditioning requires TextEncoderHandle entries "
                            f"(missing .patcher for text_encoders['{name}'])."
                        ) from exc
                    if patcher is None:
                        raise RuntimeError(
                            "txt2img conditioning requires TextEncoderHandle with non-null patcher "
                            f"for text_encoders['{name}']."
                        )
                    text_encoder_patchers.append((str(name), patcher))
                for name, patcher in text_encoder_patchers:
                    if memory_management.manager.is_model_loaded(patcher):
                        continue
                    pipeline_log(f"[conditioning] smart_offload: loading text encoder '{name}' patcher for stage")
                    memory_management.manager.load_model(patcher)

        try:
            # Preserve spatial metadata via engine helper when available
            cond = sd_model.get_learned_conditioning(prompts_payload)
            if uses_distilled_cfg:
                uncond = None
            else:
                uncond = sd_model.get_learned_conditioning(negative_payload)

            # If uncond comes back zero for a non-empty negative prompt, fail fast instead of sampling with CFG degenerate
            non_empty_negative = any(str(p or "").strip() for p in negative_prompts)
            if non_empty_negative:
                uncond_cross = uncond.get("crossattn") if isinstance(uncond, dict) else None
                if isinstance(uncond_cross, torch.Tensor):
                    norm_uncond = float(uncond_cross.abs().sum().item())
                    if norm_uncond < 1e-6:
                        negative_count = sum(1 for item in negative_prompts if str(item or "").strip())
                        raise RuntimeError(
                            "Unconditional embedding returned all zeros for a non-empty negative prompt batch "
                            f"(count={negative_count}). Check CLIP encoders or prompt handling before sampling."
                        )
        finally:
            if smart_offload_enabled():
                if text_encoder_patchers:
                    pipeline_log("[conditioning] smart_offload: unloading text encoders after stage")
                enforce_smart_offload_text_encoders_off(sd_model, stage="txt2img.conditioning(post)")

        pair = (cond, uncond)
        if cache_enabled and key is not None:
            # Always keep only the most recent entry; older cache is discarded.
            self._conditioning_cache.clear()
            self._conditioning_cache[key] = detach_to_cpu(pair)
        return pair

    def _log_conditioning(self, cond: object, uncond: object) -> None:
        """Optional conditioning diagnostics controlled by --debug-conditioning / CODEX_DEBUG_COND."""
        try:
            if not getattr(backend_args.args, "debug_conditioning", False):
                return

            def _shape(t):
                if not isinstance(t, torch.Tensor):
                    return None
                return "x".join(str(int(dim)) for dim in t.shape)

            def _dtype(t):
                return str(t.dtype) if isinstance(t, torch.Tensor) else None

            def _device(t):
                return str(t.device) if isinstance(t, torch.Tensor) else None

            def _norm(t):
                return float(t.detach().abs().mean().item()) if isinstance(t, torch.Tensor) else None

            def _split_vector(v):
                if not isinstance(v, torch.Tensor):
                    return None, None
                if int(v.shape[1]) <= 1280:
                    pooled = v
                    pooled_l2 = float(pooled.detach().float().norm().item())
                    return (
                        float(pooled.detach().abs().mean().item()),
                        0.0,
                        pooled_l2,
                        0.0,
                    )
                pooled = v[:, :1280]
                adm = v[:, 1280:]
                pooled_l2 = float(pooled.detach().float().norm().item())
                adm_l2 = float(adm.detach().float().norm().item())
                return (
                    float(pooled.detach().abs().mean().item()),
                    float(adm.detach().abs().mean().item()),
                    pooled_l2,
                    adm_l2,
                )

            ca = cond.get("crossattn") if isinstance(cond, dict) else None
            va = cond.get("vector") if isinstance(cond, dict) else None
            ga = cond.get("guidance") if isinstance(cond, dict) else None
            ua = uncond.get("crossattn") if isinstance(uncond, dict) else None
            uv = uncond.get("vector") if isinstance(uncond, dict) else None
            ug = uncond.get("guidance") if isinstance(uncond, dict) else None

            p_mean, adm_mean, p_l2, adm_l2 = _split_vector(va) if va is not None else (None, None, None, None)
            up_mean, uadm_mean, up_l2, uadm_l2 = _split_vector(uv) if uv is not None else (None, None, None, None)

            def _guidance_scalar(t):
                if not isinstance(t, torch.Tensor) or t.numel() == 0:
                    return None
                try:
                    return float(t.detach().float().view(-1)[0].item())
                except Exception:
                    return None

            emit_backend_event(
                "conditioning.cond",
                logger=self._logger.name,
                cross_shape=_shape(ca),
                cross_dtype=_dtype(ca),
                cross_device=_device(ca),
                cross_mean_abs=(_norm(ca) or 0.0),
                cross_l2=float(ca.detach().float().norm().item()) if ca is not None else 0.0,
                vector_shape=_shape(va),
                vector_dtype=_dtype(va),
                vector_device=_device(va),
                vector_mean_abs=(_norm(va) or 0.0),
                vector_l2=float(va.detach().float().norm().item()) if va is not None else 0.0,
                pooled_mean_abs=(p_mean or 0.0),
                pooled_l2=(p_l2 or 0.0),
                adm_mean_abs=(adm_mean or 0.0),
                adm_l2=(adm_l2 or 0.0),
                guidance=_guidance_scalar(ga),
            )
            # Only log uncond if it exists (distilled CFG models like Flux don't use uncond)
            if uncond is not None:
                emit_backend_event(
                    "conditioning.uncond",
                    logger=self._logger.name,
                    cross_shape=_shape(ua),
                    cross_dtype=_dtype(ua),
                    cross_device=_device(ua),
                    cross_mean_abs=(_norm(ua) or 0.0),
                    cross_l2=float(ua.detach().float().norm().item()) if ua is not None else 0.0,
                    vector_shape=_shape(uv),
                    vector_dtype=_dtype(uv),
                    vector_device=_device(uv),
                    vector_mean_abs=(_norm(uv) or 0.0),
                    vector_l2=float(uv.detach().float().norm().item()) if uv is not None else 0.0,
                    pooled_mean_abs=(up_mean or 0.0),
                    pooled_l2=(up_l2 or 0.0),
                    adm_mean_abs=(uadm_mean or 0.0),
                    adm_l2=(uadm_l2 or 0.0),
                    guidance=_guidance_scalar(ug),
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("[conditioning] diagnostics skipped: %s", exc)

    @staticmethod
    def _is_zimage_l2p_processing(processing: CodexProcessingTxt2Img) -> bool:
        return str(getattr(getattr(processing, "sd_model", None), "engine_id", "") or "") == "zimage_l2p"

    @staticmethod
    def _is_lens_processing(processing: CodexProcessingTxt2Img) -> bool:
        return str(getattr(getattr(processing, "sd_model", None), "engine_id", "") or "") == "lens"

    def _ensure_zimage_l2p_rng(
        self,
        processing: CodexProcessingTxt2Img,
        plan: SamplingPlan,
    ) -> ImageRNG:
        ensure_sampler(processing, plan)
        rng = ImageRNG(
            (3, int(processing.height), int(processing.width)),
            plan.seeds,
            subseeds=plan.subseeds,
            subseed_strength=plan.subseed_strength,
            seed_resize_from_h=getattr(processing, "seed_resize_from_h", 0),
            seed_resize_from_w=getattr(processing, "seed_resize_from_w", 0),
            settings=plan.noise_settings,
        )
        processing.rng = rng
        return rng

    def _guard_zimage_l2p_txt2img(
        self,
        processing: CodexProcessingTxt2Img,
        prompt_context: PromptContext,
        plan: SamplingPlan,
    ) -> None:
        """Fail loud before conditioning for L2P unsupported surfaces."""

        if int(processing.width) != 1024 or int(processing.height) != 1024:
            raise RuntimeError(
                f"Z-Image L2P first tranche supports exactly 1024x1024; got {processing.width}x{processing.height}."
            )
        if int(processing.batch_size) != 1 or int(processing.iterations) != 1:
            raise RuntimeError(
                "Z-Image L2P first tranche supports only batch_size=1 and iterations=1 "
                f"(got batch_size={processing.batch_size}, iterations={processing.iterations})."
            )
        effective_prompts = list(prompt_context.prompts)
        effective_negative_prompts = (
            list(prompt_context.negative_prompts) if prompt_context.negative_prompts else [processing.negative_prompt]
        )
        if len(effective_prompts) != 1 or len(effective_negative_prompts) != 1:
            raise RuntimeError(
                "Z-Image L2P first tranche supports exactly one prompt and at most one optional negative prompt per request."
            )
        prompt_loras = list(getattr(prompt_context, "loras", ()) or ())
        if not prompt_context.negative_prompts:
            try:
                _cleaned, scalar_negative_loras = parse_prompts(effective_prompts + effective_negative_prompts)
            except ExtraNetsParseError as exc:
                raise RuntimeError("Z-Image L2P does not support LoRA prompt selections in tranche 1.") from exc
            prompt_loras.extend(scalar_negative_loras)
        if bool(getattr(getattr(processing, "hires", None), "enabled", False)):
            raise RuntimeError("Z-Image L2P does not support hires.")
        if getattr(processing, "firstpass_image", None) is not None:
            raise RuntimeError("Z-Image L2P does not support precomputed first-pass images.")
        if getattr(processing, "latent_scale_mode", None) is not None:
            raise RuntimeError("Z-Image L2P does not support latent scale modes.")
        if GlobalSwapModelStage(getattr(processing, "swap_model", None)).is_enabled():
            raise RuntimeError("Z-Image L2P does not support top-level swap_model.")
        if GlobalRefinerStage(getattr(processing, "refiner", None)).is_enabled():
            raise RuntimeError("Z-Image L2P does not support refiner.")
        ip_adapter = getattr(processing, "ip_adapter", None)
        if ip_adapter is not None and bool(getattr(ip_adapter, "enabled", False)):
            raise RuntimeError("Z-Image L2P does not support IP-Adapter.")
        if tuple(collect_lora_selections(prompt_loras)):
            raise RuntimeError("Z-Image L2P does not support LoRA prompt selections in tranche 1.")
        if prompt_context.clip_skip is not None:
            raise RuntimeError("Z-Image L2P does not support clip_skip.")
        if int(getattr(processing, "seed_resize_from_h", 0) or 0) or int(getattr(processing, "seed_resize_from_w", 0) or 0):
            raise RuntimeError("Z-Image L2P does not support seed resize.")
        sampler_name = str(plan.sampler_name or "").strip().lower()
        scheduler_name = str(plan.scheduler_name or "").strip().lower()
        if sampler_name != "euler" or scheduler_name != "simple":
            raise RuntimeError(
                "Z-Image L2P first tranche supports only sampler='euler' and scheduler='simple' "
                f"(got sampler={plan.sampler_name!r}, scheduler={plan.scheduler_name!r})."
            )
        hook = getattr(getattr(processing, "sd_model", None), "sample_pixel_txt2img", None)
        if not callable(hook):
            raise RuntimeError("Z-Image L2P engine is missing required sample_pixel_txt2img hook.")

    def _guard_lens_txt2img(
        self,
        processing: CodexProcessingTxt2Img,
        prompt_context: PromptContext,
        plan: SamplingPlan,
    ) -> None:
        """Fail loud before generic prompt application, RNG construction, or conditioning for Lens."""

        require_lens_dimensions(
            int(processing.width),
            int(processing.height),
            context="Lens txt2img dimensions",
        )
        if int(processing.batch_size) != 1 or int(processing.iterations) != 1:
            raise RuntimeError(
                "Lens first hook tranche supports only batch_size=1 and iterations=1 "
                f"(got batch_size={processing.batch_size}, iterations={processing.iterations})."
            )
        effective_prompts = list(prompt_context.prompts)
        effective_negative_prompts = list(prompt_context.negative_prompts)
        if len(effective_prompts) != 1 or len(effective_negative_prompts) > 1:
            raise RuntimeError(
                "Lens first hook tranche supports exactly one prompt and at most one optional negative prompt per request."
            )
        if bool(getattr(getattr(processing, "hires", None), "enabled", False)):
            raise RuntimeError("Lens does not support hires.")
        if getattr(processing, "firstpass_image", None) is not None:
            raise RuntimeError("Lens does not support precomputed first-pass images.")
        if getattr(processing, "latent_scale_mode", None) is not None:
            raise RuntimeError("Lens does not support latent scale modes.")
        if GlobalSwapModelStage(getattr(processing, "swap_model", None)).is_enabled():
            raise RuntimeError("Lens does not support top-level swap_model.")
        if GlobalRefinerStage(getattr(processing, "refiner", None)).is_enabled():
            raise RuntimeError("Lens does not support refiner.")
        ip_adapter = getattr(processing, "ip_adapter", None)
        if ip_adapter is not None and bool(getattr(ip_adapter, "enabled", False)):
            raise RuntimeError("Lens does not support IP-Adapter.")
        if tuple(collect_lora_selections(prompt_context.loras)):
            raise RuntimeError("Lens does not support LoRA prompt selections in the hook tranche.")
        if prompt_context.clip_skip is not None:
            raise RuntimeError("Lens does not support clip_skip.")
        if int(getattr(processing, "seed_resize_from_h", 0) or 0) or int(getattr(processing, "seed_resize_from_w", 0) or 0):
            raise RuntimeError("Lens does not support seed resize.")
        sampler_name = str(plan.sampler_name or "").strip().lower()
        scheduler_name = str(plan.scheduler_name or "").strip().lower()
        if sampler_name != "euler" or scheduler_name != "simple":
            raise RuntimeError(
                "Lens first hook tranche supports only sampler='euler' and scheduler='simple' "
                f"(got sampler={plan.sampler_name!r}, scheduler={plan.scheduler_name!r})."
            )
        hook = getattr(getattr(processing, "sd_model", None), "sample_lens_txt2img", None)
        if not callable(hook):
            raise RuntimeError("Lens engine is missing required sample_lens_txt2img hook.")

    def _apply_refiner_stage(
        self,
        stage: RefinerStage,
        processing: CodexProcessingTxt2Img,
        prompt_context: PromptContext,
        noise_settings,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        if not stage.is_enabled():
            return samples

        return stage.run(
            processing=processing,
            prompt_context=prompt_context,
            noise_settings=noise_settings,
            samples=samples,
            compute_conditioning=self._compute_conditioning,
            log_conditioning=self._log_conditioning,
            log_tensor_stats=self._log_tensor_stats,
        )

    # ------------------------------------------------------------------ public API
    @pipeline_trace
    def run(
        self,
        processing: CodexProcessingTxt2Img,
        conditioning_data,
        unconditional_data,
        seeds: Sequence[int],
        subseeds: Sequence[int],
        subseed_strength: float,
        prompts: Sequence[str],
    ) -> GenerationResult:
        from apps.backend.runtime.diagnostics.timeline import auto_save_and_print, timeline as runtime_timeline

        timeline_capture = None
        capture_context = runtime_timeline.capture(name="txt2img_pipeline_run") if runtime_timeline.enabled else nullcontext(None)

        with capture_context as timeline_capture:
            setattr(processing, "_codex_last_decode_engine", None)
            setattr(processing, "_codex_pipeline_mode", "txt2img")
            telemetry = resolve_pipeline_telemetry_context(
                processing,
                default_mode="txt2img",
                require_mode=True,
            )
            emit_backend_event(
                "pipeline.run.start",
                logger=self._logger.name,
                mode=telemetry.mode,
                stage="run.start",
                correlation_id=telemetry.correlation_id,
                correlation_source=telemetry.correlation_source,
                task_id=telemetry.task_id,
                engine_id=str(getattr(getattr(processing, "sd_model", None), "engine_id", "") or "unknown"),
            )
            t_start = time.perf_counter()
            t_prepare_end: float | None = None
            t_base_end: float | None = None
            t_swap_end: float | None = None
            t_hires_end: float | None = None
            t_refiner_end: float | None = None

            model_device, model_dtype = self._sd_model_device_info(processing)
            if model_device is not None:
                self._logger.info("SDXL sd_model device=%s dtype=%s", model_device, model_dtype)

            state = self._prepare_state(
                processing,
                conditioning_data,
                unconditional_data,
                seeds,
                subseeds,
                subseed_strength,
                prompts,
            )
            emit_backend_event(
                "pipeline.stage.complete",
                logger=self._logger.name,
                mode=telemetry.mode,
                stage="prepare.complete",
                stage_name="prepare",
                correlation_id=telemetry.correlation_id,
                correlation_source=telemetry.correlation_source,
                task_id=telemetry.task_id,
                hires_enabled=bool(processing.hires.enabled),
            )
            t_prepare_end = time.perf_counter()
            base_result = self._execute_base_sampling(processing, state)
            t_base_end = time.perf_counter()
            emit_backend_event(
                "pipeline.stage.complete",
                logger=self._logger.name,
                mode=telemetry.mode,
                stage="base_sampling.complete",
                stage_name="base_sampling",
                correlation_id=telemetry.correlation_id,
                correlation_source=telemetry.correlation_source,
                task_id=telemetry.task_id,
                samples_shape=tuple(int(dim) for dim in base_result.samples.shape),
            )

            final_samples = self._maybe_run_swap_model_pass(processing, state, base_result)
            if getattr(processing, "swap_model", None) is not None:
                t_swap_end = time.perf_counter()
                emit_backend_event(
                    "pipeline.stage.complete",
                    logger=self._logger.name,
                    mode=telemetry.mode,
                    stage="swap_model_sampling.complete",
                    stage_name="swap_model_sampling",
                    correlation_id=telemetry.correlation_id,
                    correlation_source=telemetry.correlation_source,
                    task_id=telemetry.task_id,
                    samples_shape=tuple(int(dim) for dim in final_samples.shape),
                )
            else:
                t_swap_end = t_base_end
            final_decoded = base_result.decoded

            if processing.hires.enabled:
                hires_source_result = self._prepare_hires_source_result(processing, base_result, final_samples)
                self._reload_for_hires(processing, state)
                final_samples = self._run_hires_pass(processing, state, hires_source_result)
                final_decoded = None
                t_hires_end = time.perf_counter()
                emit_backend_event(
                    "pipeline.stage.complete",
                    logger=self._logger.name,
                    mode=telemetry.mode,
                    stage="hires_sampling.complete",
                    stage_name="hires_sampling",
                    correlation_id=telemetry.correlation_id,
                    correlation_source=telemetry.correlation_source,
                    task_id=telemetry.task_id,
                    samples_shape=tuple(int(dim) for dim in final_samples.shape),
                )
            else:
                t_hires_end = t_swap_end or t_base_end

            final_samples = self._maybe_run_refiner_pass(processing, state, final_samples)
            if GlobalRefinerStage(getattr(processing, "refiner", None)).is_enabled():
                final_decoded = None
            t_refiner_end = time.perf_counter()

            try:
                timings: dict[str, float] = {}
                if t_prepare_end is not None:
                    timings["prepare_ms"] = (t_prepare_end - t_start) * 1000.0
                if t_base_end is not None and t_prepare_end is not None:
                    timings["base_sampling_ms"] = (t_base_end - t_prepare_end) * 1000.0
                if getattr(processing, "swap_model", None) is not None and t_swap_end is not None and t_base_end is not None:
                    timings["swap_model_ms"] = max(0.0, (t_swap_end - t_base_end) * 1000.0)
                if processing.hires.enabled and t_hires_end is not None and t_swap_end is not None:
                    timings["hires_ms"] = (t_hires_end - t_swap_end) * 1000.0
                if t_refiner_end is not None and t_hires_end is not None:
                    timings["refiner_ms"] = max(0.0, (t_refiner_end - t_hires_end) * 1000.0)
                timings["total_pipeline_ms"] = max(0.0, (t_refiner_end or time.perf_counter()) - t_start) * 1000.0
                processing.update_extra_param("Timings (ms)", timings)
                emit_backend_event(
                    "pipeline.run.complete",
                    logger=self._logger.name,
                    mode=telemetry.mode,
                    stage="run.complete",
                    correlation_id=telemetry.correlation_id,
                    correlation_source=telemetry.correlation_source,
                    task_id=telemetry.task_id,
                    hires_enabled=bool(processing.hires.enabled),
                    total_pipeline_ms=float(timings["total_pipeline_ms"]),
                )
            except Exception:
                # Timings must never break generation; swallow errors defensively.
                pass

        # Auto-print and save timeline trace if enabled
        if runtime_timeline.enabled:
            try:
                captured_events = getattr(timeline_capture, "events", None)
                if not captured_events:
                    self._logger.warning(
                        "CODEX_TIMELINE is enabled, but txt2img run captured zero timeline events."
                    )
                trace_path = auto_save_and_print(timeline_capture)
                if trace_path:
                    processing.update_extra_param("Timeline Trace", trace_path)
            except Exception:
                pass  # Timeline should never break generation

        return GenerationResult(
            samples=final_samples,
            decoded=final_decoded,
            metadata={"conditioning_cache_hit": bool(getattr(processing, "_codex_conditioning_cache_hit", False))},
            decode_engine=getattr(processing, "_codex_last_decode_engine", None) or processing.sd_model,
        )

    # ------------------------------------------------------------------ stages
    @pipeline_trace
    def _prepare_state(
        self,
        processing: CodexProcessingTxt2Img,
        conditioning_data,
        unconditional_data,
        seeds: Sequence[int],
        subseeds: Sequence[int],
        subseed_strength: float,
        prompts: Sequence[str],
    ) -> PrepareState:
        prompt_context = build_prompt_context(processing, prompts)
        plan = build_sampling_plan(processing, seeds, subseeds, subseed_strength)
        is_lens = self._is_lens_processing(processing)
        if is_lens:
            self._guard_lens_txt2img(processing, prompt_context, plan)
            if conditioning_data is not None or unconditional_data is not None:
                raise RuntimeError("Lens txt2img hook owns conditioning; precomputed conditioning is unsupported.")
        else:
            apply_prompt_context(processing, prompt_context)

        is_zimage_l2p = self._is_zimage_l2p_processing(processing)
        if is_lens:
            rng = None
        elif is_zimage_l2p:
            rng = self._ensure_zimage_l2p_rng(processing, plan)
        else:
            rng = ensure_sampler_and_rng(processing, plan)

        processing.seeds = list(plan.seeds)
        processing.subseeds = list(plan.subseeds)
        processing.guidance_scale = plan.guidance_scale
        processing.cfg_scale = plan.guidance_scale
        processing.steps = plan.steps

        payload = ConditioningPayload(conditioning=conditioning_data, unconditional=unconditional_data)

        if is_lens:
            init_latents = None
            init_decoded = None
        else:
            processing.prepare_prompt_data()

            run_process_scripts(processing)

            init_latents, init_decoded = self._prepare_first_pass_from_image(processing)
            if processing.hires.enabled:
                self._resolve_hires_execution(processing, emit_plan_event=True)
            if is_zimage_l2p:
                self._guard_zimage_l2p_txt2img(processing, prompt_context, plan)

        # Compute conditioning if not provided (SDXL path); preserve metadata (width/height/targets) after overrides.
        cond = conditioning_data
        uncond = unconditional_data
        if is_lens:
            cond = None
            uncond = None
            payload = ConditioningPayload(conditioning=None, unconditional=None)
        elif cond is None or uncond is None:
            cond, uncond = self._compute_conditioning(processing, prompt_context)
            # For distilled CFG models (Flux), uncond is intentionally None - only cond is required
            if cond is None:
                raise RuntimeError("Failed to build conditioning for txt2img; get_learned_conditioning returned None.")
            payload = ConditioningPayload(conditioning=cond, unconditional=uncond)
            self._log_conditioning(cond, uncond)

        return PrepareState(
            prompt_context=prompt_context,
            hires_prompt_context=None,
            sampling_plan=plan,
            rng=rng,
            payload=payload,
            init_latents=init_latents,
            init_decoded=init_decoded,
            cond=cond,
            uncond=uncond,
        )

    @pipeline_trace
    def _execute_base_sampling(self, processing: CodexProcessingTxt2Img, state: PrepareState) -> SamplingOutput:
        base_samples = state.init_latents
        decoded_samples = state.init_decoded
        boundary_state: SamplingBoundaryState | None = None
        if self._is_lens_processing(processing):
            if base_samples is not None or decoded_samples is not None:
                raise RuntimeError("Lens txt2img hook must start from Lens sampler-owned sequential latent noise.")
            hook = getattr(processing.sd_model, "sample_lens_txt2img", None)
            if not callable(hook):
                raise RuntimeError("Lens engine is missing required sample_lens_txt2img hook.")

            progress_owner_token = str(getattr(processing, "_codex_progress_owner_token", "") or "").strip()
            if not progress_owner_token:
                raise RuntimeError("Lens progress owner token is missing.")
            sampling_steps = int(state.sampling_plan.steps)
            if sampling_steps <= 0:
                raise RuntimeError("Lens sampling steps must be positive for progress reporting.")

            def _lens_progress_int(payload: dict[str, object], key: str) -> int:
                if key not in payload:
                    raise RuntimeError(f"Lens progress payload missing required progress key '{key}'.")
                value = payload[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise RuntimeError(f"Lens progress payload key '{key}' must be a non-bool int.")
                return int(value)

            def _lens_progress(info: dict[str, object]) -> None:
                payload = dict(info)
                if backend_state.should_stop:
                    raise RuntimeError("cancelled")
                step = _lens_progress_int(payload, "step")
                payload_steps = _lens_progress_int(payload, "steps")
                if payload_steps <= 0:
                    raise RuntimeError("Lens progress payload steps must be positive.")
                if payload_steps != sampling_steps:
                    raise RuntimeError(
                        "Lens progress payload steps "
                        f"does not match sampling plan steps (payload={payload_steps}, plan={sampling_steps})."
                    )
                if step < 1 or step > sampling_steps:
                    raise RuntimeError(f"Lens progress step out of range (step={step}, steps={sampling_steps}).")
                backend_state.update_sampling(step=step, total=sampling_steps, owner_token=progress_owner_token)
                backend_state.reset_sampling_blocks(owner_token=progress_owner_token)
                processing.update_extra_param("Lens Progress", payload)

            backend_state.start(
                job_count=1,
                sampling_steps=sampling_steps,
                progress_owner_token=progress_owner_token,
            )
            try:
                decoded_output = hook(
                    prompt_context=state.prompt_context,
                    sampling_plan=state.sampling_plan,
                    width=int(processing.width),
                    height=int(processing.height),
                    batch_size=int(processing.batch_size),
                    iterations=int(processing.iterations),
                    progress_callback=_lens_progress,
                )
            finally:
                backend_state.end()
                backend_state.clear_flags()
            decoded_samples = validate_lens_decoded_output(
                decoded_output,
                batch_size=int(processing.batch_size),
                height=int(processing.height),
                width=int(processing.width),
            )
            self._log_tensor_stats("lens_decoded_samples", decoded_samples)
            return SamplingOutput(samples=decoded_samples, decoded=decoded_samples, boundary_state=None)

        if self._is_zimage_l2p_processing(processing):
            if base_samples is not None or decoded_samples is not None:
                raise RuntimeError("Z-Image L2P pixel txt2img must start from sampler-owned RGB noise.")
            hook = getattr(processing.sd_model, "sample_pixel_txt2img", None)
            if not callable(hook):
                raise RuntimeError("Z-Image L2P engine is missing required sample_pixel_txt2img hook.")

            progress_owner_token = str(getattr(processing, "_codex_progress_owner_token", "") or "").strip()
            if not progress_owner_token:
                raise RuntimeError("Z-Image L2P progress owner token is missing.")
            sampling_steps = int(state.sampling_plan.steps)
            if sampling_steps <= 0:
                raise RuntimeError("Z-Image L2P sampling steps must be positive for progress reporting.")

            def _l2p_progress_int(payload: dict[str, object], key: str) -> int:
                if key not in payload:
                    raise RuntimeError(f"Z-Image L2P progress payload missing required progress key '{key}'.")
                value = payload[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise RuntimeError(f"Z-Image L2P progress payload key '{key}' must be a non-bool int.")
                return int(value)

            def _l2p_progress(info: dict[str, object]) -> None:
                payload = dict(info)
                if backend_state.should_stop:
                    raise RuntimeError("cancelled")
                step = _l2p_progress_int(payload, "step")
                payload_steps = _l2p_progress_int(payload, "steps")
                if payload_steps <= 0:
                    raise RuntimeError("Z-Image L2P progress payload steps must be positive.")
                if payload_steps != sampling_steps:
                    raise RuntimeError(
                        "Z-Image L2P progress payload steps "
                        f"does not match sampling plan steps (payload={payload_steps}, plan={sampling_steps})."
                    )
                if step < 1 or step > sampling_steps:
                    raise RuntimeError(
                        f"Z-Image L2P progress step out of range (step={step}, steps={sampling_steps})."
                    )
                backend_state.update_sampling(step=step, total=sampling_steps, owner_token=progress_owner_token)
                backend_state.reset_sampling_blocks(owner_token=progress_owner_token)
                processing.update_extra_param("Z-Image L2P Progress", payload)

            backend_state.start(
                job_count=1,
                sampling_steps=sampling_steps,
                progress_owner_token=progress_owner_token,
            )
            try:
                pixel_samples = hook(
                    cond=state.cond,
                    uncond=state.uncond,
                    sampling_plan=state.sampling_plan,
                    prompt_context=state.prompt_context,
                    rng=state.rng,
                    width=int(processing.width),
                    height=int(processing.height),
                    progress_callback=_l2p_progress,
                )
            finally:
                backend_state.end()
                backend_state.clear_flags()
            if not isinstance(pixel_samples, torch.Tensor) or pixel_samples.ndim != 4:
                raise RuntimeError(
                    "Z-Image L2P pixel hook returned invalid output; "
                    f"type={type(pixel_samples).__name__} shape={getattr(pixel_samples, 'shape', None)}"
                )
            self._log_tensor_stats("zimage_l2p_pixel_samples", pixel_samples)
            return SamplingOutput(samples=pixel_samples, decoded=pixel_samples, boundary_state=None)

        swap_stage = GlobalSwapModelStage(getattr(processing, "swap_model", None))
        capture_boundary_state_at_step: int | None = None

        if swap_stage.is_enabled():
            if base_samples is not None or decoded_samples is not None:
                raise RuntimeError(
                    "Top-level swap_model exact resume requires a real first-pass sampling run; "
                    "it is unsupported when first-pass latents/decoded images were precomputed before sampling."
                )
            swap_cfg = getattr(processing, "swap_model", None)
            if swap_cfg is None:
                raise RuntimeError("swap_model is enabled but processing.swap_model is missing.")
            resolve_swap_pointer_remaining_steps(
                label="Swap Model",
                swap_at_step=int(swap_cfg.swap_at_step),
                total_steps=int(state.sampling_plan.steps),
            )
            capture_boundary_state_at_step = int(swap_cfg.swap_at_step)

        if base_samples is None and decoded_samples is None:
            if state.rng is None:
                raise RuntimeError("Generic txt2img sampling requires ImageRNG; only exact Lens processing may use rng=None.")
            base_result = execute_sampling_result(
                processing,
                state.sampling_plan,
                state.payload,
                state.prompt_context,
                state.prompt_context.loras,
                rng=state.rng,
                denoise_strength=1.0,
                capture_boundary_state_at_step=capture_boundary_state_at_step,
            )
            base_samples = base_result.samples
            boundary_state = base_result.boundary_state
            if capture_boundary_state_at_step is not None and boundary_state is None:
                raise RuntimeError(
                    "Base sampling returned without a boundary_state for top-level swap_model exact resume."
                )
            if boundary_state is None:
                decoded_samples = maybe_decode_for_hr(
                    processing,
                    base_samples,
                    hires_upscaler_id=processing.hires.require_upscaler_id() if processing.hires.enabled else None,
                )
        elif base_samples is None and decoded_samples is not None:
            tensor = self._to_device_dtype_if_needed(
                decoded_samples,
                device=devices.default_device(),
                dtype=torch.float32,
            )
            base_samples = encode_image_batch(
                processing.sd_model,
                tensor,
                stage="use_cases.txt2img.runner.execute_base_sampling.encode",
            )

        if base_samples is None:
            raise RuntimeError("txt2img failed to produce initial samples")

        self._log_tensor_stats("base_samples", base_samples)
        self._log_tensor_stats("base_decoded", decoded_samples)

        return SamplingOutput(samples=base_samples, decoded=decoded_samples, boundary_state=boundary_state)

    @pipeline_trace
    def _reload_for_hires(self, processing: CodexProcessingTxt2Img, state: PrepareState) -> None:
        swap_model = processing.hires.swap_model
        if swap_model is None or not swap_model.is_configured():
            return

        model_name = swap_model.require_model_ref(context="Hires swap_model")
        processing.firstpass_use_distilled_cfg_scale = processing.sd_model.use_distilled_cfg_scale
        load_opts = EngineLoadOptions(
            device=None,
            dtype=None,
            attention_backend=None,
            accelerator=None,
            vae_path=swap_model.vae_path,
            vae_source=swap_model.vae_source,
            tenc_path=list(swap_model.tenc_path) if isinstance(swap_model.tenc_path, tuple) else swap_model.tenc_path,
            text_encoder_override=dict(swap_model.text_encoder_override) if swap_model.text_encoder_override else None,
            checkpoint_core_only=swap_model.checkpoint_core_only,
            model_format=swap_model.model_format,
            zimage_variant=swap_model.zimage_variant,
        )
        try:
            processing.sd_model = _load_engine(model_name, options=load_opts)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load hires swap_model '{model_name}': {exc}") from exc

        if processing.sd_model.use_distilled_cfg_scale:
            processing.extra_generation_params["Hires Distilled CFG Scale"] = processing.hires.distilled_cfg

    def _prepare_hires_source_result(
        self,
        processing: CodexProcessingTxt2Img,
        base_result: SamplingOutput,
        samples: torch.Tensor,
    ) -> SamplingOutput:
        if samples is base_result.samples:
            return base_result

        decoded_samples = maybe_decode_for_hr(
            processing,
            samples,
            hires_upscaler_id=processing.hires.require_upscaler_id(),
        )
        self._log_tensor_stats("swap_model_hires_source_samples", samples)
        self._log_tensor_stats("swap_model_hires_source_decoded", decoded_samples)
        return SamplingOutput(samples=samples, decoded=decoded_samples, boundary_state=None)

    @pipeline_trace
    def _run_hires_pass(
        self,
        processing: CodexProcessingTxt2Img,
        state: PrepareState,
        base_result: SamplingOutput,
    ) -> torch.Tensor:
        hires_cfg = processing.hires
        upscaler_id, target_width, target_height, steps, denoise = self._resolve_hires_execution(
            processing,
            emit_plan_event=False,
        )
        processing.ensure_hires_prompts()

        original_attrs = {
            "prompts": processing.prompts,
            "negative_prompts": getattr(processing, "negative_prompts", []),
            "width": processing.width,
            "height": processing.height,
            "guidance_scale": processing.guidance_scale,
            "steps": processing.steps,
            "sampler_name": getattr(processing, "sampler_name", None),
            "scheduler": getattr(processing, "scheduler", None),
            "sampler": getattr(processing, "sampler", None),
        }

        hr_prompts_source = (
            processing.hires_prompts
            or processing.all_prompts
            or state.prompt_context.prompts
        )
        hr_negative_source = processing.hires_negative_prompts or original_attrs["negative_prompts"]
        hires_prompt_context = build_hires_prompt_context(
            prompt_seed=list(hr_prompts_source),
            negative_seed=list(hr_negative_source),
            base_context=state.prompt_context,
        )
        state.hires_prompt_context = hires_prompt_context

        try:
            processing.prompts = hires_prompt_context.prompts
            processing.negative_prompts = hires_prompt_context.negative_prompts
            processing.width = target_width
            processing.height = target_height
            effective_target_width = int(processing.width)
            effective_target_height = int(processing.height)
            processing.guidance_scale = float(hires_cfg.cfg or processing.guidance_scale)
            processing.cfg_scale = processing.guidance_scale
            processing.steps = int(steps)
            hires_runtime_plan = replace(
                state.sampling_plan,
                steps=int(processing.steps),
                guidance_scale=float(processing.guidance_scale),
                er_sde=None,
            )
            hires_sampler, hires_scheduler = resolve_sampler_scheduler_override(
                base_sampler=str(hires_runtime_plan.sampler_name or ""),
                base_scheduler=str(hires_runtime_plan.scheduler_name or ""),
                sampler_override=getattr(hires_cfg, "sampler_name", None),
                scheduler_override=getattr(hires_cfg, "scheduler", None),
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

            telemetry = resolve_pipeline_telemetry_context(
                processing,
                default_mode="txt2img",
                require_mode=True,
            )
            engine_id = str(getattr(getattr(processing, "sd_model", None), "engine_id", "") or "unknown")
            hires_strategy = str(getattr(processing, "_codex_hires_strategy", "unknown") or "unknown")
            base_samples_shape = tuple(int(dim) for dim in base_result.samples.shape)
            base_decoded_shape = (
                tuple(int(dim) for dim in base_result.decoded.shape)
                if isinstance(base_result.decoded, torch.Tensor)
                else None
            )
            emit_backend_event(
                "pipeline.hires.transition",
                logger=self._logger.name,
                mode=telemetry.mode,
                stage="hires.transition",
                correlation_id=telemetry.correlation_id,
                correlation_source=telemetry.correlation_source,
                task_id=telemetry.task_id,
                engine_id=engine_id,
                strategy=hires_strategy,
                upscaler_id=upscaler_id,
                target_width=effective_target_width,
                target_height=effective_target_height,
                steps=steps,
                denoise=denoise,
                sampler=hires_sampler,
                scheduler=hires_scheduler,
                base_samples_shape=base_samples_shape,
                base_decoded_shape=base_decoded_shape,
            )
            self._logger.info(
                "[hires] transition base_to_hires upscaler=%s target=%dx%d steps=%d denoise=%.4f sampler=%s scheduler=%s base_samples=%s base_decoded=%s",
                upscaler_id,
                effective_target_width,
                effective_target_height,
                steps,
                denoise,
                hires_sampler,
                hires_scheduler,
                base_samples_shape,
                base_decoded_shape,
            )

            hires_inputs = prepare_hires_latents_and_conditioning(
                processing,
                base_samples=base_result.samples,
                base_decoded=base_result.decoded,
                target_width=int(effective_target_width),
                target_height=int(effective_target_height),
                upscaler_id=upscaler_id,
                tile=getattr(hires_cfg, "tile", None),
            )
            latents = hires_inputs.latents
            image_conditioning = hires_inputs.image_conditioning
            continuation_mode = hires_inputs.continuation_mode

            hires_settings = state.sampling_plan.noise_settings
            rng_hr = ImageRNG(
                (latents.shape[1], latents.shape[2], latents.shape[3]),
                state.sampling_plan.seeds,
                subseeds=state.sampling_plan.subseeds,
                subseed_strength=state.sampling_plan.subseed_strength,
                seed_resize_from_h=getattr(processing, "seed_resize_from_h", 0),
                seed_resize_from_w=getattr(processing, "seed_resize_from_w", 0),
                settings=hires_settings,
            )
            noise = self._to_device_dtype_if_needed(
                rng_hr.next(),
                device=latents.device,
                dtype=latents.dtype,
            )
            start_index = 0
            denoise_strength: float | None = float(denoise)
            init_latent: torch.Tensor | None = latents
            sampling_image_conditioning = image_conditioning
            allow_txt2img_conditioning_fallback = True
            image_conditioning_shape = (
                tuple(int(dim) for dim in image_conditioning.shape)
                if isinstance(image_conditioning, torch.Tensor)
                else None
            )
            emit_backend_event(
                "pipeline.hires.inputs_ready",
                logger=self._logger.name,
                mode=telemetry.mode,
                stage="hires.inputs_ready",
                correlation_id=telemetry.correlation_id,
                correlation_source=telemetry.correlation_source,
                task_id=telemetry.task_id,
                engine_id=engine_id,
                strategy=hires_strategy,
                continuation_mode=continuation_mode,
                latents_shape=tuple(int(dim) for dim in latents.shape),
                image_conditioning_shape=image_conditioning_shape,
                start_at_step=start_index,
                total_steps=int(processing.steps),
                allow_txt2img_conditioning_fallback=allow_txt2img_conditioning_fallback,
            )
            self._logger.info(
                "[hires] upscale_ready latents=%s image_conditioning=%s start_at_step=%d total_steps=%d",
                tuple(int(dim) for dim in latents.shape),
                image_conditioning_shape,
                start_index,
                int(processing.steps),
            )

            hires_plan = replace(
                hires_runtime_plan,
                sampler_name=hires_sampler,
                scheduler_name=hires_scheduler,
                steps=int(processing.steps),
                guidance_scale=float(processing.guidance_scale),
                er_sde=resolve_er_sde_options_for_sampler(processing, hires_sampler),
            )

            # Recompute conditioning for hires pass with updated width/height/targets (SDXL parity).
            cond_hr, uncond_hr = self._compute_conditioning(processing, hires_prompt_context)
            if cond_hr is not None:
                state.payload = ConditioningPayload(conditioning=cond_hr, unconditional=uncond_hr)
                self._log_conditioning(cond_hr, uncond_hr)
            if continuation_mode == "image_latents":
                if not isinstance(state.payload.conditioning, dict):
                    raise TypeError(
                        "Hires Kontext continuation requires dict conditioning to inject image_latents; "
                        f"got {type(state.payload.conditioning).__name__}."
                    )
                state.payload.conditioning["image_latents"] = latents
                if isinstance(state.payload.unconditional, dict):
                    state.payload.unconditional["image_latents"] = latents
                init_latent = None
                sampling_image_conditioning = None
                denoise_strength = None
                allow_txt2img_conditioning_fallback = False
                if not (math.isclose(float(denoise), 0.0) or math.isclose(float(denoise), 1.0)):
                    self._logger.warning(
                        "[hires] kontext continuation ignores denoise schedule semantics (configured denoise=%.4f).",
                        float(denoise),
                    )

            samples = execute_sampling(
                processing,
                hires_plan,
                state.payload,
                hires_prompt_context,
                hires_prompt_context.loras,
                rng=rng_hr,
                noise=noise,
                image_conditioning=sampling_image_conditioning,
                init_latent=init_latent,
                start_at_step=start_index,
                denoise_strength=denoise_strength,
                img2img_fix_steps=True,
                allow_txt2img_conditioning_fallback=allow_txt2img_conditioning_fallback,
            )

            if processing.hires.refiner is not None:
                samples = self._apply_refiner_stage(
                    HiresRefinerStage(processing.hires.refiner),
                    processing,
                    hires_prompt_context,
                    hires_settings,
                    samples,
                )
            return samples
        finally:
            processing.prompts = original_attrs["prompts"]
            processing.negative_prompts = original_attrs["negative_prompts"]
            processing.width = original_attrs["width"]
            processing.height = original_attrs["height"]
            processing.guidance_scale = original_attrs["guidance_scale"]
            processing.cfg_scale = processing.guidance_scale
            processing.steps = original_attrs["steps"]
            processing.sampler_name = original_attrs["sampler_name"]
            processing.scheduler = original_attrs["scheduler"]
            processing.sampler = original_attrs["sampler"]
            processing.prepare_prompt_data()

    @pipeline_trace
    def _maybe_run_refiner_pass(
        self,
        processing: CodexProcessingTxt2Img,
        state: PrepareState,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        stage = GlobalRefinerStage(getattr(processing, "refiner", None))
        return self._apply_refiner_stage(
            stage,
            processing,
            state.prompt_context,
            state.sampling_plan.noise_settings,
            samples,
        )

    @pipeline_trace
    def _maybe_run_swap_model_pass(
        self,
        processing: CodexProcessingTxt2Img,
        state: PrepareState,
        base_result: SamplingOutput,
    ) -> torch.Tensor:
        stage = GlobalSwapModelStage(getattr(processing, "swap_model", None))
        if not stage.is_enabled():
            return base_result.samples
        if base_result.boundary_state is None:
            raise RuntimeError(
                "Top-level swap_model exact resume was requested but the base sampling stage did not provide a boundary_state bridge."
            )
        return stage.run(
            processing=processing,
            prompt_context=state.prompt_context,
            noise_settings=state.sampling_plan.noise_settings,
            boundary_state=base_result.boundary_state,
            compute_conditioning=self._compute_conditioning,
            log_conditioning=self._log_conditioning,
            log_tensor_stats=self._log_tensor_stats,
        )

    # ------------------------------------------------------------------ helpers
    @pipeline_trace
    def _prepare_first_pass_from_image(
        self, processing: CodexProcessingTxt2Img
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        image = processing.firstpass_image
        if image is None or not processing.hires.enabled:
            return None, None

        if processing.latent_scale_mode is None:
            array = np.array(image).astype(np.float32) / 255.0
            array = array * 2.0 - 1.0
            array = np.moveaxis(array, 2, 0)
            decoded_samples = torch.from_numpy(np.expand_dims(array, 0))
            return None, decoded_samples

        array = np.array(image).astype(np.float32) / 255.0
        array = np.moveaxis(array, 2, 0)
        tensor = self._to_device_dtype_if_needed(
            torch.from_numpy(np.expand_dims(array, axis=0)),
            device=devices.default_device(),
            dtype=torch.float32,
        )

        samples = encode_image_batch(
            processing.sd_model,
            tensor,
            stage="use_cases.txt2img.runner.prepare_first_pass_from_image.encode",
        )
        devices.torch_gc()
        return samples, None

    @pipeline_trace
    def _resolve_hires_execution(
        self,
        processing: CodexProcessingTxt2Img,
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
        target_width, target_height = hi_cfg.resolve_target_dimensions(
            base_width=int(processing.width),
            base_height=int(processing.height),
        )
        steps = hi_cfg.resolve_second_pass_steps(base_steps=int(processing.steps))
        denoise = float(hi_cfg.denoise)
        if emit_plan_event:
            telemetry = resolve_pipeline_telemetry_context(
                processing,
                default_mode="txt2img",
                require_mode=True,
            )
            emit_backend_event(
                "pipeline.hires.plan",
                logger=self._logger.name,
                mode=telemetry.mode,
                stage="hires.plan",
                correlation_id=telemetry.correlation_id,
                correlation_source=telemetry.correlation_source,
                task_id=telemetry.task_id,
                engine_id=engine_id,
                strategy=hires_strategy,
                upscaler_id=upscaler_id,
                target_width=target_width,
                target_height=target_height,
                steps=int(steps),
                denoise=float(denoise),
            )
        return upscaler_id, int(target_width), int(target_height), int(steps), float(denoise)

    # ------------------------------------------------------------------ diagnostics
    def _sd_model_device_info(
        self, processing: CodexProcessingTxt2Img
    ) -> tuple[torch.device | None, torch.dtype | None]:
        model = getattr(processing, "sd_model", None)
        if model is None:
            return None, None
        tensor = None
        try:
            params = model.parameters()
            if params is not None:
                tensor = next(params)
        except StopIteration:
            tensor = None
        except Exception:
            tensor = None
        if tensor is None:
            candidate = getattr(model, "weight", None)
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
        if tensor is not None and isinstance(tensor, torch.Tensor):
            return tensor.device, tensor.dtype
        device_attr = getattr(model, "device", None)
        if isinstance(device_attr, torch.device):
            return device_attr, None
        return None, None

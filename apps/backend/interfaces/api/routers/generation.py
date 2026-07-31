"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Generation API routes (txt2img/img2img/image-automation/txt2vid/img2vid/vid2vid).
Contains request parsing and payload validation (including hires tile config via `extras.hires.tile` / `img2img_extras.hires.tile`, Z-Image Turbo/Base
`extras.zimage_variant`, Qwen Image Edit-2511-only admission, and WAN video export options like `video_return_frames`), and delegates image task workers to
`apps/backend/interfaces/api/tasks/generation_tasks.py`.
Also owns the backend-owned `/api/image-automation` envelope (loop/seed/prompt/init-source parsing plus repo-fenced folder and wildcard roots),
validates nested IP-Adapter selectors/source kinds before delegating runtime application to the shared sampling stage, and preflights native
SUPIR mode under `img2img_extras.supir` for truthful SDXL img2img/inpaint admission, including fail-loud rejection of `img2img_extras.guidance`
when SUPIR mode is active.
Txt2img model-stage ownership is explicit: top-level `extras.swap_model` is the first-pass mid-generation stage config, `extras.hires.swap_model`
is the selector-only second-pass replacement seam, and `extras.refiner` / `extras.hires.refiner` remain SDXL-native refiner stages.
Hires supports sampler/scheduler overrides for the hires pass (txt2img: `extras.hires.sampler` / `extras.hires.scheduler`; img2img: `img2img_extras.hires.sampler` / `img2img_extras.hires.scheduler`), requires an explicit scheduler when the sampler is overridden, and validates override compatibility at API parse-time.
Img2img masking uses Forge/A1111 “Only masked” semantics only (no whole-picture inpaint area), supports optional multi-region inpaint passes via
`img2img_mask_region_split`, and is rejected at request time when the active engine capability surface does not support mask/inpaint semantics.
The public masked-runtime field is now `img2img_inpaint_mode`; the router rejects removed `img2img_mask_enforcement`, validates exact-engine mode support,
and preflights SDXL Fooocus/BrushNet assets before task creation.
Includes strict ER-SDE/guidance option parsing (`extras.er_sde` / `img2img_extras.er_sde`, `extras.guidance` / `img2img_extras.guidance`) plus release-scope
enforcement for sampler fields. Image-request sampler/scheduler validation also enforces family-scoped `supported_*` / `excluded_*` capability contracts
(base pair + hires overrides) without promoting recommendation hints into allowlists.
Uses cached inventory slot metadata for sha-selected text encoders (`tenc_sha`, plus SDXL-native `tenc1_sha` / `tenc2_sha`) and enforces WAN video
`height/width % 16 == 0` (Diffusers parity) to avoid silent patch-grid cropping (returns suggested rounded-up dimensions on invalid requests).
Resolves WAN `wan_vae_sha` through VAE inventory ownership and validates VAE config availability before runtime dispatch (`bundle_dir/config.json` for directory VAEs, or sibling/metadata `vae/config.json` for file VAEs).
Validates `extras.vae_sha` against VAE inventory ownership (rejects non-VAE asset SHAs before runtime load) to keep Flux core-only causality fail-loud at request time.
Image request selectors are explicit: the router validates `model_sha`, `checkpoint_core_only`, `model_format`, and `vae_source` for VAE-using engines
against inventory metadata instead of probing checkpoint families or inferring core-only status from checkpoint names.
Z-Image L2P requests are exact txt2img-only, pixel-space, no-VAE 1024x1024 requests; the router rejects stale VAE, variant, hires, refiner,
swap-model, IP-Adapter, LoRA, clip-skip, unsupported sampler/scheduler, unsupported model-format state, wrong-family denoisers, denoiser GGUFs
without the dedicated L2P profile metadata, wrong-root Qwen3-4B text encoders, and TEnc GGUFs without the dedicated L2P profile metadata before task creation.
Qwen Image is Edit-2511 img2img-only: txt2img and image automation fail before task creation, and its exact top-level allowlist,
required core fields, stale public variant, foreign-family selectors, classic denoise/resize/mask/hires/IP-Adapter/LoRA/SUPIR surfaces,
and removed top-level `model` selector are validated before task registration. Its public sampling contract is exact `euler` / `simple`
with at least two steps, and `img2img_extras.model_sha` is the sole transformer selector. Transformer, VAE, and text-encoder SHA
selection is scoped to exact Qwen roots and strict Edit-2511 metadata/topology; the VAE must match the vendored
`AutoencoderKLQwenImage` config.
Lens is parked in this tranche: explicit `engine="lens"` txt2img/img2img/image-automation requests fail before task creation, and
the only accepted Lens variant owner is nested `extras.lens.variant` for txt2img or `img2img_extras.lens.variant` for img2img.
Resolves `extras.lora_sha` / `img2img_extras.lora_sha` into server-side `lora_path` overrides only for engines with `supports_lora=True`
and when SHA ownership matches LoRA inventory (`inventory.loras`, `.safetensors`), rejecting unsupported-engine/non-LoRA resolution fail-loud.
Enforces generation settings contracts: top-level `smart_*` payload keys are rejected and `settings_revision` must match persisted options revision.
Uses backend API-owned WAN video request key allowlists from `interfaces/api/wan_video_request_keys.py`,
resolves exact WAN22 5B/14B engine dispatch from payload shape plus metadata/inventory cross-checks, and derives WAN
sampler/scheduler defaults from metadata scheduler assets while validating `gguf_sdpa_policy` (`auto|mem_efficient|flash|math`) fail-loud.
WAN 2.2 5B keeps top-level prompt/negative plus other core lane fields on the request owner while `wan_single` stays selector-only
(`model_sha`, `loras`, `flow_shift`); WAN 2.2 14B keeps `wan_high` selector-only and `wan_low` as the explicit second-stage execution owner.
The generic LTX video route now owns its own request contract (checkpoint-owned `ltx_execution_profile`, `32px` base geometry or `%64` final geometry
for `two_stage`, `8n+1` frames, explicit safe defaults, derived `euler` / `simple`, negative-seed random semantics, required `img2vid_init_image`,
and rejection of WAN-only `*_styles` baggage plus raw LTX `*_sampler` / `*_scheduler` wire keys)
instead of inheriting WAN `%16` / `4n+1` assumptions.
Route-level capability validation now also understands `GenerationRouteMode.VID2VID`, while `/api/vid2vid` itself remains a parked
placeholder route that rejects before any staging/task creation.
FLUX.2 img2img now accepts partial denoise (`img2img_denoising_strength != 1.0`) after backend support landed; masked FLUX.2 hires remains an explicit API reject.
Legacy WAN sampler aliases (`txt2vid_sampling`/`img2vid_sampling`) are rejected; WAN keeps `txt2vid_sampler` and `img2vid_sampler` as its canonical request keys, while LTX derives its fixed runtime lane from explicit `ltx_execution_profile`.
WAN sampler fields are strict at API parse-time: values must resolve to real WAN22 runtime lanes (`uni-pc` with metadata-compatible optional solver hint, `euler`, or `euler a`); scheduler fields remain strict (`simple`) for WAN22 requests.
Img2vid temporal execution now requires explicit `img2vid_mode` (`solo|sliding|svi2|svi2_pro`) with mode-scoped validation for chunk/window fields,
and no-stretch guide controls (`img2vid_image_scale`, `img2vid_crop_offset_x`, `img2vid_crop_offset_y`) are parsed into WAN extras for runtime preprocessing.
Requires a non-empty top-level WAN prompt owner, uses exact lane-owned stage containers (`wan_single` for 5B or `wan_high` + `wan_low` for 14B),
and still requires a non-empty `wan_low.prompt` second-stage prompt for 14B routes; top-level `negative_prompt` remains the 5B/high-stage negative owner
while `wan_low.negative_prompt` is optional. WAN stage LoRAs are provided via `wan_single` / `wan_high` / `wan_low.loras[]`
(frontend parses `<lora:...>` tags) and duplicate stage entries are deduplicated by SHA (last wins).
Video task workers emit optional contract-trace JSONL events (`CODEX_TRACE_CONTRACT=1`) with prompt hashing only (no raw prompt text) and
resolve WAN core dtype overrides from persisted options (`codex_core_compute_dtype`/`codex_core_dtype`) before orchestrator dispatch.
Worker exception paths trigger shared runtime memory cleanup (`tasks/generation_tasks.py::force_runtime_memory_cleanup`) so task failures best-effort purge engine/runtime caches.
Requires explicit per-request device selection and serializes GPU-heavy execution via the shared inference gate when `CODEX_SINGLE_FLIGHT=1` (default on).
Any cancel mode may abort while waiting on the inference gate; in-flight interruption remains `immediate`-only.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for generation endpoints.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import asyncio
import json
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.interfaces.api.path_utils import _path_for_api, _path_from_api
from apps.backend.interfaces.api.inference_gate import acquire_inference_gate, release_inference_gate, single_flight_enabled
from apps.backend.interfaces.api.public_errors import (
    build_cancelled_task_error,
    build_missing_result_task_error,
    build_public_task_error,
    public_http_error_detail,
)
from apps.backend.interfaces.api.task_registry import TaskCancelMode, TaskEntry, register_task, unregister_task

_router_log = get_backend_logger("backend.api.routers.generation")


def build_router(*, codex_root: Path, media, live_preview, opts_get, opts_snapshot, generation_provenance, save_generated_images, param_utils) -> APIRouter:
    router = APIRouter()
    CODEX_ROOT = codex_root
    _GENERATION_PROVENANCE = generation_provenance(codex_root)
    _save_generated_images = save_generated_images
    _opts_get = opts_get
    _opts_snapshot = opts_snapshot
    _p = param_utils

    from apps.backend.core.engine_interface import TaskType
    from apps.backend.core.rng import NoiseSourceKind
    from apps.backend.core.orchestrator import InferenceOrchestrator
    from apps.backend.core.requests import (
        ImageAutomationInitSource,
        ImageAutomationLoopConfig,
        ImageAutomationPromptSource,
        ImageAutomationRequest,
        ImageAutomationSeedPolicy,
        ProgressEvent,
        ResultEvent,
        Txt2ImgRequest,
        Img2ImgRequest,
        Txt2VidRequest,
        Img2VidRequest,
    )
    from apps.backend.interfaces.api.device_selection import (
        GenerationRouteMode,
        generation_route_device_policy,
        parse_device_from_payload,
    )
    from apps.backend.runtime.model_registry.capabilities import (
        ENGINE_SURFACES,
        PARKED_EXACT_ENGINES,
        SemanticEngine,
        engine_supports_cfg,
        ip_adapter_support_error,
        semantic_engine_for_engine_id,
        supir_support_error,
    )
    from apps.backend.runtime.families.qwen_image.config import (
        QWEN_IMAGE_EDIT_VARIANT,
        QWEN_IMAGE_ENGINE_ID,
        QWEN_IMAGE_PUBLIC_SAMPLER,
        QWEN_IMAGE_PUBLIC_SCHEDULER,
        QWEN_IMAGE_VARIANT_KEY,
        qwen_image_edit_condition_dimensions,
        qwen_image_edit_vae_dimensions,
    )
    from apps.backend.runtime.families.lens.config import (
        LENS_ENGINE_ID,
        LENS_EXTRAS_KEY,
        LENS_NOT_IMPLEMENTED_MESSAGE,
        require_lens_variant,
    )
    from apps.backend.runtime.families.supir.config import parse_supir_mode_config
    from apps.backend.runtime.families.supir.errors import SupirBaseModelError, SupirConfigError, SupirWeightsError
    from apps.backend.runtime.families.supir.loader import resolve_supir_assets

    def _ensure_default_engines_registered() -> None:
        # Generation endpoints require the engine registry, but API startup should remain import-light.
        # Register engines lazily so health/models endpoints can work without pulling torch-heavy deps.
        from apps.backend.engines import register_default_engines

        register_default_engines(replace=False)

    from apps.backend.types.payloads import EXTRAS_KEYS, TXT2IMG_KEYS
    from apps.backend.interfaces.api.wan_video_request_keys import (
        WAN_VIDEO_REQUEST_KEYS,
        legacy_wan_video_request_key_alias_target,
    )
    _TXT2IMG_ALLOWED_KEYS = set(TXT2IMG_KEYS.ALL) - set(TXT2IMG_KEYS.SMART)
    _IMAGE_REQUEST_SELECTOR_KEYS = {"checkpoint_core_only", "model_format", "vae_source"}
    _TXT2IMG_EXTRAS_KEYS = set(EXTRAS_KEYS.ALL) | _IMAGE_REQUEST_SELECTOR_KEYS
    _TXT2IMG_HIRES_KEYS = set(TXT2IMG_KEYS.HIRES)
    _IMG2IMG_EXTRAS_KEYS = (
        (set(EXTRAS_KEYS.ALL) | {"supir"}) - {"refiner", "batch_size", "batch_count"}
    ) | _IMAGE_REQUEST_SELECTOR_KEYS
    _IMG2IMG_HIRES_KEYS = {
        "cfg",
        "denoise",
        "distilled_cfg",
        "enable",
        "negative_prompt",
        "prompt",
        "resize_x",
        "resize_y",
        "sampler",
        "scale",
        "scheduler",
        "steps",
        "tile",
        "upscaler",
    }
    _IMG2IMG_UNSUPPORTED_HIRES_KEYS = {"modules", "refiner", "swap_model"}
    _IMAGE_AUTOMATION_ALLOWED_KEYS = {"mode", "template", "loop", "seed_policy", "prompt_source", "init_source"}
    _IMAGE_AUTOMATION_LOOP_KEYS = {"mode", "count", "delay_ms", "stop_on_error"}
    _IMAGE_AUTOMATION_SEED_POLICY_KEYS = {"mode", "increment_step"}
    _IMAGE_AUTOMATION_PROMPT_SOURCE_KEYS = {"kind", "text", "insert_position", "wildcard_root", "wildcard_mode"}
    _IMAGE_AUTOMATION_INIT_SOURCE_KEYS = {"kind", "folder_path", "selection_mode", "count", "order", "sort_by", "use_crop"}
    _IP_ADAPTER_KEYS = {"enabled", "model", "image_encoder", "weight", "start_at", "end_at", "source"}
    _IP_ADAPTER_SOURCE_KEYS = {"kind", "reference_image_data", "folder_path", "selection_mode", "count", "order", "sort_by"}
    _IMG2IMG_PIXEL_RESIZE_MODES = {"just_resize", "crop_and_resize", "resize_and_fill"}
    _QWEN_IMAGE_ENGINE_ID = QWEN_IMAGE_ENGINE_ID
    _QWEN_IMAGE_EDIT_VARIANT = QWEN_IMAGE_EDIT_VARIANT
    _QWEN_IMAGE_VARIANT_KEY = QWEN_IMAGE_VARIANT_KEY
    _QWEN_IMAGE_SAMPLER = QWEN_IMAGE_PUBLIC_SAMPLER
    _QWEN_IMAGE_SCHEDULER = QWEN_IMAGE_PUBLIC_SCHEDULER
    _LENS_ENGINE_ID = LENS_ENGINE_ID
    _LENS_EXTRAS_KEY = LENS_EXTRAS_KEY
    _LENS_NOT_IMPLEMENTED_MESSAGE = LENS_NOT_IMPLEMENTED_MESSAGE
    _ZIMAGE_L2P_ENGINE_ID = "zimage_l2p"
    _ZIMAGE_L2P_MODEL_FORMATS = {"checkpoint", "gguf"}
    _ZIMAGE_L2P_TXT2IMG_REJECTED_EXTRAS_KEYS = {
        "er_sde",
        "guidance",
        "hires",
        "ip_adapter",
        "lora_sha",
        "refiner",
        "swap_model",
        "tenc1_sha",
        "tenc2_sha",
        "text_encoder_override",
        "vae_sha",
        "vae_source",
        "zimage_variant",
    }
    _QWEN_IMAGE_ASSET_EXTRAS_KEYS = {
        "checkpoint_core_only",
        "model_format",
        "model_sha",
        "tenc_sha",
        "vae_sha",
        "vae_source",
    }
    _QWEN_IMAGE_IMG2IMG_ALLOWED_KEYS = {
        "device",
        "engine",
        "img2img_batch_count",
        "img2img_batch_size",
        "img2img_cfg_scale",
        "img2img_extras",
        "img2img_init_image",
        "img2img_neg_prompt",
        "img2img_prompt",
        "img2img_sampling",
        "img2img_scheduler",
        "img2img_seed",
        "img2img_steps",
        "img2img_styles",
        "settings_revision",
    }
    _QWEN_IMAGE_IMG2IMG_REQUIRED_KEYS = (
        "device",
        "engine",
        "img2img_cfg_scale",
        "img2img_extras",
        "img2img_init_image",
        "img2img_neg_prompt",
        "img2img_prompt",
        "img2img_sampling",
        "img2img_scheduler",
        "img2img_seed",
        "img2img_steps",
        "settings_revision",
    )
    _QWEN_IMAGE_IMG2IMG_REJECTED_TOP_LEVEL_KEYS = {
        "img2img_clip_skip",
        "img2img_denoising_strength",
        "img2img_distilled_cfg_scale",
        "img2img_eta_noise_seed_delta",
        "img2img_height",
        "img2img_image_cfg_scale",
        "img2img_inpaint_full_res_padding",
        "img2img_inpainting_fill",
        "img2img_inpainting_mask_invert",
        "img2img_inpaint_mode",
        "img2img_mask",
        "img2img_mask_blur",
        "img2img_mask_blur_x",
        "img2img_mask_blur_y",
        "img2img_mask_enforcement",
        "img2img_mask_region_split",
        "img2img_mask_round",
        "img2img_noise_source",
        "img2img_per_step_blend_steps",
        "img2img_per_step_blend_strength",
        "img2img_resize_mode",
        "img2img_width",
    }
    _IMG2IMG_ALLOWED_KEYS = {
        "device",
        "engine",
        "img2img_batch_count",
        "img2img_batch_size",
        "img2img_cfg_scale",
        "img2img_clip_skip",
        "img2img_denoising_strength",
        "img2img_distilled_cfg_scale",
        "img2img_eta_noise_seed_delta",
        "img2img_extras",
        "img2img_height",
        "img2img_image_cfg_scale",
        "img2img_init_image",
        "img2img_inpaint_full_res_padding",
        "img2img_inpainting_fill",
        "img2img_inpainting_mask_invert",
        "img2img_mask",
        "img2img_mask_enforcement",
        "img2img_mask_blur",
        "img2img_mask_blur_x",
        "img2img_mask_blur_y",
        "img2img_per_step_blend_strength",
        "img2img_per_step_blend_steps",
        "img2img_inpaint_mode",
        "img2img_mask_region_split",
        "img2img_mask_round",
        "img2img_neg_prompt",
        "img2img_noise_source",
        "img2img_prompt",
        "img2img_resize_mode",
        "img2img_sampling",
        "img2img_scheduler",
        "img2img_seed",
        "img2img_steps",
        "img2img_styles",
        "img2img_width",
        "model",
        "settings_revision",
    }
    _TXT2VID_ALLOWED_KEYS = set(WAN_VIDEO_REQUEST_KEYS.TXT2VID_ALL)
    _IMG2VID_ALLOWED_KEYS = set(WAN_VIDEO_REQUEST_KEYS.IMG2VID_ALL)
    _LTX2_EXECUTION_PROFILE_KEY = "ltx_execution_profile"
    _LTX2_GENERIC_BLOCKED_KEYS = frozenset(
        {
            "txt2vid_sampler",
            "txt2vid_scheduler",
            "txt2vid_styles",
            "img2vid_sampler",
            "img2vid_scheduler",
            "img2vid_styles",
        }
    )
    _VIDEO_GENERIC_SELECTOR_KEYS = {"engine", "model", "model_sha", "vae_sha", "tenc_sha", "lora_sha"}
    _VIDEO_GENERIC_COMMON_ALLOWED_KEYS = (
        set(WAN_VIDEO_REQUEST_KEYS.DEVICE)
        | set(WAN_VIDEO_REQUEST_KEYS.REVISION)
        | set(WAN_VIDEO_REQUEST_KEYS.VIDEO_EXPORT)
        | set(WAN_VIDEO_REQUEST_KEYS.VIDEO_INTERPOLATION)
        | set(WAN_VIDEO_REQUEST_KEYS.VIDEO_UPSCALING)
        | set(WAN_VIDEO_REQUEST_KEYS.GGUF_RUNTIME)
        | _VIDEO_GENERIC_SELECTOR_KEYS
    )
    _TXT2VID_GENERIC_ALLOWED_KEYS = _VIDEO_GENERIC_COMMON_ALLOWED_KEYS | set(WAN_VIDEO_REQUEST_KEYS.TXT2VID)
    _LTX2_TXT2VID_GENERIC_ALLOWED_KEYS = (_TXT2VID_GENERIC_ALLOWED_KEYS - _LTX2_GENERIC_BLOCKED_KEYS) | {
        _LTX2_EXECUTION_PROFILE_KEY
    }
    _IMG2VID_GENERIC_CORE_KEYS = {
        "img2vid_prompt",
        "img2vid_neg_prompt",
        "img2vid_width",
        "img2vid_height",
        "img2vid_steps",
        "img2vid_fps",
        "img2vid_num_frames",
        "img2vid_sampler",
        "img2vid_scheduler",
        "img2vid_seed",
        "img2vid_cfg_scale",
        "img2vid_styles",
        "img2vid_init_image",
    }
    _IMG2VID_GENERIC_ALLOWED_KEYS = _VIDEO_GENERIC_COMMON_ALLOWED_KEYS | _IMG2VID_GENERIC_CORE_KEYS
    _LTX2_IMG2VID_GENERIC_ALLOWED_KEYS = (_IMG2VID_GENERIC_ALLOWED_KEYS - _LTX2_GENERIC_BLOCKED_KEYS) | {
        _LTX2_EXECUTION_PROFILE_KEY
    }
    _WAN_SINGLE_ALLOWED_KEYS = set(WAN_VIDEO_REQUEST_KEYS.WAN_SINGLE_ALLOWED)
    _WAN_HIGH_ALLOWED_KEYS = set(WAN_VIDEO_REQUEST_KEYS.WAN_HIGH_ALLOWED)
    _WAN_LOW_ALLOWED_KEYS = set(WAN_VIDEO_REQUEST_KEYS.WAN_LOW_ALLOWED)
    _WAN_STAGE_LORA_ALLOWED_KEYS = {"sha", "weight"}
    _ER_SDE_OPTION_KEYS = {"solver_type", "max_stage", "eta", "s_noise"}
    _GUIDANCE_OPTION_KEYS = {
        "apg_enabled",
        "apg_start_step",
        "apg_eta",
        "apg_momentum",
        "apg_norm_threshold",
        "apg_rescale",
        "guidance_rescale",
        "cfg_trunc_ratio",
        "renorm_cfg",
    }
    _WAN_PROMPT_LORA_TAG_RE = re.compile(r"<\s*lora\s*:", re.IGNORECASE)
    from apps.backend.runtime.vision.upscalers.specs import tile_config_from_payload

    _NOISE_SOURCE_VALUES = tuple(member.value for member in NoiseSourceKind)

    def _reject_unknown_keys(obj: Mapping[str, Any], allowed: set[str], context: str) -> None:
        unknown = sorted(set(obj.keys()) - allowed)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "unknown_request_keys",
                    "message": f"Unexpected {context} key(s): {', '.join(unknown)}",
                    "context": context,
                    "unknown_keys": unknown,
                },
            )

    def _reject_removed_txt2img_hires_modules_from_hires(hires: object) -> None:
        if isinstance(hires, Mapping) and "modules" in hires:
            raise HTTPException(
                status_code=400,
                detail="'extras.hires.modules' has been removed because hires modules have no execution owner.",
            )

    def _validate_txt2img_hires_request_payload(payload: Mapping[str, Any]) -> None:
        extras = payload.get("extras")
        if extras is None:
            return
        if not isinstance(extras, Mapping):
            raise HTTPException(status_code=400, detail="'extras' must be an object")
        hires = extras.get("hires")
        if hires is None:
            return
        if not isinstance(hires, Mapping):
            raise HTTPException(status_code=400, detail="'extras.hires' must be an object")
        _reject_removed_txt2img_hires_modules_from_hires(hires)
        _reject_unknown_keys(hires, _TXT2IMG_HIRES_KEYS, "extras.hires")

    def _require_nested_fields(payload: Mapping[str, Any], required_keys: Sequence[str], *, context: str) -> None:
        for required_key in required_keys:
            if required_key not in payload or payload.get(required_key) is None:
                raise HTTPException(status_code=400, detail=f"Missing '{context}.{required_key}'")

    def _reject_legacy_wan_request_key_aliases(payload: Mapping[str, Any], *, context: str) -> None:
        aliases: dict[str, str] = {}
        for raw_key in payload.keys():
            if not isinstance(raw_key, str):
                continue
            canonical = legacy_wan_video_request_key_alias_target(raw_key)
            if canonical is None:
                continue
            aliases[raw_key] = canonical
        if not aliases:
            return
        ordered = sorted(aliases.items())
        raise HTTPException(
            status_code=400,
            detail={
                "error": "legacy_request_key_alias",
                "message": (
                    f"Legacy {context} key alias(es) are unsupported. "
                    "Use canonical request keys."
                ),
                "context": context,
                "legacy_keys": [alias for alias, _ in ordered],
                "canonical_keys": [canonical for _, canonical in ordered],
                "replacements": {
                    alias: canonical
                    for alias, canonical in ordered
                },
            },
        )

    def _current_settings_revision() -> int:
        snapshot = _opts_snapshot()
        revision_raw = getattr(snapshot, "codex_options_revision", 0)
        if isinstance(revision_raw, bool) or not isinstance(revision_raw, (int, float)):
            raise RuntimeError(
                "Invalid options snapshot: 'codex_options_revision' must be numeric "
                f"(got {type(revision_raw).__name__})."
            )
        if isinstance(revision_raw, float):
            if not revision_raw.is_integer():
                raise RuntimeError(
                    "Invalid options snapshot: 'codex_options_revision' must be an integer "
                    f"(got {revision_raw!r})."
                )
            revision = int(revision_raw)
        else:
            revision = int(revision_raw)
        return max(0, revision)

    def _enforce_generation_settings_contract(payload: Mapping[str, Any]) -> int:
        payload_obj = payload if isinstance(payload, dict) else dict(payload)
        smart_keys = sorted(k for k in payload_obj if isinstance(k, str) and k.startswith("smart_"))
        if smart_keys:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unexpected generation key(s): {', '.join(smart_keys)}. "
                    "Smart flags are configured only via /api/options."
                ),
            )

        if "settings_revision" not in payload_obj:
            raise HTTPException(status_code=400, detail="Missing 'settings_revision'")
        provided_raw = payload_obj.get("settings_revision")
        if isinstance(provided_raw, bool) or not isinstance(provided_raw, (int, float)):
            raise HTTPException(status_code=400, detail="'settings_revision' must be an integer")
        if isinstance(provided_raw, float):
            if not provided_raw.is_integer():
                raise HTTPException(status_code=400, detail="'settings_revision' must be an integer")
            provided_revision = int(provided_raw)
        else:
            provided_revision = int(provided_raw)
        if provided_revision < 0:
            raise HTTPException(status_code=400, detail="'settings_revision' must be >= 0")

        current_revision = _current_settings_revision()
        if provided_revision != current_revision:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "settings_revision_conflict",
                    "message": "Generation settings_revision does not match persisted options revision.",
                    "current_revision": current_revision,
                    "provided_revision": provided_revision,
                },
            )
        return provided_revision

    def _resolve_smart_flags() -> Tuple[bool, bool, bool]:
        """Resolve effective smart flags from persisted options only.

        Contract:
        - Persisted options are the single source of truth.
        - Generation payload must not include smart_* keys.
        """
        snap = _opts_snapshot()
        smart_offload = _require_options_bool(snap, "codex_smart_offload")
        smart_fallback = _require_options_bool(snap, "codex_smart_fallback")
        smart_cache = _require_options_bool(snap, "codex_smart_cache")
        return smart_offload, smart_fallback, smart_cache


    def _require_str_field(payload: Dict[str, Any], key: str, *, allow_empty: bool = False, trim: bool = True) -> str:
        if key not in payload:
            raise HTTPException(status_code=400, detail=f"Missing '{key}'")
        value = payload[key]
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a string")
        result = value.strip() if trim else value
        if not allow_empty and result == "":
            raise HTTPException(status_code=400, detail=f"'{key}' must not be empty")
        return result if trim else value


    def _require_int_field(payload: Dict[str, Any], key: str, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
        if key not in payload:
            raise HTTPException(status_code=400, detail=f"Missing '{key}'")
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{key}' must be an integer")
        if isinstance(value, float):
            if not value.is_integer():
                raise HTTPException(status_code=400, detail=f"'{key}' must be an integer")
            value = int(value)
        else:
            value = int(value)
        if minimum is not None and value < minimum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be >= {minimum}")
        if maximum is not None and value > maximum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be <= {maximum}")
        return value


    def _snap_dimension(
        value: int,
        *,
        minimum: int = 8,
        maximum: int = 8192,
        multiple: int = 8,
        strategy: str = "nearest",
    ) -> int:
        if not value:
            return 0
        clamped = max(minimum, min(maximum, int(value)))
        if multiple <= 1:
            return clamped
        if strategy == "floor":
            snapped = int((clamped // multiple) * multiple)
        else:
            snapped = int(((clamped + (multiple // 2)) // multiple) * multiple)
        return max(minimum, min(maximum, snapped))


    def _img2img_dimension_multiple_for_engine(engine_key: str) -> int:
        return 16 if engine_key == "zimage" else 8


    def _normalize_img2img_dimensions_for_engine(
        engine_key: str,
        width: int,
        height: int,
    ) -> Tuple[int, int]:
        multiple = _img2img_dimension_multiple_for_engine(engine_key)
        strategy = "floor" if engine_key == "zimage" else "nearest"
        minimum = multiple if engine_key == "zimage" else 8
        return (
            _snap_dimension(width, minimum=minimum, multiple=multiple, strategy=strategy),
            _snap_dimension(height, minimum=minimum, multiple=multiple, strategy=strategy),
        )


    def _parse_img2img_resize_mode(payload: Dict[str, Any]) -> Optional[str]:
        if "img2img_resize_mode" not in payload:
            return None
        resize_mode = _require_str_field(payload, "img2img_resize_mode")
        if resize_mode not in _IMG2IMG_PIXEL_RESIZE_MODES:
            allowed = ", ".join(sorted(_IMG2IMG_PIXEL_RESIZE_MODES))
            raise HTTPException(
                status_code=400,
                detail=f"'img2img_resize_mode' must be one of: {allowed}",
            )
        return resize_mode


    def _require_float_field(payload: Dict[str, Any], key: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
        if key not in payload:
            raise HTTPException(status_code=400, detail=f"Missing '{key}'")
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a finite number")
        if minimum is not None and result < minimum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be >= {minimum}")
        if maximum is not None and result > maximum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be <= {maximum}")
        return result


    def _require_sha256_field(payload: Mapping[str, Any], key: str) -> str:
        value = payload.get(key)
        if isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a string sha256, got object")
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(status_code=400, detail=f"'{key}' is required and must be a non-empty sha256 string")
        normalized = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise HTTPException(status_code=400, detail=f"'{key}' must be sha256 (64 lowercase hex)")
        return normalized

    def _merge_wan_stage_loras(*segments: list[dict[str, object]]) -> list[dict[str, object]]:
        merged: list[dict[str, object]] = []
        index_by_sha: dict[str, int] = {}
        for segment in segments:
            for entry in segment:
                sha_raw = str(entry.get("sha") or "").strip().lower()
                if not re.fullmatch(r"[0-9a-f]{64}", sha_raw):
                    raise HTTPException(status_code=400, detail="WAN stage LoRA entry has invalid sha256")
                weight_raw = entry.get("weight", 1.0)
                if isinstance(weight_raw, bool) or not isinstance(weight_raw, (int, float)):
                    raise HTTPException(status_code=400, detail="WAN stage LoRA entry has non-numeric weight")
                weight = float(weight_raw)
                if not math.isfinite(weight):
                    raise HTTPException(status_code=400, detail="WAN stage LoRA entry has non-finite weight")
                normalized_entry = {"sha": sha_raw, "weight": weight}
                existing_index = index_by_sha.get(sha_raw)
                if isinstance(existing_index, int):
                    merged[existing_index] = normalized_entry
                    continue
                index_by_sha[sha_raw] = len(merged)
                merged.append(normalized_entry)
        return merged

    def _parse_wan_stage_prompt_loras(
        *,
        stage_key: str,
        prompt: str,
        negative_prompt: str | None,
    ) -> tuple[str, str | None, list[dict[str, object]]]:
        if _WAN_PROMPT_LORA_TAG_RE.search(prompt):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{stage_key}.prompt' must not contain '<lora:...>' tags; "
                    f"use '{stage_key}.loras[]' with sha/weight entries."
                ),
            )
        if negative_prompt is not None and _WAN_PROMPT_LORA_TAG_RE.search(negative_prompt):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{stage_key}.negative_prompt' must not contain '<lora:...>' tags; "
                    f"use '{stage_key}.loras[]' with sha/weight entries."
                ),
            )
        return prompt, negative_prompt, []

    def _parse_wan_request_prompt_loras(
        *,
        prompt_field_name: str,
        negative_prompt_field_name: str,
        lora_owner_field_name: str,
        prompt: str,
        negative_prompt: str | None,
    ) -> tuple[str, str | None]:
        if _WAN_PROMPT_LORA_TAG_RE.search(prompt):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{prompt_field_name}' must not contain '<lora:...>' tags; "
                    f"use '{lora_owner_field_name}.loras[]' with sha/weight entries."
                ),
            )
        if negative_prompt is not None and _WAN_PROMPT_LORA_TAG_RE.search(negative_prompt):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{negative_prompt_field_name}' must not contain '<lora:...>' tags; "
                    f"use '{lora_owner_field_name}.loras[]' with sha/weight entries."
                ),
            )
        return prompt, negative_prompt

    def _normalize_wan_stage_loras(
        *,
        stage_raw: Mapping[str, Any],
        stage_key: str,
        resolve_asset_by_sha_fn: Callable[[str], object | None],
    ) -> list[dict[str, object]]:
        if stage_raw.get("lora_path") not in (None, ""):
            raise HTTPException(
                status_code=400,
                detail=f"'{stage_key}.lora_path' is unsupported; use '{stage_key}.loras'",
            )
        if stage_raw.get("lora_sha") not in (None, ""):
            raise HTTPException(
                status_code=400,
                detail=f"'{stage_key}.lora_sha' is unsupported; use '{stage_key}.loras'",
            )
        if stage_raw.get("lora_weight") not in (None, ""):
            raise HTTPException(
                status_code=400,
                detail=f"'{stage_key}.lora_weight' is unsupported; use '{stage_key}.loras'",
            )
        raw_loras = stage_raw.get("loras")
        if raw_loras is None:
            return []
        if not isinstance(raw_loras, list):
            raise HTTPException(
                status_code=400,
                detail=f"'{stage_key}.loras' must be an array when provided",
            )
        from apps.backend.inventory.scanners.loras import iter_lora_files

        known_lora_paths = {
            os.path.normcase(os.path.realpath(os.path.expanduser(path)))
            for path in iter_lora_files()
        }
        if not known_lora_paths:
            raise HTTPException(
                status_code=409,
                detail=f"'{stage_key}.loras' was provided, but no LoRA assets are available in inventory.",
            )

        normalized_loras: list[dict[str, object]] = []
        for index, raw_lora in enumerate(raw_loras):
            lora_context = f"{stage_key}.loras[{index}]"
            if not isinstance(raw_lora, dict):
                raise HTTPException(status_code=400, detail=f"'{lora_context}' must be an object")
            _reject_unknown_keys(raw_lora, _WAN_STAGE_LORA_ALLOWED_KEYS, lora_context)
            lora_sha = _require_sha256_field(raw_lora, "sha")
            lora_path = resolve_asset_by_sha_fn(lora_sha)
            if not lora_path:
                raise HTTPException(status_code=409, detail=f"WAN stage LoRA not found for sha: {lora_sha}")
            if not str(lora_path).lower().endswith(".safetensors"):
                raise HTTPException(
                    status_code=409,
                    detail=f"WAN stage LoRA sha must resolve to a .safetensors file: {lora_sha}",
                )
            canonical_lora_path = os.path.normcase(
                os.path.realpath(os.path.expanduser(str(lora_path)))
            )
            if canonical_lora_path not in known_lora_paths:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"'{lora_context}.sha' resolved to a non-LoRA asset path: {lora_path}. "
                        "Select a SHA from inventory.loras."
                    ),
                )
            raw_weight = raw_lora.get("weight")
            if raw_weight is None:
                lora_weight = 1.0
            else:
                if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                    raise HTTPException(
                        status_code=400,
                        detail=f"'{lora_context}.weight' must be numeric when provided",
                    )
                lora_weight = float(raw_weight)
                if not math.isfinite(lora_weight):
                    raise HTTPException(
                        status_code=400,
                        detail=f"'{lora_context}.weight' must be finite",
                    )
            normalized_loras.append({"sha": lora_sha, "weight": lora_weight})
        return _merge_wan_stage_loras(normalized_loras)

    def _reject_legacy_wan_stage_lora_keys(*, stage_key: str, stage_raw: Mapping[str, Any]) -> None:
        legacy_messages = {
            "lora_path": f"'{stage_key}.lora_path' is unsupported; use '{stage_key}.loras'",
            "lora_sha": f"'{stage_key}.lora_sha' is unsupported; use '{stage_key}.loras'",
            "lora_weight": f"'{stage_key}.lora_weight' is unsupported; use '{stage_key}.loras'",
        }
        for key, detail in legacy_messages.items():
            if stage_raw.get(key) not in (None, ""):
                raise HTTPException(status_code=400, detail=detail)


    def _require_bool_field(payload: Dict[str, Any], key: str) -> bool:
        if key not in payload:
            raise HTTPException(status_code=400, detail=f"Missing '{key}'")
        value = payload[key]
        if not isinstance(value, bool):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a boolean")
        return value


    def _optional_bool_field(payload: Dict[str, Any], key: str) -> Optional[bool]:
        if key not in payload or payload.get(key) is None:
            return None
        value = payload[key]
        if not isinstance(value, bool):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a boolean")
        return value


    def _parse_optional_bool_selector(
        *,
        payload: Mapping[str, Any],
        key: str,
        field_name: str,
    ) -> Optional[bool]:
        if key not in payload or payload.get(key) is None:
            return None
        value = payload.get(key)
        if not isinstance(value, bool):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be a boolean")
        return value


    def _parse_optional_vae_source_selector(
        *,
        payload: Mapping[str, Any],
        key: str,
        field_name: str,
    ) -> Optional[str]:
        if key not in payload or payload.get(key) is None:
            return None
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' must be 'built_in' or 'external'",
            )
        normalized = value.strip().lower()
        if normalized not in {"built_in", "external"}:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' must be 'built_in' or 'external'",
            )
        return normalized

    def _parse_optional_model_format_selector(
        *,
        payload: Mapping[str, Any],
        key: str,
        field_name: str,
    ) -> Optional[str]:
        if key not in payload or payload.get(key) is None:
            return None
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' must be one of: checkpoint, diffusers, gguf",
            )
        normalized = value.strip().lower()
        if normalized not in {"checkpoint", "diffusers", "gguf"}:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' must be one of: checkpoint, diffusers, gguf",
            )
        return normalized

    def _parse_optional_zimage_variant_selector(
        *,
        payload: Mapping[str, Any],
        key: str,
        field_name: str,
    ) -> Optional[str]:
        if key not in payload or payload.get(key) is None:
            return None
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' must be one of: turbo, base",
            )
        normalized = value.strip().lower()
        if normalized not in {"turbo", "base"}:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' must be one of: turbo, base",
            )
        return normalized

    def _parse_optional_non_negative_int(value: object, *, field_name: str) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if not re.fullmatch(r"[+-]?\d+", text):
                raise HTTPException(status_code=400, detail=f"'{field_name}' must be an integer")
            parsed = int(text)
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be an integer")
        elif isinstance(value, float):
            if not value.is_integer():
                raise HTTPException(status_code=400, detail=f"'{field_name}' must be an integer")
            parsed = int(value)
        else:
            parsed = int(value)
        if parsed < 0:
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be >= 0")
        return parsed

    def _normalize_gguf_cache_controls(extras: Dict[str, Any]) -> None:
        has_policy = "gguf_cache_policy" in extras
        has_limit = "gguf_cache_limit_mb" in extras
        if not has_policy and not has_limit:
            return

        policy: Optional[str] = None
        if has_policy:
            raw_policy = extras.get("gguf_cache_policy")
            if not isinstance(raw_policy, str):
                raise HTTPException(status_code=400, detail="'gguf_cache_policy' must be a string")
            policy_raw = raw_policy.strip().lower()
            if policy_raw in {"", "none", "off"}:
                policy = "none"
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid gguf_cache_policy: {raw_policy!r} (expected 'none'|'off').",
                )

        limit_mb = (
            _parse_optional_non_negative_int(extras.get("gguf_cache_limit_mb"), field_name="gguf_cache_limit_mb")
            if has_limit
            else None
        )

        if policy is None and limit_mb is not None:
            raise HTTPException(
                status_code=400,
                detail="'gguf_cache_limit_mb' requires 'gguf_cache_policy'.",
            )
        if policy == "none" and limit_mb not in (None, 0):
            raise HTTPException(
                status_code=400,
                detail="'gguf_cache_limit_mb' must be omitted or 0 when 'gguf_cache_policy' is 'none' or 'off'.",
            )

        if has_policy:
            extras["gguf_cache_policy"] = policy
        if has_limit and limit_mb is not None:
            extras["gguf_cache_limit_mb"] = int(limit_mb)

    def _normalize_gguf_runtime_controls(extras: Dict[str, Any]) -> None:
        if "gguf_offload" in extras:
            offload_raw = extras.get("gguf_offload")
            if not isinstance(offload_raw, bool):
                raise HTTPException(status_code=400, detail="'gguf_offload' must be a boolean")

        for key in ("gguf_offload_level", "gguf_attn_chunk", "gguf_log_mem_interval"):
            if key not in extras:
                continue
            parsed = _parse_optional_non_negative_int(extras.get(key), field_name=key)
            if parsed is None:
                extras.pop(key, None)
            else:
                extras[key] = int(parsed)

    def _normalize_gguf_te_device(extras: Dict[str, Any]) -> None:
        if "gguf_te_device" not in extras:
            return
        raw_value = extras.get("gguf_te_device")
        if not isinstance(raw_value, str):
            raise HTTPException(status_code=400, detail="'gguf_te_device' must be a string")
        normalized = raw_value.strip().lower()
        if normalized == "gpu":
            normalized = "cuda"
        if normalized in {"cpu", "cuda", "auto"} or re.fullmatch(r"cuda:\d+", normalized):
            extras["gguf_te_device"] = normalized
            return
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid gguf_te_device: "
                f"{raw_value!r} (expected 'auto', 'cpu', 'cuda', or 'cuda:<index>')."
            ),
        )

    def _apply_gguf_video_runtime_controls_from_payload(*, payload: Mapping[str, Any], extras: Dict[str, Any]) -> None:
        for key in (
            "gguf_offload",
            "gguf_offload_level",
            "gguf_sdpa_policy",
            "gguf_attention_mode",
            "gguf_attn_chunk",
            "gguf_cache_policy",
            "gguf_cache_limit_mb",
            "gguf_log_mem_interval",
            "gguf_te_device",
        ):
            if key in payload and payload.get(key) is not None:
                extras[key] = payload.get(key)
        if 'gguf_attention_mode' in extras:
            attn_mode = str(extras.get('gguf_attention_mode') or '').strip().lower()
            if attn_mode not in {'global', 'sliding'}:
                raise HTTPException(status_code=400, detail=f"Invalid gguf_attention_mode: {extras.get('gguf_attention_mode')!r}")
            extras['gguf_attention_mode'] = attn_mode
        if 'gguf_sdpa_policy' in extras:
            sdpa_policy = str(extras.get('gguf_sdpa_policy') or '').strip().lower()
            if sdpa_policy not in {'auto', 'mem_efficient', 'flash', 'math'}:
                raise HTTPException(status_code=400, detail=f"Invalid gguf_sdpa_policy: {extras.get('gguf_sdpa_policy')!r}")
            extras['gguf_sdpa_policy'] = sdpa_policy
        _normalize_gguf_runtime_controls(extras)
        _normalize_gguf_te_device(extras)
        _normalize_gguf_cache_controls(extras)

    def _optional_video_interpolation_field(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "video_interpolation" not in payload:
            return None
        raw = payload.get("video_interpolation")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="'video_interpolation' must be an object when provided")
        _reject_unknown_keys(raw, {"enabled", "model", "times"}, "video_interpolation")

        if "enabled" not in raw:
            raise HTTPException(status_code=400, detail="'video_interpolation.enabled' is required when video_interpolation is provided")
        enabled = raw.get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(status_code=400, detail="'video_interpolation.enabled' must be a boolean")

        normalized: Dict[str, Any] = {"enabled": enabled}
        model_raw = raw.get("model")
        if model_raw is not None:
            if not isinstance(model_raw, str):
                raise HTTPException(status_code=400, detail="'video_interpolation.model' must be a string when provided")
            model = model_raw.strip()
            normalized["model"] = model if model else None

        times_raw = raw.get("times")
        if times_raw is not None:
            if isinstance(times_raw, bool) or not isinstance(times_raw, int):
                raise HTTPException(status_code=400, detail="'video_interpolation.times' must be an integer when provided")
            times_value = int(times_raw)
            if times_value < 2:
                raise HTTPException(status_code=400, detail="'video_interpolation.times' must be >= 2 when provided")
            normalized["times"] = times_value

        return normalized

    _VIDEO_UPSCALING_COLOR_CORRECTIONS = {
        "lab",
        "wavelet",
        "wavelet_adaptive",
        "hsv",
        "adain",
        "none",
    }

    def _optional_video_upscaling_field(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "video_upscaling" not in payload:
            return None
        raw = payload.get("video_upscaling")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="'video_upscaling' must be an object when provided")
        _reject_unknown_keys(
            raw,
            {
                "enabled",
                "dit_model",
                "resolution",
                "max_resolution",
                "batch_size",
                "uniform_batch_size",
                "temporal_overlap",
                "prepend_frames",
                "color_correction",
                "input_noise_scale",
                "latent_noise_scale",
            },
            "video_upscaling",
        )

        if "enabled" not in raw:
            raise HTTPException(
                status_code=400,
                detail="'video_upscaling.enabled' is required when video_upscaling is provided",
            )
        enabled = raw.get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(status_code=400, detail="'video_upscaling.enabled' must be a boolean")

        normalized: Dict[str, Any] = {"enabled": enabled}

        def _optional_int(field: str, *, minimum: int) -> None:
            if field not in raw:
                return
            value = raw.get(field)
            if value is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"'video_upscaling.{field}' must be an integer when provided",
                )
            if isinstance(value, bool) or not isinstance(value, int):
                raise HTTPException(
                    status_code=400,
                    detail=f"'video_upscaling.{field}' must be an integer when provided",
                )
            parsed = int(value)
            if parsed < minimum:
                raise HTTPException(
                    status_code=400,
                    detail=f"'video_upscaling.{field}' must be >= {minimum} when provided",
                )
            normalized[field] = parsed

        _optional_int("resolution", minimum=16)
        _optional_int("max_resolution", minimum=0)
        _optional_int("batch_size", minimum=1)
        _optional_int("temporal_overlap", minimum=0)
        _optional_int("prepend_frames", minimum=0)

        batch_size = normalized.get("batch_size")
        if isinstance(batch_size, int) and ((batch_size - 1) % 4 != 0):
            raise HTTPException(
                status_code=400,
                detail="'video_upscaling.batch_size' must satisfy 4n+1 when provided",
            )

        if "uniform_batch_size" in raw:
            uniform_raw = raw.get("uniform_batch_size")
            if not isinstance(uniform_raw, bool):
                raise HTTPException(
                    status_code=400,
                    detail="'video_upscaling.uniform_batch_size' must be a boolean when provided",
                )
            normalized["uniform_batch_size"] = uniform_raw

        if "dit_model" in raw:
            model_raw = raw.get("dit_model")
            if not isinstance(model_raw, str):
                raise HTTPException(
                    status_code=400,
                    detail="'video_upscaling.dit_model' must be a string when provided",
                )
            model = model_raw.strip()
            if not model:
                raise HTTPException(
                    status_code=400,
                    detail="'video_upscaling.dit_model' must be a non-empty string when provided",
                )
            normalized["dit_model"] = model

        if "color_correction" in raw:
            color_raw = raw.get("color_correction")
            if not isinstance(color_raw, str):
                raise HTTPException(
                    status_code=400,
                    detail="'video_upscaling.color_correction' must be a string when provided",
                )
            color_value = color_raw.strip().lower()
            if color_value not in _VIDEO_UPSCALING_COLOR_CORRECTIONS:
                allowed = ", ".join(sorted(_VIDEO_UPSCALING_COLOR_CORRECTIONS))
                raise HTTPException(
                    status_code=400,
                    detail=f"'video_upscaling.color_correction' must be one of {{{allowed}}}",
                )
            normalized["color_correction"] = color_value

        def _optional_float(field: str) -> None:
            if field not in raw:
                return
            value = raw.get(field)
            if value is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"'video_upscaling.{field}' must be a number when provided",
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail=f"'video_upscaling.{field}' must be a number when provided",
                )
            parsed = float(value)
            if parsed < 0.0 or parsed > 1.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"'video_upscaling.{field}' must be within [0, 1] when provided",
                )
            normalized[field] = parsed

        _optional_float("input_noise_scale")
        _optional_float("latent_noise_scale")

        return normalized

    def _parse_video_output_options(
        payload: Dict[str, Any],
        *,
        route_label: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        extras_updates: Dict[str, Any] = {}
        if "video_return_frames" in payload:
            raw_return_frames = payload.get("video_return_frames")
            if raw_return_frames is not None and not isinstance(raw_return_frames, bool):
                raise HTTPException(status_code=400, detail="'video_return_frames' must be a boolean when provided")
            if isinstance(raw_return_frames, bool):
                extras_updates["video_return_frames"] = raw_return_frames

        try:
            from apps.backend.core.params.video import VideoExportOptions

            video_options = VideoExportOptions(
                filename_prefix=(str(payload.get("video_filename_prefix")).strip() if payload.get("video_filename_prefix") else None),
                format=(str(payload.get("video_format")).strip() if payload.get("video_format") else None),
                pix_fmt=(str(payload.get("video_pix_fmt")).strip() if payload.get("video_pix_fmt") else None),
                crf=(int(payload.get("video_crf")) if payload.get("video_crf") is not None else None),
                loop_count=(int(payload.get("video_loop_count")) if payload.get("video_loop_count") is not None else None),
                pingpong=_optional_bool_field(payload, "video_pingpong"),
                save_metadata=_optional_bool_field(payload, "video_save_metadata"),
                save_output=_optional_bool_field(payload, "video_save_output"),
                trim_to_audio=_optional_bool_field(payload, "video_trim_to_audio"),
            ).as_dict()
        except HTTPException:
            raise
        except Exception as exc:
            _router_log.warning("%s video export options validation failed: %s", route_label, exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid video export options"),
            ) from exc

        video_interpolation = _optional_video_interpolation_field(payload)
        if video_interpolation is not None:
            extras_updates["video_interpolation"] = video_interpolation
        video_upscaling = _optional_video_upscaling_field(payload)
        if video_upscaling is not None:
            extras_updates["video_upscaling"] = video_upscaling
        return video_options, extras_updates

    def _require_options_bool(options_snapshot: Any, key: str) -> bool:
        value = getattr(options_snapshot, key, False)
        if not isinstance(value, bool):
            raise RuntimeError(f"Invalid options value: '{key}' must be a boolean (got {type(value).__name__}).")
        return value

    _ALLOWED_CORE_DTYPE_CHOICES = {"fp16", "bf16", "fp32"}

    def _normalize_options_dtype_choice(options_snapshot: Any, key: str) -> Optional[str]:
        value = getattr(options_snapshot, key, None)
        if value is None:
            return None
        if not isinstance(value, str):
            raise RuntimeError(f"Invalid options value: '{key}' must be a string (got {type(value).__name__}).")
        normalized = value.strip().lower()
        if normalized in {"", "auto"}:
            return None
        if normalized not in _ALLOWED_CORE_DTYPE_CHOICES:
            raise RuntimeError(
                f"Invalid options value: '{key}' must be one of auto/fp16/bf16/fp32 (got {value!r})."
            )
        return normalized

    def _resolve_core_dtype_overrides(options_snapshot: Any) -> Tuple[Optional[str], Optional[str]]:
        storage_dtype = _normalize_options_dtype_choice(options_snapshot, "codex_core_dtype")
        compute_dtype = _normalize_options_dtype_choice(options_snapshot, "codex_core_compute_dtype")
        return storage_dtype, (compute_dtype if compute_dtype is not None else storage_dtype)


    _PARKED_VID2VID_ROUTE_DETAIL = "/api/vid2vid is parked; no families are implemented yet."
    _TXT2VID_BLANK_PROMPT_DETAIL = "'txt2vid_prompt' must be a non-empty string"
    _IMAGE_AUTOMATION_EMPTY_LIST_DETAIL = "prompt_source.text must include at least one non-empty prompt line."


    def _reject_not_implemented_engine(engine_key: str, *, field_name: str) -> None:
        if engine_key == _LENS_ENGINE_ID:
            raise HTTPException(status_code=501, detail=_parked_lens_detail())
        if engine_key in {"sd35", "netflix_void", "svd", "hunyuan_video"}:
            raise HTTPException(
                status_code=501,
                detail=f"Engine '{field_name}={engine_key}' is not implemented yet.",
            )


    def _resolve_wan_metadata_dir(payload: Dict[str, Any]) -> str:
        """Resolve the WAN metadata directory for GGUF runs.

        Preferred contract: pass `wan_metadata_repo="Org/Repo"` and resolve it under
        `apps/backend/huggingface/` (vendored HF mirror).

        Canonical fallback: accept `wan_metadata_dir` as an explicit path.
        """
        raw_repo = payload.get("wan_metadata_repo")
        if isinstance(raw_repo, str) and raw_repo.strip():
            repo_id = raw_repo.strip()
            if repo_id.count("/") != 1:
                raise HTTPException(status_code=400, detail="'wan_metadata_repo' must be a repo id like 'Org/Repo'")
            org, repo = repo_id.split("/", 1)
            if not org or not repo or org in {".", ".."} or repo in {".", ".."}:
                raise HTTPException(status_code=400, detail="'wan_metadata_repo' must be a repo id like 'Org/Repo'")
            if Path(repo_id).is_absolute():
                raise HTTPException(status_code=400, detail="'wan_metadata_repo' must be a repo id (not a filesystem path)")

            hf_root = (CODEX_ROOT / "apps" / "backend" / "huggingface").resolve()
            local_dir = (hf_root / org / repo).resolve()
            try:
                local_dir.relative_to(hf_root)
            except Exception:
                raise HTTPException(status_code=400, detail="'wan_metadata_repo' resolves outside the vendored HF root")
            if not local_dir.is_dir():
                raise HTTPException(status_code=409, detail=f"WAN metadata repo not found locally: {repo_id}")
            return str(local_dir)

        meta_dir = payload.get("wan_metadata_dir")
        if isinstance(meta_dir, str) and meta_dir.strip():
            try:
                return _path_from_api(meta_dir)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid WAN metadata path: {exc}") from exc

        raise HTTPException(status_code=400, detail="'wan_metadata_repo' (or 'wan_metadata_dir') is required for WAN GGUF")

    _WAN22_ENGINE_HINTS: tuple[tuple[str, str], ...] = (
        ("wan2.2-ti2v-5b-diffusers", "wan22_5b"),
        ("wan2.2-ti2v-5b", "wan22_5b"),
        ("wan2.2-animate-14b-diffusers", "wan22_14b_animate"),
        ("wan2.2-animate-14b", "wan22_14b_animate"),
        ("wan2.2-i2v-a14b-diffusers", "wan22_14b"),
        ("wan2.2-i2v-a14b", "wan22_14b"),
        ("wan2.2-t2v-a14b-diffusers", "wan22_14b"),
        ("wan2.2-t2v-a14b", "wan22_14b"),
    )

    def _engine_key_from_wan_hint(hint: str) -> Optional[str]:
        raw = str(hint or "").strip().lower()
        if not raw:
            return None
        for token, engine_key in _WAN22_ENGINE_HINTS:
            if token in raw:
                return engine_key
        # Fallback heuristics are variant-preserving and must never collapse 14B hints into 5B.
        if "animate" in raw and "14b" in raw:
            return "wan22_14b_animate"
        if "14b" in raw:
            return "wan22_14b"
        if "ti2v" in raw or "5b" in raw:
            return "wan22_5b"
        return None

    def _resolve_wan_sampler_scheduler_defaults_from_assets(metadata_dir: str) -> Tuple[str, str]:
        """Resolve WAN sampler/scheduler defaults from metadata assets.

        Fail loud when required scheduler metadata is missing or invalid.
        """
        vendor_dir = os.path.expanduser(str(metadata_dir or "").strip())
        if not vendor_dir:
            raise HTTPException(status_code=400, detail="WAN metadata directory is required.")

        scheduler_dir = Path(vendor_dir) / "scheduler"
        if not scheduler_dir.is_dir():
            parent_scheduler = Path(vendor_dir).parent / "scheduler"
            if parent_scheduler.is_dir():
                scheduler_dir = parent_scheduler
            else:
                raise HTTPException(
                    status_code=409,
                    detail=f"WAN metadata scheduler directory is missing: {scheduler_dir}",
                )

        config_path = scheduler_dir / "scheduler_config.json"
        if not config_path.is_file():
            config_path = scheduler_dir / "config.json"
        if not config_path.is_file():
            raise HTTPException(
                status_code=409,
                detail=(
                    "WAN metadata scheduler config is missing: "
                    f"expected '{scheduler_dir / 'scheduler_config.json'}' or '{scheduler_dir / 'config.json'}'."
                ),
            )

        try:
            config_raw = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail=f"WAN metadata scheduler config is invalid: {config_path}: {exc}",
            ) from exc
        if not isinstance(config_raw, dict):
            raise HTTPException(
                status_code=409,
                detail=f"WAN metadata scheduler config must be a JSON object: {config_path}",
            )

        class_name = str(config_raw.get("_class_name") or "").strip()
        if class_name == "UniPCMultistepScheduler":
            raw_solver_type = config_raw.get("solver_type")
            if raw_solver_type is None:
                return ("uni-pc", "simple")
            if not isinstance(raw_solver_type, str):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "WAN metadata scheduler config solver_type must be a string when provided: "
                        f"{config_path}: {raw_solver_type!r}"
                    ),
                )
            solver_type = raw_solver_type.strip().lower()
            if not solver_type:
                return ("uni-pc", "simple")
            return (f"uni-pc {solver_type}", "simple")
        if not class_name:
            raise HTTPException(
                status_code=409,
                detail=f"WAN metadata scheduler config missing _class_name: {config_path}",
            )
        raise HTTPException(
            status_code=400,
            detail=(
                f"WAN metadata scheduler {class_name!r} is not supported for WAN22 GGUF requests. "
                "Use metadata with UniPCMultistepScheduler."
            ),
        )

    def _resolve_wan22_engine_key(
        payload: Dict[str, Any],
        *,
        metadata_dir: str,
        task_type: TaskType,
        requested_engine_key: str,
        resolved_stage_paths: tuple[str, ...],
    ) -> Tuple[str, str]:
        from apps.backend.core.exceptions import EngineNotFoundError
        from apps.backend.core.registry import registry as _engine_registry

        del task_type

        has_single = isinstance(payload.get("wan_single"), dict)
        has_high = isinstance(payload.get("wan_high"), dict)
        has_low = isinstance(payload.get("wan_low"), dict)
        if has_single:
            if has_high or has_low:
                raise HTTPException(
                    status_code=400,
                    detail="WAN22 requests must use either 'wan_single' or ('wan_high' + 'wan_low'), not both.",
                )
            candidate = "wan22_5b"
        elif has_high or has_low:
            if not (has_high and has_low):
                missing_stage = "wan_high" if not has_high else "wan_low"
                raise HTTPException(
                    status_code=400,
                    detail=f"WAN22 14B requests must provide both 'wan_high' and 'wan_low' (missing '{missing_stage}').",
                )
            candidate = "wan22_14b"
        else:
            raise HTTPException(
                status_code=400,
                detail="WAN22 requests must include either 'wan_single' or both 'wan_high' and 'wan_low'.",
            )

        if requested_engine_key:
            normalized_requested = str(requested_engine_key).strip().lower()
            if normalized_requested != candidate:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "WAN22 request shape does not match the selected engine. "
                        f"Expected '{candidate}' from payload shape, got engine '{normalized_requested}'."
                    ),
                )

        def _cross_check_signal(*, source: str, raw_value: object | None) -> None:
            if not isinstance(raw_value, str) or not raw_value.strip():
                return
            signal_engine = _engine_key_from_wan_hint(raw_value)
            if signal_engine is None:
                return
            if signal_engine != candidate:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "WAN22 request shape does not match metadata/inventory evidence. "
                        f"Payload shape requires '{candidate}', but {source} indicates '{signal_engine}'."
                    ),
                )

        _cross_check_signal(source="wan_metadata_repo", raw_value=payload.get("wan_metadata_repo"))
        _cross_check_signal(source="wan_metadata_dir", raw_value=metadata_dir)
        for index, stage_path in enumerate(resolved_stage_paths):
            _cross_check_signal(source=f"resolved_stage_path[{index}]", raw_value=stage_path)

        from apps.backend.runtime.model_registry.detectors.wan22 import inspect_wan22_gguf_path
        from apps.backend.runtime.model_registry.specs import ModelFamily

        expected_family = ModelFamily.WAN22_5B if candidate == "wan22_5b" else ModelFamily.WAN22_14B
        for index, stage_path in enumerate(resolved_stage_paths):
            try:
                structural_metadata = inspect_wan22_gguf_path(stage_path)
            except Exception as exc:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "WAN22 request could not structurally inspect the selected GGUF. "
                        f"resolved_stage_path[{index}]={stage_path}: {exc}"
                    ),
                ) from exc
            if structural_metadata.family != expected_family:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "WAN22 request shape does not match the structurally detected GGUF family. "
                        f"Payload shape requires '{candidate}', but resolved_stage_path[{index}] "
                        f"detected '{structural_metadata.family.value}'."
                    ),
                )

        model_index_path = Path(os.path.expanduser(str(metadata_dir))) / "model_index.json"
        if model_index_path.is_file():
            try:
                model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
            except Exception:
                model_index = None
            if isinstance(model_index, dict):
                def _has_component(value: Any) -> bool:
                    if value is None:
                        return False
                    if isinstance(value, (list, tuple)):
                        return any(item is not None for item in value)
                    return True

                class_name = str(model_index.get("_class_name") or "").strip().lower()
                has_transformer_2 = _has_component(model_index.get("transformer_2"))
                has_image_encoder = _has_component(model_index.get("image_encoder"))

                metadata_candidate = None
                if "wananimatepipeline" in class_name or "animate" in str(model_index_path).lower():
                    metadata_candidate = "wan22_14b_animate"
                elif has_image_encoder and not has_transformer_2:
                    metadata_candidate = "wan22_14b_animate"
                elif has_transformer_2:
                    metadata_candidate = "wan22_14b"
                elif model_index.get("expand_timesteps") is not None:
                    metadata_candidate = "wan22_5b"
                if metadata_candidate is not None and metadata_candidate != candidate:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "WAN22 request shape does not match metadata model_index.json evidence. "
                            f"Payload shape requires '{candidate}', but metadata indicates '{metadata_candidate}'."
                        ),
                    )

        try:
            _ensure_default_engines_registered()
        except Exception as exc:
            _router_log.exception("engine registry initialization failed")
            raise HTTPException(
                status_code=500,
                detail=public_http_error_detail(exc, fallback="Engine registry init failed"),
            ) from exc
        try:
            return _engine_registry.get_descriptor(candidate).key, candidate
        except EngineNotFoundError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"WAN engine '{candidate}' is not registered. Verify engine registration for WAN22.",
            ) from exc

    def _resolve_wan_vae_path_from_sha(
        *,
        wan_vae_sha: str,
        metadata_dir: str,
        resolve_asset_by_sha,  # type: ignore[no-untyped-def]
        resolve_vae_path_by_sha,  # type: ignore[no-untyped-def]
    ) -> str:
        vae_path = resolve_vae_path_by_sha(wan_vae_sha)
        if not vae_path:
            non_vae_path = resolve_asset_by_sha(wan_vae_sha)
            if non_vae_path:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"'wan_vae_sha' resolved to a non-VAE asset path: {non_vae_path}. "
                        "Select a SHA from inventory.vaes."
                    ),
                )
            raise HTTPException(status_code=409, detail=f"WAN VAE not found for sha: {wan_vae_sha}")

        resolved_path = os.path.expanduser(str(vae_path))
        if os.path.isfile(resolved_path):
            sibling_dir = os.path.dirname(resolved_path)
            sibling_config = os.path.join(sibling_dir, "config.json")
            meta_root = os.path.expanduser(str(metadata_dir))
            meta_candidates = (
                os.path.join(meta_root, "vae"),
                os.path.join(os.path.dirname(meta_root), "vae"),
            )
            if os.path.isfile(sibling_config):
                return resolved_path
            if any(os.path.isfile(os.path.join(candidate, "config.json")) for candidate in meta_candidates):
                return resolved_path
            raise HTTPException(
                status_code=409,
                detail=(
                    "WAN VAE sha resolved to an invalid file VAE config source (missing config.json): "
                    f"{wan_vae_sha} -> {resolved_path}. "
                    f"Expected sibling config.json or metadata config at '{meta_candidates[0]}/config.json' "
                    f"(or '{meta_candidates[1]}/config.json')."
                ),
            )
        if not os.path.isdir(resolved_path):
            raise HTTPException(
                status_code=409,
                detail=f"WAN VAE asset path not found on disk for sha: {wan_vae_sha} -> {resolved_path}",
            )
        bundle_dir = resolved_path
        config_path = os.path.join(bundle_dir, "config.json")
        if not os.path.isfile(config_path):
            raise HTTPException(
                status_code=409,
                detail=(
                    "WAN VAE sha resolved to an invalid bundle (missing config.json): "
                    f"{wan_vae_sha} -> {resolved_path}. "
                    "Select a VAE bundle directory (or a file inside a directory that contains config.json)."
                ),
            )
        weights_candidates = (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "model.safetensors",
            "model.bin",
            "pytorch_model.bin",
        )
        if not any(os.path.isfile(os.path.join(bundle_dir, name)) for name in weights_candidates):
            raise HTTPException(
                status_code=409,
                detail=(
                    "WAN VAE sha resolved to an invalid bundle (missing weights file): "
                    f"{wan_vae_sha} -> {bundle_dir}. "
                    f"Expected one of {weights_candidates}."
                ),
            )
        return bundle_dir


    def _parse_styles(payload: Dict[str, Any]) -> List[str]:
        raw = payload.get('styles')
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise HTTPException(status_code=400, detail="'styles' must be an array of strings")
        out: List[str] = []
        for entry in raw:
            if not isinstance(entry, str):
                raise HTTPException(status_code=400, detail="'styles' must be an array of strings")
            text = entry.strip()
            if text:
                out.append(text)
        return out

    def _parse_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw = payload.get('metadata')
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="'metadata' must be an object")
        return dict(raw)

    def _parse_noise_source_field(raw: object, *, field_name: str) -> str:
        if not isinstance(raw, str):
            allowed = ", ".join(_NOISE_SOURCE_VALUES)
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be one of: {allowed}")
        normalized = raw.strip().lower()
        if not normalized:
            allowed = ", ".join(_NOISE_SOURCE_VALUES)
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be one of: {allowed}")
        try:
            return NoiseSourceKind.from_string(normalized).value
        except ValueError as exc:
            allowed = ", ".join(_NOISE_SOURCE_VALUES)
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be one of: {allowed}") from exc


    def _normalize_er_sde_solver_type(value: object, *, field_name: str) -> str:
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be a string")
        normalized = value.strip().lower().replace("-", " ").replace("_", " ")
        mapping = {
            "er sde": "er_sde",
            "reverse time sde": "reverse_time_sde",
            "ode": "ode",
        }
        solver_type = mapping.get(normalized)
        if solver_type is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{field_name}' must be one of: ER-SDE, Reverse-time SDE, ODE "
                    "(or canonical tokens: er_sde, reverse_time_sde, ode)"
                ),
            )
        return solver_type


    def _parse_er_sde_options(value: object, *, field_name: str) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be an object")
        _reject_unknown_keys(value, _ER_SDE_OPTION_KEYS, field_name)

        options = dict(value)
        solver_type = _normalize_er_sde_solver_type(
            options.get("solver_type", "er_sde"),
            field_name=f"{field_name}.solver_type",
        )
        max_stage_raw = options.get("max_stage", 3)
        if isinstance(max_stage_raw, bool) or not isinstance(max_stage_raw, (int, float)):
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.max_stage' must be an integer in [1, 3]",
            )
        if isinstance(max_stage_raw, float) and not max_stage_raw.is_integer():
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.max_stage' must be an integer in [1, 3]",
            )
        max_stage = int(max_stage_raw)
        if max_stage < 1 or max_stage > 3:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.max_stage' must be in [1, 3]",
            )

        eta_raw = options.get("eta", 1.0)
        if isinstance(eta_raw, bool) or not isinstance(eta_raw, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{field_name}.eta' must be numeric")
        eta = float(eta_raw)
        if not math.isfinite(eta):
            raise HTTPException(status_code=400, detail=f"'{field_name}.eta' must be finite")
        if eta < 0.0:
            raise HTTPException(status_code=400, detail=f"'{field_name}.eta' must be >= 0")

        s_noise_raw = options.get("s_noise", 1.0)
        if isinstance(s_noise_raw, bool) or not isinstance(s_noise_raw, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{field_name}.s_noise' must be numeric")
        s_noise = float(s_noise_raw)
        if not math.isfinite(s_noise):
            raise HTTPException(status_code=400, detail=f"'{field_name}.s_noise' must be finite")
        if s_noise < 0.0:
            raise HTTPException(status_code=400, detail=f"'{field_name}.s_noise' must be >= 0")

        if solver_type == "ode" or (solver_type == "reverse_time_sde" and eta == 0.0):
            eta = 0.0
            s_noise = 0.0

        return {
            "solver_type": solver_type,
            "max_stage": int(max_stage),
            "eta": float(eta),
            "s_noise": float(s_noise),
        }


    def _parse_guidance_options(value: object, *, field_name: str) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be an object")
        _reject_unknown_keys(value, _GUIDANCE_OPTION_KEYS, field_name)
        options = dict(value)
        parsed: Dict[str, Any] = {}

        if "apg_enabled" in options:
            apg_enabled = options.get("apg_enabled")
            if not isinstance(apg_enabled, bool):
                raise HTTPException(status_code=400, detail=f"'{field_name}.apg_enabled' must be a boolean")
            parsed["apg_enabled"] = apg_enabled

        if "apg_start_step" in options:
            start_step = options.get("apg_start_step")
            if isinstance(start_step, bool) or not isinstance(start_step, (int, float)):
                raise HTTPException(status_code=400, detail=f"'{field_name}.apg_start_step' must be an integer >= 0")
            if isinstance(start_step, float) and not start_step.is_integer():
                raise HTTPException(status_code=400, detail=f"'{field_name}.apg_start_step' must be an integer >= 0")
            start_step_i = int(start_step)
            if start_step_i < 0:
                raise HTTPException(status_code=400, detail=f"'{field_name}.apg_start_step' must be >= 0")
            parsed["apg_start_step"] = start_step_i

        def _parse_optional_float(
            key: str,
            *,
            minimum: float | None = None,
            maximum: float | None = None,
            maximum_inclusive: bool = True,
        ) -> None:
            if key not in options:
                return
            raw = options.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise HTTPException(status_code=400, detail=f"'{field_name}.{key}' must be numeric")
            value_f = float(raw)
            if not math.isfinite(value_f):
                raise HTTPException(status_code=400, detail=f"'{field_name}.{key}' must be finite")
            if minimum is not None and value_f < minimum:
                raise HTTPException(status_code=400, detail=f"'{field_name}.{key}' must be >= {minimum}")
            if maximum is not None:
                if maximum_inclusive:
                    if value_f > maximum:
                        raise HTTPException(status_code=400, detail=f"'{field_name}.{key}' must be <= {maximum}")
                elif value_f >= maximum:
                    raise HTTPException(status_code=400, detail=f"'{field_name}.{key}' must be < {maximum}")
            parsed[key] = value_f

        _parse_optional_float("apg_eta")
        _parse_optional_float("apg_momentum", minimum=0.0, maximum=1.0, maximum_inclusive=False)
        _parse_optional_float("apg_norm_threshold", minimum=0.0)
        _parse_optional_float("apg_rescale", minimum=0.0, maximum=1.0)
        _parse_optional_float("guidance_rescale", minimum=0.0, maximum=1.0)
        _parse_optional_float("cfg_trunc_ratio", minimum=0.0, maximum=1.0)
        _parse_optional_float("renorm_cfg", minimum=0.0)

        return parsed


    def _parse_text_encoder_override_payload(value: Any, *, field_name: str) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be an object")
        _reject_unknown_keys(value, {"family", "label", "components"}, field_name)
        family_raw = value.get("family")
        label_raw = value.get("label")
        if not isinstance(family_raw, str) or not family_raw.strip():
            raise HTTPException(status_code=400, detail=f"'{field_name}.family' must be a non-empty string")
        if not isinstance(label_raw, str) or not label_raw.strip():
            raise HTTPException(status_code=400, detail=f"'{field_name}.label' must be a non-empty string")
        family = family_raw.strip()
        label = label_raw.strip()
        if "/" in label and not label.startswith(f"{family}/"):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name}.label must start with '<family>/'",
            )
        components_val = value.get("components")
        components: list[str] | None = None
        if components_val is not None:
            if not isinstance(components_val, list) or any(not isinstance(c, str) for c in components_val):
                raise HTTPException(
                    status_code=400,
                    detail=f"'{field_name}.components' must be an array of strings",
                )
            components = [c.strip() for c in components_val if isinstance(c, str) and c.strip()]
        parsed: Dict[str, Any] = {"family": family, "label": label}
        if components:
            parsed["components"] = components
        return parsed


    def _parse_swap_model_selection_fields(
        value: Any,
        *,
        field_name: str,
        allow_zimage_variant: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be an object")
        model_raw = value.get("model")
        model = str(model_raw).strip() if isinstance(model_raw, str) else ""
        model_sha_raw = value.get("model_sha")
        model_sha = str(model_sha_raw).strip().lower() if isinstance(model_sha_raw, str) else ""
        if not model and not model_sha:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' requires 'model' or '{field_name}.model_sha'",
            )
        parsed: Dict[str, Any] = {}
        if model:
            parsed["model"] = model
        if model_sha:
            parsed["model_sha"] = model_sha
        checkpoint_core_only = _parse_optional_bool_selector(
            payload=value,
            key="checkpoint_core_only",
            field_name=f"{field_name}.checkpoint_core_only",
        )
        if checkpoint_core_only is not None:
            parsed["checkpoint_core_only"] = checkpoint_core_only
        model_format = _parse_optional_model_format_selector(
            payload=value,
            key="model_format",
            field_name=f"{field_name}.model_format",
        )
        if model_format is not None:
            parsed["model_format"] = model_format
        vae_source = _parse_optional_vae_source_selector(
            payload=value,
            key="vae_source",
            field_name=f"{field_name}.vae_source",
        )
        if vae_source is not None:
            parsed["vae_source"] = vae_source
        if "vae_sha" in value:
            vae_sha_raw = value.get("vae_sha")
            if vae_sha_raw is not None:
                if not isinstance(vae_sha_raw, str) or not vae_sha_raw.strip():
                    raise HTTPException(status_code=400, detail=f"'{field_name}.vae_sha' must be a non-empty string")
                parsed["vae_sha"] = vae_sha_raw.strip().lower()
        if "tenc_sha" in value:
            tenc_raw = value.get("tenc_sha")
            if isinstance(tenc_raw, str):
                tenc_sha = tenc_raw.strip().lower()
                if tenc_sha:
                    parsed["tenc_sha"] = tenc_sha
            elif isinstance(tenc_raw, list):
                tenc_shas: list[str] = []
                for entry in tenc_raw:
                    if not isinstance(entry, str):
                        raise HTTPException(
                            status_code=400,
                            detail=f"'{field_name}.tenc_sha' must be a string or array of strings",
                        )
                    normalized = entry.strip().lower()
                    if normalized:
                        tenc_shas.append(normalized)
                if tenc_shas:
                    parsed["tenc_sha"] = tenc_shas
            elif tenc_raw is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{field_name}.tenc_sha' must be a string or array of strings",
                )
        for numbered_key in ("tenc1_sha", "tenc2_sha"):
            if numbered_key not in value:
                continue
            numbered_raw = value.get(numbered_key)
            if numbered_raw is None:
                continue
            if not isinstance(numbered_raw, str) or not numbered_raw.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"'{field_name}.{numbered_key}' must be a non-empty string",
                )
            parsed[numbered_key] = numbered_raw.strip().lower()
        if "text_encoder_override" in value and value.get("text_encoder_override") is not None:
            parsed["text_encoder_override"] = _parse_text_encoder_override_payload(
                value.get("text_encoder_override"),
                field_name=f"{field_name}.text_encoder_override",
            )
        if allow_zimage_variant:
            zimage_variant = _parse_optional_zimage_variant_selector(
                payload=value,
                key="zimage_variant",
                field_name=f"{field_name}.zimage_variant",
            )
            if zimage_variant is not None:
                parsed["zimage_variant"] = zimage_variant
        return parsed


    def _parse_swap_model_payload(
        value: Any,
        *,
        field_name: str,
        allow_zimage_variant: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be an object")
        allowed_keys = {
            "model",
            "model_sha",
            "checkpoint_core_only",
            "model_format",
            "vae_source",
            "vae_sha",
            "tenc_sha",
            "tenc1_sha",
            "tenc2_sha",
            "text_encoder_override",
        }
        if allow_zimage_variant:
            allowed_keys.add("zimage_variant")
        _reject_unknown_keys(
            value,
            allowed_keys,
            field_name,
        )
        return _parse_swap_model_selection_fields(
            value,
            field_name=field_name,
            allow_zimage_variant=allow_zimage_variant,
        )


    def _parse_swap_stage_payload(
        value: Any,
        *,
        field_name: str,
        allow_zimage_variant: bool = False,
    ) -> Dict[str, Any] | None:
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be an object")
        allowed_keys = {
            "enable",
            "switch_at_step",
            "cfg",
            "seed",
            "model",
            "model_sha",
            "checkpoint_core_only",
            "model_format",
            "vae_source",
            "vae_sha",
            "tenc_sha",
            "tenc1_sha",
            "tenc2_sha",
            "text_encoder_override",
        }
        if allow_zimage_variant:
            allowed_keys.add("zimage_variant")
        _reject_unknown_keys(
            value,
            allowed_keys,
            field_name,
        )
        if _optional_bool_field(value, "enable") is not True:
            return None
        parsed = _parse_swap_model_selection_fields(
            value,
            field_name=field_name,
            allow_zimage_variant=allow_zimage_variant,
        )
        parsed.update(
            {
                "enable": True,
                "switch_at_step": _require_int_field(value, "switch_at_step", minimum=1),
                "cfg": _require_float_field(value, "cfg"),
                "seed": _require_int_field(value, "seed"),
            }
        )
        return parsed


    def _parse_refiner_payload(value: Any, *, field_name: str) -> Dict[str, Any] | None:
        return _parse_swap_stage_payload(value, field_name=field_name, allow_zimage_variant=False)


    def _resolve_inventory_scoped_path(
        raw: object,
        *,
        field_name: str,
        inventory_key: str,
        inventory_label: str,
    ) -> str:
        if not isinstance(raw, str) or not raw.strip():
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be a non-empty string")
        raw_value = raw.strip()
        from apps.backend.inventory import cache as _inventory_cache

        inventory = _inventory_cache.get()
        try:
            raw_absolute = str(Path(raw_value).expanduser().resolve(strict=False))
        except Exception:
            raw_absolute = ""

        for item in inventory.get(inventory_key, []):
            if not isinstance(item, dict):
                continue
            item_path = item.get("path")
            if not isinstance(item_path, str) or not item_path.strip():
                continue
            absolute_path = str(Path(item_path).expanduser().resolve(strict=False))
            api_path = _path_for_api(item_path)
            if raw_value == api_path or (raw_absolute and raw_absolute == absolute_path):
                return absolute_path

        raise HTTPException(
            status_code=400,
            detail=(
                f"'{field_name}' must resolve to an inventory-backed {inventory_label} path from "
                f"'/api/models' ({inventory_key})."
            ),
        )


    def _parse_ip_adapter_payload(
        value: Any,
        *,
        field_name: str,
        allow_same_as_init: bool,
        allow_server_folder: bool,
    ) -> Dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be an object")
        _reject_unknown_keys(value, _IP_ADAPTER_KEYS, field_name)
        if _optional_bool_field(value, "enabled") is not True:
            return None

        model_path = _resolve_inventory_scoped_path(
            value.get("model"),
            field_name=f"{field_name}.model",
            inventory_key="ip_adapter_models",
            inventory_label="IP-Adapter model",
        )
        image_encoder_path = _resolve_inventory_scoped_path(
            value.get("image_encoder"),
            field_name=f"{field_name}.image_encoder",
            inventory_key="ip_adapter_image_encoders",
            inventory_label="IP-Adapter image encoder",
        )
        weight = _require_float_field(value, "weight") if "weight" in value else 1.0
        if not math.isfinite(weight) or weight < 0.0:
            raise HTTPException(status_code=400, detail=f"'{field_name}.weight' must be a finite number >= 0.0")
        start_at = _require_float_field(value, "start_at") if "start_at" in value else 0.0
        end_at = _require_float_field(value, "end_at") if "end_at" in value else 1.0
        if not 0.0 <= start_at <= 1.0:
            raise HTTPException(status_code=400, detail=f"'{field_name}.start_at' must be between 0.0 and 1.0")
        if not 0.0 <= end_at <= 1.0:
            raise HTTPException(status_code=400, detail=f"'{field_name}.end_at' must be between 0.0 and 1.0")
        if start_at > end_at:
            raise HTTPException(status_code=400, detail=f"'{field_name}.start_at' must be <= '{field_name}.end_at'")

        source_raw = value.get("source")
        if not isinstance(source_raw, dict):
            raise HTTPException(status_code=400, detail=f"'{field_name}.source' must be an object")
        _reject_unknown_keys(source_raw, _IP_ADAPTER_SOURCE_KEYS, f"{field_name}.source")
        source_kind = _require_str_field(source_raw, "kind", allow_empty=False)
        allowed_kinds = {"uploaded"}
        if allow_same_as_init:
            allowed_kinds.add("same_as_init")
        if allow_server_folder:
            allowed_kinds.add("server_folder")
        if source_kind not in allowed_kinds:
            allowed = ", ".join(sorted(allowed_kinds))
            raise HTTPException(status_code=400, detail=f"'{field_name}.source.kind' must be one of: {allowed}")

        reference_image_data = source_raw.get("reference_image_data")
        if source_kind != "server_folder":
            for folder_only_key in ("folder_path", "selection_mode", "count", "order", "sort_by"):
                if folder_only_key in source_raw:
                    raise HTTPException(
                        status_code=400,
                        detail=f"'{field_name}.source.{folder_only_key}' is only valid when kind='server_folder'",
                    )
        if source_kind == "uploaded":
            if not isinstance(reference_image_data, str) or not reference_image_data.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"'{field_name}.source.reference_image_data' is required when kind='uploaded'",
                )
            return {
                "enabled": True,
                "model": model_path,
                "image_encoder": image_encoder_path,
                "weight": float(weight),
                "start_at": float(start_at),
                "end_at": float(end_at),
                "source": {
                    "kind": "uploaded",
                    "reference_image_data": reference_image_data.strip(),
                },
            }

        if reference_image_data is not None:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.source.reference_image_data' is only valid when kind='uploaded'",
            )

        if source_kind == "same_as_init":
            return {
                "enabled": True,
                "model": model_path,
                "image_encoder": image_encoder_path,
                "weight": float(weight),
                "start_at": float(start_at),
                "end_at": float(end_at),
                "source": {"kind": "same_as_init"},
            }

        selection_mode = source_raw.get("selection_mode")
        if selection_mode is None:
            selection_mode = "all"
        elif not isinstance(selection_mode, str):
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.source.selection_mode' must be one of: all, count",
            )
        else:
            selection_mode = selection_mode.strip()
        if selection_mode not in {"all", "count"}:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.source.selection_mode' must be one of: all, count",
            )
        order = str(source_raw.get("order", "sorted") or "").strip()
        if order not in {"random", "sorted"}:
            raise HTTPException(status_code=400, detail=f"'{field_name}.source.order' must be one of: random, sorted")
        sort_by = source_raw.get("sort_by")
        if sort_by is None:
            sort_by = "name"
        elif not isinstance(sort_by, str):
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.source.sort_by' must be one of: name, size, created_at, modified_at",
            )
        else:
            sort_by = sort_by.strip()
        if sort_by not in {"name", "size", "created_at", "modified_at"}:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.source.sort_by' must be one of: name, size, created_at, modified_at",
            )
        selection_count = None
        if selection_mode == "count":
            raw_count = source_raw.get("count")
            if raw_count is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{field_name}.source.count' is required when selection_mode='count'",
                )
            selection_count = _require_int_field(source_raw, "count", minimum=1)
        elif source_raw.get("count") is not None:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.source.count' is only valid when selection_mode='count'",
            )
        folder_raw = source_raw.get("folder_path")
        if not isinstance(folder_raw, str) or not folder_raw.strip():
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.source.folder_path' is required when kind='server_folder'",
            )
        try:
            folder_path = _path_from_api(folder_raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"'{field_name}.source.folder_path' resolves outside CODEX_ROOT") from exc
        return {
            "enabled": True,
            "model": model_path,
            "image_encoder": image_encoder_path,
            "weight": float(weight),
            "start_at": float(start_at),
            "end_at": float(end_at),
            "source": {
                "kind": "server_folder",
                "folder_path": folder_path,
                "selection_mode": str(selection_mode),
                "count": selection_count,
                "order": order,
                "sort_by": str(sort_by),
            },
        }


    def _parse_txt2img_extras(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        raw = payload.get('extras')
        if raw is None:
            return {}, None
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="'extras' must be an object")
        _reject_public_qwen_image_variant(raw, context="extras")
        _reject_unknown_keys(raw, _TXT2IMG_EXTRAS_KEYS, "extras")
        if "supir" in raw:
            raise HTTPException(
                status_code=400,
                detail="'extras.supir' is unsupported for txt2img; SUPIR mode is available only on SDXL img2img/inpaint.",
            )
        extras: Dict[str, Any] = {}
        ip_adapter_payload = _parse_ip_adapter_payload(
            raw.get("ip_adapter"),
            field_name="extras.ip_adapter",
            allow_same_as_init=False,
            allow_server_folder=False,
        )
        if ip_adapter_payload is not None:
            extras["ip_adapter"] = ip_adapter_payload
        if 'eta_noise_seed_delta' in raw:
            val = raw['eta_noise_seed_delta']
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise HTTPException(status_code=400, detail="'extras.eta_noise_seed_delta' must be numeric")
            extras['eta_noise_seed_delta'] = int(val)
        # SHA keys for asset selection (from dataclass)
        from apps.backend.types.payloads import SHA_KEYS
        for key in SHA_KEYS.ALL:
            if key not in raw:
                continue
            value = raw.get(key)
            if value is None:
                continue
            if key in {"tenc_sha", "lora_sha"}:
                if isinstance(value, str):
                    sha = value.strip()
                    if sha:
                        extras[key] = sha
                    continue
                if isinstance(value, list):
                    shas: list[str] = []
                    for entry in value:
                        if not isinstance(entry, str):
                            raise HTTPException(status_code=400, detail=f"'extras.{key}' must be a string or array of strings")
                        sha = entry.strip()
                        if sha:
                            shas.append(sha)
                    if shas:
                        extras[key] = shas
                    continue
                raise HTTPException(status_code=400, detail=f"'extras.{key}' must be a string or array of strings")

            if not isinstance(value, str):
                raise HTTPException(status_code=400, detail=f"'extras.{key}' must be a string")
            sha = value.strip()
            if sha:
                extras[key] = sha
        checkpoint_core_only = _parse_optional_bool_selector(
            payload=raw,
            key="checkpoint_core_only",
            field_name="extras.checkpoint_core_only",
        )
        if checkpoint_core_only is not None:
            extras["checkpoint_core_only"] = checkpoint_core_only
        model_format = _parse_optional_model_format_selector(
            payload=raw,
            key="model_format",
            field_name="extras.model_format",
        )
        if model_format is not None:
            extras["model_format"] = model_format
        vae_source = _parse_optional_vae_source_selector(
            payload=raw,
            key="vae_source",
            field_name="extras.vae_source",
        )
        if vae_source is not None:
            extras["vae_source"] = vae_source
        # Batch params
        if 'batch_size' in raw:
            extras['batch_size'] = int(raw['batch_size'])
        if 'batch_count' in raw:
            extras['batch_count'] = int(raw['batch_count'])
        # Z-Image variant selection (Turbo/Base). This is used by the engine to pick
        # variant-specific scheduler semantics (flow_shift) and CFG behavior.
        if 'zimage_variant' in raw:
            val = raw.get('zimage_variant')
            if val is None:
                pass
            elif not isinstance(val, str):
                raise HTTPException(status_code=400, detail="'extras.zimage_variant' must be a string")
            else:
                variant = val.strip().lower()
                if variant not in {"turbo", "base"}:
                    raise HTTPException(
                        status_code=400,
                        detail="'extras.zimage_variant' must be one of: turbo, base",
                    )
                extras['zimage_variant'] = variant
        if "er_sde" in raw:
            extras["er_sde"] = _parse_er_sde_options(raw["er_sde"], field_name="extras.er_sde")
        if "guidance" in raw:
            extras["guidance"] = _parse_guidance_options(raw["guidance"], field_name="extras.guidance")
        # Hires options
        hires = raw.get('hires')
        hires_cfg: Optional[Dict[str, Any]] = None
        if hires is not None:
            if not isinstance(hires, dict):
                raise HTTPException(status_code=400, detail="'extras.hires' must be an object")
            _reject_removed_txt2img_hires_modules_from_hires(hires)
            _reject_unknown_keys(hires, _TXT2IMG_HIRES_KEYS, "extras.hires")
            if _optional_bool_field(hires, "enable") is True:
                required = ['denoise', 'scale', 'resize_x', 'resize_y', 'steps', 'upscaler']
                _require_nested_fields(hires, required, context="extras.hires")
                refiner_raw = hires.get('refiner')
                refiner_cfg = (
                    _parse_refiner_payload(refiner_raw, field_name="extras.hires.refiner")
                    if refiner_raw is not None
                    else None
                )
                swap_model_cfg = (
                    _parse_swap_model_payload(
                        hires.get("swap_model"),
                        field_name="extras.hires.swap_model",
                        allow_zimage_variant=True,
                    )
                    if hires.get("swap_model") is not None
                    else None
                )
                try:
                    tile_cfg = tile_config_from_payload(hires.get("tile"), context="extras.hires.tile")
                except ValueError as exc:
                    _router_log.warning("txt2img extras.hires.tile validation failed: %s", exc)
                    raise HTTPException(
                        status_code=400,
                        detail=public_http_error_detail(exc, fallback="Invalid 'extras.hires.tile' configuration"),
                    ) from None
                tile = {
                    "tile": int(tile_cfg.tile),
                    "overlap": int(tile_cfg.overlap),
                    "fallback_on_oom": bool(tile_cfg.fallback_on_oom),
                    "min_tile": int(tile_cfg.min_tile),
                }
                hires_cfg = {
                    "denoise": _require_float_field(hires, 'denoise', minimum=0.0, maximum=1.0),
                    "scale": _require_float_field(hires, 'scale'),
                    "resize_x": _require_int_field(hires, 'resize_x'),
                    "resize_y": _require_int_field(hires, 'resize_y'),
                    "steps": _require_int_field(hires, 'steps', minimum=0),
                    "upscaler": _require_str_field(hires, 'upscaler', allow_empty=False, trim=True),
                    "tile": tile,
                    "swap_model": swap_model_cfg,
                    "sampler": hires.get('sampler'),
                    "scheduler": hires.get('scheduler'),
                    "prompt": hires.get('prompt') or '',
                    "negative_prompt": hires.get('negative_prompt') or '',
                    "cfg": _require_float_field(hires, 'cfg') if hires.get('cfg') is not None else None,
                    "distilled_cfg": _require_float_field(hires, 'distilled_cfg') if hires.get('distilled_cfg') is not None else None,
                    "refiner": refiner_cfg,
                }

        swap_model = raw.get("swap_model")
        if swap_model is not None:
            swap_stage_cfg = _parse_swap_stage_payload(
                swap_model,
                field_name="extras.swap_model",
                allow_zimage_variant=True,
            )
            if swap_stage_cfg is not None:
                extras["swap_model"] = swap_stage_cfg

        refiner = raw.get('refiner')
        if refiner is not None:
            ref_cfg = _parse_refiner_payload(refiner, field_name="extras.refiner")
            if ref_cfg is not None:
                extras['refiner'] = ref_cfg

        # Text encoder override (family + label [+ optional components])
        te_override = raw.get('text_encoder_override')
        if te_override is not None:
            extras["text_encoder_override"] = _parse_text_encoder_override_payload(
                te_override,
                field_name="extras.text_encoder_override",
            )

        return extras, hires_cfg

    def _reject_removed_img2img_second_pass_keys(payload: Mapping[str, Any]) -> None:
        for key in payload.keys():
            if not isinstance(key, str) or not key.startswith("img2img_"):
                continue
            suffix = key[len("img2img_"):]
            if suffix.startswith("hires_") or suffix.startswith("hr_"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported removed img2img hires key: {key}. Use 'img2img_extras.hires'.",
                )

    def _parse_img2img_nested_hires_config(
        hires: object,
        *,
        default_cfg: float,
        default_distilled: float,
    ) -> Optional[Dict[str, Any]]:
        if hires is None:
            return None
        if not isinstance(hires, dict):
            raise HTTPException(status_code=400, detail="'img2img_extras.hires' must be an object")
        for key in sorted(_IMG2IMG_UNSUPPORTED_HIRES_KEYS):
            if key in hires:
                raise HTTPException(
                    status_code=400,
                    detail=f"'img2img_extras.hires.{key}' is unsupported for img2img.",
                )
        _reject_unknown_keys(hires, _IMG2IMG_HIRES_KEYS, "img2img_extras.hires")
        if _optional_bool_field(hires, "enable") is not True:
            return None
        required = ("denoise", "scale", "resize_x", "resize_y", "steps", "upscaler")
        _require_nested_fields(hires, required, context="img2img_extras.hires")
        try:
            tile_cfg = tile_config_from_payload(hires.get("tile"), context="img2img_extras.hires.tile")
        except ValueError as exc:
            _router_log.warning("img2img_extras.hires.tile validation failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid 'img2img_extras.hires.tile' configuration"),
            ) from None
        return {
            "denoise": _require_float_field(hires, "denoise", minimum=0.0, maximum=1.0),
            "scale": _require_float_field(hires, "scale"),
            "resize_x": _require_int_field(hires, "resize_x", minimum=0),
            "resize_y": _require_int_field(hires, "resize_y", minimum=0),
            "steps": _require_int_field(hires, "steps", minimum=0),
            "upscaler": _require_str_field(hires, "upscaler", allow_empty=False, trim=True),
            "tile": {
                "tile": int(tile_cfg.tile),
                "overlap": int(tile_cfg.overlap),
                "fallback_on_oom": bool(tile_cfg.fallback_on_oom),
                "min_tile": int(tile_cfg.min_tile),
            },
            "sampler": hires.get("sampler"),
            "scheduler": hires.get("scheduler"),
            "prompt": hires.get("prompt") or "",
            "negative_prompt": hires.get("negative_prompt") or "",
            "cfg": _require_float_field(hires, "cfg") if hires.get("cfg") is not None else float(default_cfg),
            "distilled_cfg": (
                _require_float_field(hires, "distilled_cfg")
                if hires.get("distilled_cfg") is not None
                else float(default_distilled)
            ),
        }

    def _parse_img2img_nested_hires_from_extras(
        raw_extras: object,
        *,
        default_cfg: float,
        default_distilled: float,
    ) -> Optional[Dict[str, Any]]:
        if raw_extras is None:
            return None
        if not isinstance(raw_extras, dict):
            raise HTTPException(status_code=400, detail="'img2img_extras' must be an object")
        _reject_unknown_keys(raw_extras, _IMG2IMG_EXTRAS_KEYS, "img2img_extras")
        return _parse_img2img_nested_hires_config(
            raw_extras.get("hires"),
            default_cfg=default_cfg,
            default_distilled=default_distilled,
        )

    def _validate_pre_task_img2img_payload(payload: Mapping[str, Any]) -> None:
        raw_engine = payload.get("engine")
        if raw_engine is None:
            raise HTTPException(status_code=400, detail="Missing 'engine'")
        if not isinstance(raw_engine, str) or not raw_engine.strip():
            raise HTTPException(status_code=400, detail="'engine' must be a non-empty string")
        _reject_lens_parked_request(payload, route_label="img2img")
        _reject_removed_img2img_second_pass_keys(payload)
        _reject_public_qwen_image_variant(payload, context="img2img")
        if _is_qwen_image_engine(raw_engine):
            _validate_qwen_image_img2img_core_fields(payload)
            return
        hires_cfg = _parse_img2img_nested_hires_from_extras(
            payload.get("img2img_extras"),
            default_cfg=1.0,
            default_distilled=3.5,
        )
        if hires_cfg is None:
            return
        if "img2img_mask" in payload and payload.get("img2img_mask") is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "img2img hires does not support masks/inpaint in this backend seam yet. "
                    "Disable hires or remove 'img2img_mask'."
                ),
            )
        raw_extras = payload.get("img2img_extras")
        raw_supir_payload = raw_extras.get("supir") if isinstance(raw_extras, dict) else None
        try:
            supir_config = parse_supir_mode_config(raw_supir_payload)
        except SupirConfigError as exc:
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid 'img2img_extras.supir' configuration"),
            ) from None
        if supir_config is not None:
            raise HTTPException(
                status_code=400,
                detail="'img2img_extras.supir' cannot be combined with img2img hires in tranche 1.",
            )

    def _validate_pre_task_txt2img_payload(payload: Mapping[str, Any]) -> None:
        _reject_lens_parked_request(payload, route_label="txt2img")
        _reject_public_qwen_image_variant(payload, context="txt2img")
        if _is_qwen_image_engine(payload.get("engine")):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Engine 'qwen_image' is Edit-2511 img2img-only. "
                    "Use POST /api/img2img with one init image and the exact transformer/TEnc/VAE selectors."
                ),
            )

    def _resolve_model_ref_from_sha_or_name(
        *,
        model_override: Any,
        extras: Dict[str, Any],
        field_prefix: str,
        models_api: Any,
    ) -> str:
        """Legacy checkpoint resolution used by generic video routes."""

        model_sha = extras.get("model_sha")
        sha_candidate = None
        if isinstance(model_sha, str) and model_sha.strip():
            sha_candidate = model_sha.strip()
        elif isinstance(model_override, str):
            maybe = model_override.strip()
            if len(maybe) in (10, 64) and all(c in "0123456789abcdef" for c in maybe.lower()):
                sha_candidate = maybe

        resolved = model_override
        if sha_candidate:
            record = models_api.find_checkpoint_by_sha(sha_candidate)
            if record is None:
                raise HTTPException(status_code=409, detail=f"Checkpoint not found for sha: {sha_candidate}")
            resolved = record.filename
            extras["model_path"] = record.filename

        model_sha_field = f"{field_prefix}.model_sha" if field_prefix else "model_sha"
        if not isinstance(resolved, str) or not resolved.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Missing model selection: provide 'model' or '{model_sha_field}'",
            )
        return resolved.strip()

    def _resolve_checkpoint_selection(
        *,
        model_override: Any,
        extras: Dict[str, Any],
        field_prefix: str,
        models_api: Any,
    ) -> tuple[str, Any]:
        """Resolve the selected checkpoint record for a request."""

        model_sha = extras.get("model_sha")
        record = None
        if model_sha is not None:
            model_sha_field = f"{field_prefix}.model_sha" if field_prefix else "model_sha"
            if not isinstance(model_sha, str) or not model_sha.strip():
                raise HTTPException(status_code=400, detail=f"'{model_sha_field}' must be a non-empty string")
            sha_candidate = model_sha.strip().lower()
            record = models_api.find_checkpoint_by_sha(sha_candidate)
            if record is None:
                raise HTTPException(status_code=409, detail=f"Checkpoint not found for sha: {sha_candidate}")
        else:
            if not isinstance(model_override, str) or not model_override.strip():
                model_sha_field = f"{field_prefix}.model_sha" if field_prefix else "model_sha"
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing model selection: provide 'model' or '{model_sha_field}'",
                )
            record = models_api.find_checkpoint(model_override.strip())
            if record is None:
                raise HTTPException(status_code=409, detail=f"Selected checkpoint not found: {model_override.strip()}")

        resolved_model_ref = str(getattr(record, "filename", "") or "").strip()
        if not resolved_model_ref:
            raise HTTPException(status_code=409, detail="Selected checkpoint record is missing filename metadata.")

        extras["model_path"] = resolved_model_ref
        return resolved_model_ref, record

    def _normalize_checkpoint_core_only_field(value: object, *, field_label: str) -> bool | None:
        if value is None:
            return None
        if not isinstance(value, bool):
            raise HTTPException(status_code=400, detail=f"'{field_label}' must be a boolean")
        return value

    def _normalize_model_format_field(value: object, *, field_label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(status_code=400, detail=f"'{field_label}' must be a non-empty string")
        normalized = value.strip().lower()
        if normalized not in {"checkpoint", "diffusers", "gguf"}:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_label}' must be one of: checkpoint, diffusers, gguf",
            )
        return normalized

    def _record_checkpoint_format(record: Any) -> str | None:
        raw = getattr(record, "format", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
        value = getattr(raw, "value", None)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        return None

    def _resolve_checkpoint_contract_from_request(
        *,
        checkpoint_record: Any,
        extras: Dict[str, Any],
        field_prefix: str,
        require_explicit_contract: bool,
    ) -> tuple[bool, str]:
        core_only_field = _asset_field_label(field_prefix=field_prefix, field_name="checkpoint_core_only")
        model_format_field = _asset_field_label(field_prefix=field_prefix, field_name="model_format")
        explicit_core_only = _normalize_checkpoint_core_only_field(
            extras.get("checkpoint_core_only"),
            field_label=core_only_field,
        )
        explicit_model_format = _normalize_model_format_field(
            extras.get("model_format"),
            field_label=model_format_field,
        )

        record_core_only = getattr(checkpoint_record, "core_only", None)
        if not isinstance(record_core_only, bool):
            raise HTTPException(
                status_code=409,
                detail="Selected checkpoint record is missing core_only metadata. Refresh model inventory and retry.",
            )

        record_model_format = _record_checkpoint_format(checkpoint_record)
        if record_model_format is None:
            raise HTTPException(
                status_code=409,
                detail="Selected checkpoint record is missing format metadata. Refresh model inventory and retry.",
            )

        if explicit_core_only is None:
            if require_explicit_contract:
                raise HTTPException(status_code=400, detail=f"Missing '{core_only_field}'")
            resolved_core_only = record_core_only
        else:
            resolved_core_only = explicit_core_only

        if explicit_model_format is None:
            if require_explicit_contract:
                raise HTTPException(status_code=400, detail=f"Missing '{model_format_field}'")
            resolved_model_format = record_model_format
        else:
            resolved_model_format = explicit_model_format

        if explicit_core_only is not None and explicit_core_only != record_core_only:
            raise HTTPException(
                status_code=409,
                detail=f"'{core_only_field}' does not match selected checkpoint inventory metadata. Refresh model inventory and retry.",
            )
        if explicit_model_format is not None and explicit_model_format != record_model_format:
            raise HTTPException(
                status_code=409,
                detail=f"'{model_format_field}' does not match selected checkpoint inventory metadata. Refresh model inventory and retry.",
            )

        extras["checkpoint_core_only"] = resolved_core_only
        extras["model_format"] = resolved_model_format
        return resolved_core_only, resolved_model_format

    def _parse_flux2_submitted_guidance(
        *,
        payload: Dict[str, Any],
        cfg_field: str,
        distilled_cfg_field: str,
        distilled_cfg_default: Optional[float],
    ) -> Tuple[float, Optional[float]]:
        has_cfg = cfg_field in payload
        has_distilled_cfg = distilled_cfg_field in payload
        if has_cfg == has_distilled_cfg:
            raise HTTPException(
                status_code=400,
                detail=f"FLUX.2 requires exactly one of '{cfg_field}' or '{distilled_cfg_field}'.",
            )
        if has_cfg:
            return _require_float_field(payload, cfg_field), distilled_cfg_default
        return 1.0, _require_float_field(payload, distilled_cfg_field)

    def _build_hires(cfg: Optional[Dict[str, Any]], width: int, height: int, fallback_cfg: float, fallback_distilled: float = 3.5) -> Dict[str, Any]:
        if cfg is None:
            return {
                "enable": False,
                "denoise": 0.0,
                "scale": 1.0,
                "upscaler": "Use same upscaler",
                "steps": 0,
                "resize_x": width,
                "resize_y": height,
                "swap_model": None,
                "sampler_name": None,
                "scheduler": None,
                "prompt": "",
                "negative_prompt": "",
                "cfg": fallback_cfg,
                "distilled_cfg": fallback_distilled,
                "refiner": None,
            }
        return {
            "enable": True,
            "denoise": cfg["denoise"],
            "scale": cfg["scale"],
            "upscaler": cfg["upscaler"],
            "tile": cfg.get("tile"),
            "steps": cfg["steps"],
            "resize_x": cfg["resize_x"],
            "resize_y": cfg["resize_y"],
            "swap_model": cfg.get("swap_model"),
            "sampler_name": cfg.get("sampler"),
            "scheduler": cfg.get("scheduler"),
            "prompt": cfg.get("prompt") or "",
            "negative_prompt": cfg.get("negative_prompt") or "",
            "cfg": cfg.get("cfg") if cfg.get("cfg") is not None else fallback_cfg,
            "distilled_cfg": cfg.get("distilled_cfg") if cfg.get("distilled_cfg") is not None else fallback_distilled,
            "refiner": cfg.get("refiner"),
        }

    def _canonical_engine_key(value: object) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        key = raw.lower()
        if key in {"sd35", "sd3", "sd-3.5"}:
            return "sd35"
        if key == _QWEN_IMAGE_ENGINE_ID:
            return _QWEN_IMAGE_ENGINE_ID
        if key == _LENS_ENGINE_ID:
            return _LENS_ENGINE_ID
        from apps.backend.core.registry import registry as _engine_registry
        try:
            _ensure_default_engines_registered()
        except Exception as exc:
            _router_log.exception("engine registry initialization failed")
            raise HTTPException(
                status_code=500,
                detail=public_http_error_detail(exc, fallback="Engine registry init failed"),
            ) from exc
        try:
            return _engine_registry.get_descriptor(key).key
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unknown engine key: {key}") from exc

    _WAN_VIDEO_ENGINE_KEYS = {"wan22_5b", "wan22_14b", "wan22_14b_animate"}

    def _is_legacy_or_wan_video_route_engine(engine_key: str) -> bool:
        normalized = str(engine_key or "").strip().lower()
        return normalized == "" or normalized in _WAN_VIDEO_ENGINE_KEYS

    def _validate_route_engine_capability(
        payload: Mapping[str, Any],
        *,
        route_mode: GenerationRouteMode,
    ) -> None:
        raw_engine = payload.get("engine")
        if raw_engine is None:
            return
        engine_key = _canonical_engine_key(raw_engine)
        if not engine_key:
            return
        if engine_key == _LENS_ENGINE_ID:
            raise HTTPException(status_code=501, detail=_parked_lens_detail())
        capability_attr, route_label = {
            GenerationRouteMode.TXT2IMG: ("supports_txt2img", "txt2img"),
            GenerationRouteMode.IMG2IMG: ("supports_img2img", "img2img"),
            GenerationRouteMode.TXT2VID: ("supports_txt2vid", "txt2vid"),
            GenerationRouteMode.IMG2VID: ("supports_img2vid", "img2vid"),
            GenerationRouteMode.VID2VID: ("supports_vid2vid", "vid2vid"),
        }.get(route_mode, ("", ""))
        if not capability_attr:
            return
        try:
            semantic_engine = semantic_engine_for_engine_id(engine_key)
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Engine '{engine_key}' is not advertised by /api/engines/capabilities "
                    f"for route '{route_label}'."
                ),
            ) from None
        surface = ENGINE_SURFACES[semantic_engine]
        if not getattr(surface, capability_attr):
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{engine_key}' does not support route '{route_label}'.",
            )
        mask_value = payload.get("img2img_mask")
        mask_requested = isinstance(mask_value, str) and bool(mask_value.strip())
        if (
            route_mode == GenerationRouteMode.IMG2IMG
            and mask_requested
            and not bool(surface.supports_img2img_masking)
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{engine_key}' does not support img2img masking/inpaint.",
            )

    def _validate_pre_task_txt2vid_payload(payload: Mapping[str, Any]) -> None:
        raw_engine = payload.get("engine")
        engine_key = _canonical_engine_key(raw_engine) if raw_engine is not None else ""
        if engine_key.startswith("wan22"):
            return
        raw_prompt = payload.get("txt2vid_prompt")
        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            raise HTTPException(status_code=400, detail=_TXT2VID_BLANK_PROMPT_DETAIL)

    def _resolve_image_family_capability_contract(
        engine_key: str,
    ) -> tuple[str | None, Mapping[str, object] | None]:
        from apps.backend.runtime.model_registry.capabilities import (
            primary_family_for_engine_id,
            serialize_family_capabilities,
        )

        normalized_engine = str(engine_key or "").strip()
        if normalized_engine == "":
            return None, None
        try:
            family = primary_family_for_engine_id(normalized_engine)
        except KeyError:
            return None, None
        family_capabilities = serialize_family_capabilities()
        capability_contract = family_capabilities.get(family.value)
        if isinstance(capability_contract, Mapping):
            return family.value, capability_contract
        return family.value, None

    def _enforce_ip_adapter_engine_support(*, engine_key: str, field_name: str) -> None:
        detail = ip_adapter_support_error(engine_key)
        if detail is None:
            return
        raise HTTPException(status_code=400, detail=detail)

    def _enforce_supir_engine_support(*, engine_key: str, field_name: str) -> None:
        del field_name
        detail = supir_support_error(engine_key)
        if detail is None:
            return
        raise HTTPException(status_code=400, detail=detail)

    def _enforce_img2img_inpaint_mode_support(*, engine_key: str, mode: str) -> None:
        from apps.backend.runtime.model_registry.capabilities import inpaint_mode_support_error

        detail = inpaint_mode_support_error(engine_key, mode)
        if detail is None:
            return
        raise HTTPException(status_code=400, detail=detail)

    def _reject_supir_prompt_loras(*, prompt: str, negative_prompt: str) -> None:
        from apps.backend.runtime.text_processing.extra_nets import ExtraNetsParseError, parse_prompts

        try:
            _cleaned_prompts, parsed_loras = parse_prompts([prompt, negative_prompt])
        except ExtraNetsParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if parsed_loras:
            raise HTTPException(
                status_code=400,
                detail="'img2img_extras.supir' cannot be combined with LoRA prompt tags in tranche 1.",
            )

    def _reject_zimage_l2p_prompt_loras(*, prompt: str, negative_prompt: str) -> None:
        from apps.backend.runtime.text_processing.extra_nets import ExtraNetsParseError, parse_prompts

        try:
            _cleaned_prompts, parsed_loras = parse_prompts([prompt, negative_prompt])
        except ExtraNetsParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if parsed_loras:
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_ZIMAGE_L2P_ENGINE_ID}' does not support LoRA prompt tags.",
            )

    def _enforce_txt2img_ip_adapter_stage_support(
        *,
        engine_key: str,
        extras: Mapping[str, Any],
        hires_cfg: Mapping[str, Any] | None,
    ) -> None:
        if not isinstance(extras.get("ip_adapter"), dict):
            return
        _enforce_ip_adapter_engine_support(
            engine_key=engine_key,
            field_name="extras.ip_adapter",
        )
        if isinstance(extras.get("refiner"), dict):
            _enforce_ip_adapter_engine_support(
                engine_key="sdxl_refiner",
                field_name="extras.refiner",
            )
        if isinstance(hires_cfg, Mapping) and isinstance(hires_cfg.get("refiner"), dict):
            _enforce_ip_adapter_engine_support(
                engine_key="sdxl_refiner",
                field_name="extras.hires.refiner",
            )

    def _normalize_capability_name_list(value: object) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in value:
            if not isinstance(raw, str):
                continue
            name = raw.strip()
            if not name:
                continue
            lowered = name.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(name)
        return normalized

    def _enforce_family_sampler_scheduler_support(
        *,
        engine_key: str,
        family_name: str | None,
        family_capability: Mapping[str, object] | None,
        sampler_name: str,
        scheduler_name: str,
        sampler_field_name: str,
        scheduler_field_name: str,
    ) -> None:
        if family_capability is None:
            return
        family_label = family_name or "unknown"

        supported_samplers = _normalize_capability_name_list(family_capability.get("supported_samplers"))
        supported_schedulers = _normalize_capability_name_list(family_capability.get("supported_schedulers"))
        excluded_samplers = _normalize_capability_name_list(family_capability.get("excluded_samplers"))
        excluded_schedulers = _normalize_capability_name_list(family_capability.get("excluded_schedulers"))

        sampler_normalized = str(sampler_name).strip().lower()
        scheduler_normalized = str(scheduler_name).strip().lower()
        supported_sampler_set = {name.lower() for name in supported_samplers}
        supported_scheduler_set = {name.lower() for name in supported_schedulers}
        excluded_sampler_set = {name.lower() for name in excluded_samplers}
        excluded_scheduler_set = {name.lower() for name in excluded_schedulers}

        if supported_sampler_set and sampler_normalized not in supported_sampler_set:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported sampler '{sampler_name}' for '{sampler_field_name}' "
                    f"(engine='{engine_key}', family='{family_label}'). "
                    f"Supported samplers: {supported_samplers}"
                ),
            )
        if sampler_normalized in excluded_sampler_set:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported sampler '{sampler_name}' for '{sampler_field_name}' "
                    f"(engine='{engine_key}', family='{family_label}'): sampler is excluded."
                ),
            )
        if supported_scheduler_set and scheduler_normalized not in supported_scheduler_set:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported scheduler '{scheduler_name}' for '{scheduler_field_name}' "
                    f"(engine='{engine_key}', family='{family_label}'). "
                    f"Supported schedulers: {supported_schedulers}"
                ),
            )
        if scheduler_normalized in excluded_scheduler_set:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported scheduler '{scheduler_name}' for '{scheduler_field_name}' "
                    f"(engine='{engine_key}', family='{family_label}'): scheduler is excluded."
                ),
            )

    def _is_qwen_image_engine(engine_key: object) -> bool:
        return str(engine_key or "").strip().lower() == _QWEN_IMAGE_ENGINE_ID

    def _is_zimage_l2p_engine(engine_key: object) -> bool:
        return str(engine_key or "").strip().lower() == _ZIMAGE_L2P_ENGINE_ID

    def _parked_lens_detail() -> str:
        stub = PARKED_EXACT_ENGINES.get(_LENS_ENGINE_ID)
        if stub is None:
            return _LENS_NOT_IMPLEMENTED_MESSAGE
        return stub.detail

    def _reject_lens_variant_aliases(payload: Mapping[str, Any], *, context: str) -> None:
        for alias_key in ("lens_variant", "lensVariant", "variant", "engine_options"):
            if alias_key in payload:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{context}.{alias_key}' is not a valid Lens variant owner. "
                        "Use 'extras.lens.variant'."
                    ),
                )

    def _validate_lens_extras_owner(raw_extras: object, *, context: str) -> bool:
        if raw_extras is None:
            return False
        if not isinstance(raw_extras, Mapping):
            return False
        _reject_lens_variant_aliases(raw_extras, context=context)
        if _LENS_EXTRAS_KEY not in raw_extras:
            return False
        lens_payload = raw_extras.get(_LENS_EXTRAS_KEY)
        if not isinstance(lens_payload, Mapping):
            raise HTTPException(status_code=400, detail=f"'{context}.lens' must be an object")
        unexpected_keys = sorted(str(key) for key in lens_payload.keys() if key != "variant")
        if unexpected_keys:
            raise HTTPException(
                status_code=400,
                detail=f"'{context}.lens' supports only 'variant'; unexpected keys: {', '.join(unexpected_keys)}.",
            )
        try:
            require_lens_variant(lens_payload.get("variant"), context=f"{context}.lens.variant")
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return True

    def _reject_lens_parked_request(payload: Mapping[str, Any], *, route_label: str) -> None:
        _reject_lens_variant_aliases(payload, context=route_label)
        raw_engine = payload.get("engine")
        engine_key = _canonical_engine_key(raw_engine) if raw_engine is not None else ""
        lens_extras_context = "img2img_extras" if route_label == "img2img" else "extras"
        raw_extras = payload.get(lens_extras_context)
        has_lens_extras = _validate_lens_extras_owner(raw_extras, context=lens_extras_context)
        if has_lens_extras and engine_key != _LENS_ENGINE_ID:
            raise HTTPException(status_code=400, detail=f"'{lens_extras_context}.lens' is only valid for engine 'lens'.")
        if engine_key == _LENS_ENGINE_ID:
            raise HTTPException(status_code=501, detail=_parked_lens_detail())

    def _reject_public_qwen_image_variant(payload: Mapping[str, Any], *, context: str) -> None:
        if _QWEN_IMAGE_VARIANT_KEY in payload:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{context}.{_QWEN_IMAGE_VARIANT_KEY}' is internal-only. "
                    "The live Qwen Image contract is fixed to Edit-2511 img2img."
                ),
            )

    def _require_qwen_image_sampling_pair(*, sampler_name: object, scheduler_name: object, sampler_field: str, scheduler_field: str) -> None:
        sampler = str(sampler_name or "").strip().lower()
        scheduler = str(scheduler_name or "").strip().lower()
        if sampler != _QWEN_IMAGE_SAMPLER:
            raise HTTPException(
                status_code=400,
                detail=f"'{sampler_field}' must be '{_QWEN_IMAGE_SAMPLER}' for engine '{_QWEN_IMAGE_ENGINE_ID}'.",
            )
        if scheduler != _QWEN_IMAGE_SCHEDULER:
            raise HTTPException(
                status_code=400,
                detail=f"'{scheduler_field}' must be '{_QWEN_IMAGE_SCHEDULER}' for engine '{_QWEN_IMAGE_ENGINE_ID}'.",
            )

    def _parse_qwen_image_asset_extras(raw: object, *, field_prefix: str) -> Dict[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail=f"'{field_prefix}' must be an object")
        _reject_public_qwen_image_variant(raw, context=field_prefix)
        _reject_unknown_keys(raw, _QWEN_IMAGE_ASSET_EXTRAS_KEYS, field_prefix)

        extras: Dict[str, Any] = {}
        for key in ("model_sha", "tenc_sha", "vae_sha"):
            if key in raw:
                value = raw.get(key)
                if value is not None:
                    extras[key] = value
        checkpoint_core_only = _parse_optional_bool_selector(
            payload=raw,
            key="checkpoint_core_only",
            field_name=f"{field_prefix}.checkpoint_core_only",
        )
        if checkpoint_core_only is not None:
            extras["checkpoint_core_only"] = checkpoint_core_only
        model_format = _parse_optional_model_format_selector(
            payload=raw,
            key="model_format",
            field_name=f"{field_prefix}.model_format",
        )
        if model_format is not None:
            extras["model_format"] = model_format
        vae_source = _parse_optional_vae_source_selector(
            payload=raw,
            key="vae_source",
            field_name=f"{field_prefix}.vae_source",
        )
        if vae_source is not None:
            extras["vae_source"] = vae_source
        return extras

    def _reject_qwen_image_img2img_surfaces(payload: Mapping[str, Any]) -> None:
        _reject_public_qwen_image_variant(payload, context="img2img")
        for key in sorted(_QWEN_IMAGE_IMG2IMG_REJECTED_TOP_LEVEL_KEYS):
            if key in payload and payload.get(key) is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{key}' is unsupported for engine '{_QWEN_IMAGE_ENGINE_ID}' img2img edit.",
                )
        batch_count = _require_int_field(payload, "img2img_batch_count", minimum=1) if "img2img_batch_count" in payload else 1
        batch_size = _require_int_field(payload, "img2img_batch_size", minimum=1) if "img2img_batch_size" in payload else 1
        if batch_count != 1:
            raise HTTPException(status_code=400, detail="'img2img_batch_count' must be 1 for Qwen Image Edit.")
        if batch_size != 1:
            raise HTTPException(status_code=400, detail="'img2img_batch_size' must be 1 for Qwen Image Edit.")

    def _validate_qwen_image_img2img_core_fields(payload: Mapping[str, Any]) -> None:
        _reject_unknown_keys(payload, _QWEN_IMAGE_IMG2IMG_ALLOWED_KEYS, "img2img")
        for required_key in _QWEN_IMAGE_IMG2IMG_REQUIRED_KEYS:
            if required_key not in payload or payload.get(required_key) is None:
                raise HTTPException(status_code=400, detail=f"Missing '{required_key}'")
        _reject_qwen_image_img2img_surfaces(payload)
        init_image_data = payload.get("img2img_init_image")
        if not isinstance(init_image_data, str) or not init_image_data.strip():
            raise HTTPException(
                status_code=400,
                detail="'img2img_init_image' must be a non-empty string",
            )
        try:
            init_image = media.decode_image(init_image_data)
            init_width, init_height = init_image.size  # type: ignore[attr-defined]
            qwen_image_edit_vae_dimensions(int(init_width), int(init_height))
            qwen_image_edit_condition_dimensions(int(init_width), int(init_height))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(
                    exc,
                    fallback="Invalid Qwen Image 'img2img_init_image' payload",
                ),
            ) from None
        _require_str_field(payload, "img2img_prompt", allow_empty=True)
        _require_str_field(payload, "img2img_neg_prompt", allow_empty=True)
        if "img2img_styles" in payload:
            _p.as_list(payload, "img2img_styles")
        _require_int_field(payload, "img2img_steps", minimum=2)
        _require_float_field(payload, "img2img_cfg_scale")
        sampler_name = _require_str_field(payload, "img2img_sampling")
        scheduler_name = _require_str_field(payload, "img2img_scheduler")
        _require_qwen_image_sampling_pair(
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            sampler_field="img2img_sampling",
            scheduler_field="img2img_scheduler",
        )
        _require_int_field(payload, "img2img_seed")
        _require_int_field(payload, "settings_revision", minimum=0)

        raw_extras = payload.get("img2img_extras")
        if not isinstance(raw_extras, Mapping):
            raise HTTPException(status_code=400, detail="'img2img_extras' must be an object")
        _reject_public_qwen_image_variant(raw_extras, context="img2img_extras")
        _reject_unknown_keys(raw_extras, _QWEN_IMAGE_ASSET_EXTRAS_KEYS, "img2img_extras")
        _preflight_qwen_image_asset_selector_payload(
            raw_extras,
            field_prefix="img2img_extras",
        )

    def _parse_optional_sampler_field(*, value: object, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be a string")
        sampler = value.strip()
        if not sampler:
            raise HTTPException(status_code=400, detail=f"'{field_name}' must not be empty")
        return sampler

    def _parse_optional_scheduler_field(*, value: object, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be a string")
        scheduler = value.strip()
        if not scheduler:
            raise HTTPException(status_code=400, detail=f"'{field_name}' must not be empty")
        return scheduler

    def _validate_er_sde_release_scope(*, engine_key: str, sampler: str, field_name: str) -> None:
        if str(sampler).strip().lower() != "er sde":
            return
        if engine_key == SemanticEngine.ANIMA.value:
            return
        raise HTTPException(
            status_code=400,
            detail=(
                f"Sampler 'er sde' in '{field_name}' is currently enabled only for engine 'anima'."
            ),
        )

    def _validate_swap_at_step_pointer(*, pointer: int, total_steps: int, field_name: str) -> None:
        if total_steps < 2:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' requires total steps >= 2 (got {total_steps})",
            )
        if pointer < 1 or pointer >= total_steps:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' must be in [1, {total_steps - 1}] (got {pointer})",
            )

    def _resolve_hires_sampler_scheduler_override(
        *,
        base_sampler: str,
        base_scheduler: str,
        sampler_override: str | None,
        scheduler_override: str | None,
        sampler_field_name: str,
        scheduler_field_name: str,
    ) -> tuple[str, str]:
        from apps.backend.runtime.pipeline_stages.sampling_plan import resolve_sampler_scheduler_override

        try:
            return resolve_sampler_scheduler_override(
                base_sampler=base_sampler,
                base_scheduler=base_scheduler,
                sampler_override=sampler_override,
                scheduler_override=scheduler_override,
            )
        except (TypeError, ValueError) as exc:
            message = str(exc).strip() or "invalid hires sampler/scheduler override"
            lower = message.lower()
            if "not supported for sampler" in lower:
                detail = (
                    f"Incompatible hires sampler/scheduler override: {message}. "
                    f"Check '{sampler_field_name}' and '{scheduler_field_name}'."
                )
            elif "scheduler" in lower:
                detail = f"Invalid '{scheduler_field_name}': {message}"
            elif "sampler" in lower:
                detail = f"Invalid '{sampler_field_name}': {message}"
            else:
                detail = (
                    "Invalid hires sampler/scheduler override "
                    f"for '{sampler_field_name}' and '{scheduler_field_name}': {message}"
                )
            raise HTTPException(status_code=400, detail=detail) from exc

    from apps.backend.core.contracts.asset_requirements import (
        EngineAssetContract,
        contract_for_request,
        format_text_encoder_kind_label,
    )

    def _normalize_sha_field(value: object, *, field_label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"'{field_label}' must be a string")
        norm = value.strip().lower()
        return norm or None

    def _normalize_sha_list_field(value: object, *, field_label: str) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            sha = value.strip().lower()
            return [sha] if sha else []
        if isinstance(value, list):
            out: list[str] = []
            for entry in value:
                if not isinstance(entry, str):
                    raise HTTPException(status_code=400, detail=f"'{field_label}' must be a string or array of strings")
                sha = entry.strip().lower()
                if sha:
                    out.append(sha)
            return out
        raise HTTPException(status_code=400, detail=f"'{field_label}' must be a string or array of strings")

    def _normalize_path_for_compare(path_value: str) -> str:
        return os.path.normcase(os.path.realpath(os.path.expanduser(path_value)))

    def _format_required_tenc_message(
        *,
        engine_id: str,
        contract: EngineAssetContract,
        field_label: str,
    ) -> str:
        count = int(contract.tenc_count)
        kind = format_text_encoder_kind_label(contract.tenc_kind)
        if count == 1:
            return f"Engine '{engine_id}' requires exactly 1 text encoder ({kind}) via '{field_label}'"
        return f"Engine '{engine_id}' requires exactly {count} text encoders ({kind}) via '{field_label}'"

    def _asset_field_label(*, field_prefix: str, field_name: str) -> str:
        prefix = str(field_prefix or "").strip().strip(".")
        if not prefix:
            return field_name
        return f"{prefix}.{field_name}"

    def _reject_zimage_l2p_txt2img_payload(payload: Mapping[str, Any]) -> None:
        engine_key = _canonical_engine_key(payload.get("engine"))
        if not _is_zimage_l2p_engine(engine_key):
            return

        payload_dict = dict(payload)
        _reject_unknown_keys(payload_dict, _TXT2IMG_ALLOWED_KEYS, "txt2img")

        width = _require_int_field(payload_dict, "width", minimum=1024, maximum=1024)
        height = _require_int_field(payload_dict, "height", minimum=1024, maximum=1024)
        if width != 1024 or height != 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_ZIMAGE_L2P_ENGINE_ID}' requires width=1024 and height=1024.",
            )

        sampler = _require_str_field(payload_dict, "sampler", allow_empty=False).strip().lower()
        if sampler != "euler":
            raise HTTPException(
                status_code=400,
                detail=f"'sampler' must be 'euler' for engine '{_ZIMAGE_L2P_ENGINE_ID}'.",
            )
        scheduler = _require_str_field(payload_dict, "scheduler", allow_empty=False).strip().lower()
        if scheduler != "simple":
            raise HTTPException(
                status_code=400,
                detail=f"'scheduler' must be 'simple' for engine '{_ZIMAGE_L2P_ENGINE_ID}'.",
            )

        if "clip_skip" in payload_dict:
            raise HTTPException(
                status_code=400,
                detail=f"'clip_skip' is unsupported for engine '{_ZIMAGE_L2P_ENGINE_ID}'.",
            )

        prompt = _require_str_field(payload_dict, "prompt", allow_empty=True)
        negative_raw = payload_dict.get("negative_prompt", "")
        if negative_raw is None:
            negative_prompt = ""
        elif isinstance(negative_raw, str):
            negative_prompt = negative_raw
        else:
            raise HTTPException(status_code=400, detail="'negative_prompt' must be a string")
        _reject_zimage_l2p_prompt_loras(prompt=prompt, negative_prompt=negative_prompt)

        raw_extras = payload_dict.get("extras")
        if not isinstance(raw_extras, Mapping):
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_ZIMAGE_L2P_ENGINE_ID}' requires an 'extras' object with explicit asset selectors.",
            )
        _reject_unknown_keys(raw_extras, _TXT2IMG_EXTRAS_KEYS, "extras")

        model_field = _asset_field_label(field_prefix="extras", field_name="model_sha")
        if not _normalize_sha_field(raw_extras.get("model_sha"), field_label=model_field):
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_ZIMAGE_L2P_ENGINE_ID}' requires '{model_field}' (sha256).",
            )

        core_field = _asset_field_label(field_prefix="extras", field_name="checkpoint_core_only")
        core_only = _normalize_checkpoint_core_only_field(raw_extras.get("checkpoint_core_only"), field_label=core_field)
        if core_only is not True:
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_ZIMAGE_L2P_ENGINE_ID}' requires '{core_field}' = true.",
            )

        format_field = _asset_field_label(field_prefix="extras", field_name="model_format")
        model_format = _normalize_model_format_field(raw_extras.get("model_format"), field_label=format_field)
        if model_format is None:
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_ZIMAGE_L2P_ENGINE_ID}' requires '{format_field}'.",
            )
        if model_format not in _ZIMAGE_L2P_MODEL_FORMATS:
            allowed = ", ".join(sorted(_ZIMAGE_L2P_MODEL_FORMATS))
            raise HTTPException(
                status_code=400,
                detail=f"'{format_field}' must be one of: {allowed} for engine '{_ZIMAGE_L2P_ENGINE_ID}'.",
            )

        tenc_field = _asset_field_label(field_prefix="extras", field_name="tenc_sha")
        tenc_shas = _normalize_sha_list_field(raw_extras.get("tenc_sha"), field_label=tenc_field)
        if len(tenc_shas) != 1:
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_ZIMAGE_L2P_ENGINE_ID}' requires exactly 1 Qwen3-4B text encoder via '{tenc_field}'.",
            )

        for key in sorted(_ZIMAGE_L2P_TXT2IMG_REJECTED_EXTRAS_KEYS):
            if key in raw_extras:
                raise HTTPException(
                    status_code=400,
                    detail=f"'extras.{key}' is unsupported for engine '{_ZIMAGE_L2P_ENGINE_ID}'.",
                )

        for batch_key in ("batch_size", "batch_count"):
            if batch_key not in raw_extras:
                continue
            batch_value = _require_int_field(dict(raw_extras), batch_key, minimum=1, maximum=1)
            if batch_value != 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"'extras.{batch_key}' must be 1 for engine '{_ZIMAGE_L2P_ENGINE_ID}'.",
                )

    def _resolve_qwen_image_vae_sha_to_path(
        *,
        vae_sha: str,
        vae_field: str,
        resolve_asset_by_sha,  # type: ignore[no-untyped-def]
        resolve_vae_path_by_sha,  # type: ignore[no-untyped-def]
    ) -> str:
        from apps.backend.infra.config.paths import get_paths_for as _get_paths_for
        from apps.backend.runtime.families.qwen_image.vae import qwen_image_validate_external_vae_path

        vae_path = resolve_vae_path_by_sha(vae_sha)
        if not vae_path:
            non_vae_path = resolve_asset_by_sha(vae_sha)
            if non_vae_path:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"'{vae_field}' resolved to a non-VAE asset path: {non_vae_path}. "
                        "Select a SHA from inventory.vaes."
                    ),
                )
            raise HTTPException(status_code=409, detail=f"Asset not found for sha: {vae_sha}")

        qwen_image_vae_roots = tuple(_get_paths_for("qwen_image_vae"))
        if not qwen_image_vae_roots:
            raise HTTPException(status_code=409, detail="No qwen_image_vae roots are configured.")
        try:
            return qwen_image_validate_external_vae_path(
                vae_path,
                allowed_roots=qwen_image_vae_roots,
                context=f"{vae_field} Qwen Image VAE",
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    def _path_is_under_configured_roots(path: str, roots: Sequence[object]) -> bool:
        resolved_path = Path(path).expanduser().resolve()
        for raw_root in roots:
            if not isinstance(raw_root, str) or not raw_root.strip():
                continue
            root = Path(raw_root.strip()).expanduser().resolve()
            if root.is_file():
                if resolved_path == root:
                    return True
                continue
            try:
                resolved_path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _format_configured_roots(roots: Sequence[object]) -> str:
        return ", ".join(str(root) for root in roots if isinstance(root, str) and root.strip()) or "<none>"

    def _require_path_under_configured_roots(
        *,
        path: str,
        roots_key: str,
        field_label: str,
        engine_id: str,
        asset_label: str,
    ) -> str:
        roots = tuple(get_paths_for(roots_key))
        if not roots:
            raise HTTPException(status_code=409, detail=f"No {roots_key} roots are configured.")
        path_text = str(path or "").strip()
        if not path_text:
            raise HTTPException(
                status_code=409,
                detail=f"'{field_label}' for engine '{engine_id}' resolved to an empty {asset_label} path.",
            )
        if not _path_is_under_configured_roots(path_text, roots):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{field_label}' for engine '{engine_id}' must resolve under {roots_key} roots; "
                    f"got {path_text}. Roots: {_format_configured_roots(roots)}."
                ),
            )
        return path_text

    def _require_zimage_l2p_checkpoint_record(
        *,
        checkpoint_record: Any,
        field_prefix: str,
    ) -> None:
        model_field = _asset_field_label(field_prefix=field_prefix, field_name="model_sha")
        family_hint = str(getattr(checkpoint_record, "family_hint", "") or "").strip().lower()
        metadata_raw = getattr(checkpoint_record, "metadata", None)
        metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
        signature_source = str(metadata.get("signature_source") or "").strip()
        format_raw = getattr(checkpoint_record, "format", None)
        format_value = str(getattr(format_raw, "value", format_raw) or "").strip().lower()
        expected_signature_source = (
            "zimage_l2p_gguf_profile" if format_value == "gguf" else "zimage_l2p_safetensors_header"
        )
        if family_hint != _ZIMAGE_L2P_ENGINE_ID or signature_source != expected_signature_source:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{model_field}' for engine '{_ZIMAGE_L2P_ENGINE_ID}' must resolve to a Z-Image L2P denoiser "
                    f"record discovered from zimage_l2p_ckpt roots; got family_hint={family_hint or '<missing>'!r} "
                    f"signature_source={signature_source or '<missing>'!r} expected={expected_signature_source!r}."
                ),
            )
        if format_value == "gguf":
            try:
                from apps.backend.runtime.model_registry.detectors.zimage_l2p import (
                    validate_zimage_l2p_gguf_component_metadata,
                )

                validate_zimage_l2p_gguf_component_metadata(metadata, component="denoiser")
            except Exception as exc:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"'{model_field}' for engine '{_ZIMAGE_L2P_ENGINE_ID}' must resolve to a GGUF generated "
                        f"with profile_id='zimage_l2p_denoiser': {exc}"
                    ),
                ) from exc
        filename = str(getattr(checkpoint_record, "filename", "") or "").strip()
        _require_path_under_configured_roots(
            path=filename,
            roots_key="zimage_l2p_ckpt",
            field_label=model_field,
            engine_id=_ZIMAGE_L2P_ENGINE_ID,
            asset_label="denoiser",
        )

    def _resolve_qwen_image_text_encoder_sha_to_path(
        *,
        tenc_sha: str,
        tenc_field: str,
        resolve_asset_by_sha,  # type: ignore[no-untyped-def]
    ) -> str:
        from apps.backend.core.contracts.text_encoder_slots import (
            TextEncoderSlotError,
            classify_text_encoder_slot,
        )

        tenc_path = resolve_asset_by_sha(tenc_sha)
        if not tenc_path:
            raise HTTPException(status_code=409, detail=f"Asset not found for sha: {tenc_sha}")
        resolved_tenc_path = _require_path_under_configured_roots(
            path=str(tenc_path),
            roots_key="qwen_image_tenc",
            field_label=tenc_field,
            engine_id=_QWEN_IMAGE_ENGINE_ID,
            asset_label="text encoder",
        )
        try:
            slot = classify_text_encoder_slot(resolved_tenc_path)
        except TextEncoderSlotError as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{tenc_field}' must resolve to the exact Qwen Image Edit-2511 Qwen2.5-VL GGUF: {exc}"
                ),
            ) from exc
        if slot != "qwen2_5_vl_7b":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{tenc_field}' requires slot 'qwen2_5_vl_7b'; "
                    f"got {slot!r} for {resolved_tenc_path!r}."
                ),
            )
        return resolved_tenc_path

    def _preflight_qwen_image_checkpoint_record(checkpoint_record: Any, *, model_field: str) -> None:
        family_hint = str(getattr(checkpoint_record, "family_hint", "") or "").strip().lower()
        if family_hint != _QWEN_IMAGE_ENGINE_ID:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{model_field}' must resolve to family '{_QWEN_IMAGE_ENGINE_ID}'; "
                    f"got {family_hint or '<missing>'!r}."
                ),
            )
        format_value = getattr(getattr(checkpoint_record, "format", None), "value", None)
        if str(format_value or "").strip().lower() != "gguf":
            raise HTTPException(
                status_code=409,
                detail=f"'{model_field}' must resolve to the exact Edit-2511 transformer GGUF.",
            )
        if not bool(getattr(checkpoint_record, "core_only", False)):
            raise HTTPException(
                status_code=409,
                detail=f"'{model_field}' must resolve to a core-only Qwen Image transformer checkpoint.",
            )
        core_only_reason = str(getattr(checkpoint_record, "core_only_reason", "") or "").strip()
        if core_only_reason != "qwen_image_edit_2511_gguf_profile_core_only":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{model_field}' lacks the exact Edit-2511 core-only admission profile; "
                    f"got reason={core_only_reason or '<missing>'!r}."
                ),
            )

        metadata = getattr(checkpoint_record, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise HTTPException(
                status_code=409,
                detail=f"'{model_field}' is missing Qwen Image Edit-2511 checkpoint metadata.",
            )
        expected_metadata = {
            "requires_vae": True,
            "qwen_image_variant": _QWEN_IMAGE_EDIT_VARIANT,
            "zero_cond_t": True,
            "signature_source": "qwen_image_edit_2511_gguf_profile",
            "codex.quant_policy": "qwen_image_mq_v2",
            "codex.quant_policy_preset": "MQ",
            "model.architecture": "qwen_image",
            "model.name": "Qwen/Qwen-Image-Edit-2511",
        }
        for key, expected in expected_metadata.items():
            actual = metadata.get(key)
            if actual != expected:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"'{model_field}' metadata mismatch for {key!r}: "
                        f"got={actual!r} expected={expected!r}."
                    ),
                )
        slots = metadata.get("text_encoder_slots")
        if not isinstance(slots, list) or slots != ["qwen2_5_vl_7b"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{model_field}' must declare text_encoder_slots=['qwen2_5_vl_7b']; "
                    f"got={slots!r}."
                ),
            )
        file_size = getattr(checkpoint_record, "file_size", None)
        if file_size != 13_233_594_912:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{model_field}' Edit-2511 transformer file-size mismatch: "
                    f"got={file_size!r} expected=13233594912."
                ),
            )
        filename = str(getattr(checkpoint_record, "filename", "") or "").strip()
        _require_path_under_configured_roots(
            path=filename,
            roots_key="qwen_image_ckpt",
            field_label=model_field,
            engine_id=_QWEN_IMAGE_ENGINE_ID,
            asset_label="transformer",
        )

    def _resolve_zimage_l2p_text_encoder_sha_to_path(
        *,
        tenc_sha: str,
        tenc_field: str,
        resolve_asset_by_sha,  # type: ignore[no-untyped-def]
    ) -> str:
        tenc_path = resolve_asset_by_sha(tenc_sha)
        if not tenc_path:
            raise HTTPException(status_code=409, detail=f"Asset not found for sha: {tenc_sha}")
        resolved_tenc_path = _require_path_under_configured_roots(
            path=str(tenc_path),
            roots_key="zimage_tenc",
            field_label=tenc_field,
            engine_id=_ZIMAGE_L2P_ENGINE_ID,
            asset_label="Qwen3-4B text encoder",
        )
        if Path(resolved_tenc_path).suffix.lower() == ".gguf":
            try:
                from apps.backend.runtime.checkpoint.io import read_gguf_metadata
                from apps.backend.runtime.model_registry.detectors.zimage_l2p import (
                    validate_zimage_l2p_gguf_component_metadata,
                )

                validate_zimage_l2p_gguf_component_metadata(
                    read_gguf_metadata(resolved_tenc_path),
                    component="tenc",
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"'{tenc_field}' for engine '{_ZIMAGE_L2P_ENGINE_ID}' must resolve to a GGUF generated "
                        f"with profile_id='zimage_l2p_tenc': {exc}"
                    ),
                ) from exc
        return resolved_tenc_path

    def _preflight_qwen_image_asset_selector_payload(raw_extras: Mapping[str, Any], *, field_prefix: str) -> None:
        from apps.backend.inventory.cache import resolve_asset_by_sha, resolve_vae_path_by_sha
        from apps.backend.runtime.models import api as _models_api

        model_field = _asset_field_label(field_prefix=field_prefix, field_name="model_sha")
        vae_field = _asset_field_label(field_prefix=field_prefix, field_name="vae_sha")
        vae_source_field = _asset_field_label(field_prefix=field_prefix, field_name="vae_source")
        tenc_field = _asset_field_label(field_prefix=field_prefix, field_name="tenc_sha")

        if raw_extras.get("checkpoint_core_only") is not True:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Engine '{_QWEN_IMAGE_ENGINE_ID}' requires "
                    f"'{field_prefix}.checkpoint_core_only' set to true."
                ),
            )
        model_format_raw = raw_extras.get("model_format")
        if not isinstance(model_format_raw, str) or model_format_raw.strip().lower() != "gguf":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Engine '{_QWEN_IMAGE_ENGINE_ID}' requires "
                    f"'{field_prefix}.model_format' set to 'gguf'."
                ),
            )
        model_sha = _normalize_sha_field(raw_extras.get("model_sha"), field_label=model_field)
        if not model_sha:
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_QWEN_IMAGE_ENGINE_ID}' requires '{model_field}' (sha256)",
            )
        checkpoint_record = _models_api.find_checkpoint_by_sha(model_sha)
        if checkpoint_record is None:
            raise HTTPException(status_code=409, detail=f"Checkpoint not found for sha: {model_sha}")
        _preflight_qwen_image_checkpoint_record(checkpoint_record, model_field=model_field)

        vae_source_raw = raw_extras.get("vae_source")
        if not isinstance(vae_source_raw, str) or vae_source_raw.strip().lower() != "external":
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_QWEN_IMAGE_ENGINE_ID}' requires '{vae_source_field}' set to 'external'.",
            )
        vae_sha = _normalize_sha_field(raw_extras.get("vae_sha"), field_label=vae_field)
        if not vae_sha:
            raise HTTPException(status_code=400, detail=f"Engine '{_QWEN_IMAGE_ENGINE_ID}' requires '{vae_field}' (sha256)")
        _resolve_qwen_image_vae_sha_to_path(
            vae_sha=vae_sha,
            vae_field=vae_field,
            resolve_asset_by_sha=resolve_asset_by_sha,
            resolve_vae_path_by_sha=resolve_vae_path_by_sha,
        )

        tenc_shas = _normalize_sha_list_field(raw_extras.get("tenc_sha"), field_label=tenc_field)
        if len(tenc_shas) != 1:
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_QWEN_IMAGE_ENGINE_ID}' requires exactly 1 text encoder via '{tenc_field}'.",
            )
        _resolve_qwen_image_text_encoder_sha_to_path(
            tenc_sha=tenc_shas[0],
            tenc_field=tenc_field,
            resolve_asset_by_sha=resolve_asset_by_sha,
        )

    def _apply_asset_contract_to_extras(
        *,
        engine_id: str,
        checkpoint_record: Any,
        extras: Dict[str, Any],
        field_prefix: str,
        require_explicit_checkpoint_contract: bool,
        resolve_asset_by_sha,  # type: ignore[no-untyped-def]
        resolve_vae_path_by_sha,  # type: ignore[no-untyped-def]
    ) -> None:
        vae_field = _asset_field_label(field_prefix=field_prefix, field_name="vae_sha")
        vae_source_field = _asset_field_label(field_prefix=field_prefix, field_name="vae_source")
        tenc_field = _asset_field_label(field_prefix=field_prefix, field_name="tenc_sha")
        tenc1_field = _asset_field_label(field_prefix=field_prefix, field_name="tenc1_sha")
        tenc2_field = _asset_field_label(field_prefix=field_prefix, field_name="tenc2_sha")
        lora_field = _asset_field_label(field_prefix=field_prefix, field_name="lora_sha")
        text_encoder_override_field = _asset_field_label(field_prefix=field_prefix, field_name="text_encoder_override")
        path_scope = f"{field_prefix}.*_path" if field_prefix else "*_path"
        sha_scope = f"{field_prefix}.*_sha" if field_prefix else "*_sha"

        if "vae_path" in extras or "tenc_path" in extras:
            raise HTTPException(
                status_code=400,
                detail=f"Payload must not include raw '{path_scope}' fields; use sha256 via '{sha_scope}'",
            )

        if engine_id in ("flux1", "flux1_kontext") and "text_encoder_override" in extras:
            raise HTTPException(
                status_code=400,
                detail=f"Do not send {text_encoder_override_field} for Flux.1; use {tenc_field} only.",
            )
        if engine_id in ("sdxl", "sdxl_refiner") and "text_encoder_override" in extras:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Do not send {text_encoder_override_field} for SDXL; use explicit "
                    f"{tenc1_field}/{tenc2_field} for core-only checkpoints or {tenc_field} otherwise."
                ),
            )
        if engine_id == _QWEN_IMAGE_ENGINE_ID and "text_encoder_override" in extras:
            raise HTTPException(
                status_code=400,
                detail=f"Do not send {text_encoder_override_field} for Qwen Image; use {tenc_field} only.",
            )

        checkpoint_core_only, _model_format = _resolve_checkpoint_contract_from_request(
            checkpoint_record=checkpoint_record,
            extras=extras,
            field_prefix=field_prefix,
            require_explicit_contract=require_explicit_checkpoint_contract,
        )
        if engine_id == _ZIMAGE_L2P_ENGINE_ID:
            _require_zimage_l2p_checkpoint_record(
                checkpoint_record=checkpoint_record,
                field_prefix=field_prefix,
            )
        try:
            contract = contract_for_request(engine_id=engine_id, checkpoint_core_only=checkpoint_core_only)
        except Exception as exc:
            _router_log.exception("asset contract resolution failed for engine '%s'", engine_id)
            raise HTTPException(
                status_code=500,
                detail=public_http_error_detail(
                    exc,
                    fallback=f"Asset contract resolution failed for engine '{engine_id}'",
                ),
            ) from exc

        vae_sha = _normalize_sha_field(extras.get("vae_sha"), field_label=vae_field)
        vae_source_raw = extras.get("vae_source")
        vae_source: str | None = None
        if not contract.uses_vae:
            if vae_source_raw is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Engine '{engine_id}' does not use '{vae_source_field}'.",
                )
            if vae_sha:
                raise HTTPException(
                    status_code=400,
                    detail=f"Engine '{engine_id}' does not use '{vae_field}'.",
                )
        else:
            if vae_source_raw is not None:
                if not isinstance(vae_source_raw, str) or not vae_source_raw.strip():
                    raise HTTPException(
                        status_code=400,
                        detail=f"'{vae_source_field}' must be 'built_in' or 'external'",
                    )
                vae_source = vae_source_raw.strip().lower()
                if vae_source not in {"built_in", "external"}:
                    raise HTTPException(
                        status_code=400,
                        detail=f"'{vae_source_field}' must be 'built_in' or 'external'",
                    )
                extras["vae_source"] = vae_source
            elif require_explicit_checkpoint_contract:
                raise HTTPException(status_code=400, detail=f"Missing '{vae_source_field}'")
        tenc_shas = _normalize_sha_list_field(extras.get("tenc_sha"), field_label=tenc_field)
        lora_shas = _normalize_sha_list_field(extras.get("lora_sha"), field_label=lora_field)
        explicit_tenc1_sha = _normalize_sha_field(extras.get("tenc1_sha"), field_label=tenc1_field)
        explicit_tenc2_sha = _normalize_sha_field(extras.get("tenc2_sha"), field_label=tenc2_field)

        uses_explicit_sdxl_tenc_fields = engine_id in ("sdxl", "sdxl_refiner") and checkpoint_core_only
        if uses_explicit_sdxl_tenc_fields:
            if tenc_shas:
                raise HTTPException(
                    status_code=400,
                    detail=f"Do not send '{tenc_field}' for SDXL core-only checkpoints; use '{tenc1_field}'/'{tenc2_field}'.",
                )
            expected_slots = tuple(contract.tenc_slots or ())
            if len(expected_slots) == 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal error: SDXL core-only asset contract for engine '{engine_id}' is missing text encoder slots.",
                )
            explicit_fields = [explicit_tenc1_sha, explicit_tenc2_sha]
            required_fields = [tenc1_field, tenc2_field]
            tenc_shas = []
            for index, slot in enumerate(expected_slots):
                if index >= len(explicit_fields):
                    raise HTTPException(
                        status_code=500,
                        detail=f"Internal error: unsupported SDXL core-only slot count for engine '{engine_id}'.",
                    )
                explicit_sha = explicit_fields[index]
                if not explicit_sha:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Engine '{engine_id}' requires '{required_fields[index]}' "
                            f"for text encoder slot '{slot}'."
                        ),
                    )
                tenc_shas.append(explicit_sha)
            if len(expected_slots) < 2 and explicit_tenc2_sha:
                raise HTTPException(
                    status_code=400,
                    detail=f"Engine '{engine_id}' does not allow '{tenc2_field}'.",
                )
        elif explicit_tenc1_sha or explicit_tenc2_sha:
            raise HTTPException(
                status_code=400,
                detail=f"'{tenc1_field}' and '{tenc2_field}' are only allowed for SDXL core-only checkpoints.",
            )

        if contract.uses_vae:
            if vae_source == "external" and not vae_sha:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{vae_source_field}' set to 'external' requires '{vae_field}' (sha256)",
                )
            if vae_source == "built_in" and vae_sha:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{vae_source_field}' set to 'built_in' does not allow '{vae_field}'",
                )
        if contract.requires_vae:
            if vae_source == "built_in":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Engine '{engine_id}' requires an external VAE via '{vae_field}' "
                        f"and does not allow '{vae_source_field}=built_in'."
                    ),
                )
            if not vae_sha:
                raise HTTPException(status_code=400, detail=f"Engine '{engine_id}' requires '{vae_field}' (sha256)")

        if contract.requires_text_encoders:
            if len(tenc_shas) == 0:
                raise HTTPException(status_code=400, detail=f"Engine '{engine_id}' requires '{tenc_field}' (sha256)")
            if len(tenc_shas) != int(contract.tenc_count):
                raise HTTPException(
                    status_code=400,
                    detail=_format_required_tenc_message(engine_id=engine_id, contract=contract, field_label=tenc_field),
                )

        if vae_sha:
            if engine_id == _QWEN_IMAGE_ENGINE_ID:
                vae_path = _resolve_qwen_image_vae_sha_to_path(
                    vae_sha=vae_sha,
                    vae_field=vae_field,
                    resolve_asset_by_sha=resolve_asset_by_sha,
                    resolve_vae_path_by_sha=resolve_vae_path_by_sha,
                )
            else:
                vae_path = resolve_vae_path_by_sha(vae_sha)
                if not vae_path:
                    non_vae_path = resolve_asset_by_sha(vae_sha)
                    if non_vae_path:
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                f"'{vae_field}' resolved to a non-VAE asset path: {non_vae_path}. "
                                "Select a SHA from inventory.vaes."
                            ),
                        )
                    raise HTTPException(status_code=409, detail=f"Asset not found for sha: {vae_sha}")
            extras["vae_path"] = vae_path

        if tenc_shas:
            tenc_paths: list[str] = []
            for sha in tenc_shas:
                if engine_id == _QWEN_IMAGE_ENGINE_ID:
                    path = _resolve_qwen_image_text_encoder_sha_to_path(
                        tenc_sha=sha,
                        tenc_field=tenc_field,
                        resolve_asset_by_sha=resolve_asset_by_sha,
                    )
                elif engine_id == _ZIMAGE_L2P_ENGINE_ID:
                    path = _resolve_zimage_l2p_text_encoder_sha_to_path(
                        tenc_sha=sha,
                        tenc_field=tenc_field,
                        resolve_asset_by_sha=resolve_asset_by_sha,
                    )
                else:
                    path = resolve_asset_by_sha(sha)
                    if not path:
                        raise HTTPException(status_code=409, detail=f"Asset not found for sha: {sha}")
                tenc_paths.append(path)

            slot_to_path: dict[str, str] | None = None
            if contract.requires_text_encoders and contract.tenc_slots:
                from apps.backend.core.contracts.text_encoder_slots import (
                    TextEncoderSlotError,
                    classify_text_encoder_slot,
                )
                from apps.backend.inventory.cache import resolve_text_encoder_slot_by_sha

                try:
                    expected = tuple(contract.tenc_slots)
                    slot_to_path = {}
                    for sha, path in zip(tenc_shas, tenc_paths):
                        slot = resolve_text_encoder_slot_by_sha(sha) or ""
                        if not slot:
                            slot = classify_text_encoder_slot(path)
                        if slot not in expected:
                            raise TextEncoderSlotError(
                                f"Text encoder slot mismatch: got slot={slot!r} for sha={sha!r}, expected one of {list(expected)}."
                            )
                        if slot in slot_to_path:
                            raise TextEncoderSlotError(
                                f"Duplicate text encoder slot {slot!r} for slots={list(expected)} (sha={sha!r})."
                            )
                        slot_to_path[slot] = path

                    missing = [slot for slot in expected if slot not in slot_to_path]
                    if missing:
                        raise TextEncoderSlotError(
                            f"Missing required text encoder slot(s) {missing} for slots={list(expected)} (classified={sorted(slot_to_path)})."
                        )
                except TextEncoderSlotError as exc:
                    _router_log.warning("text encoder slot validation failed for engine '%s': %s", engine_id, exc)
                    raise HTTPException(
                        status_code=400,
                        detail=public_http_error_detail(
                            exc,
                            fallback="Invalid text encoder slot mapping for requested assets",
                        ),
                    ) from exc

                # Normalize order to the canonical slot list so downstream code never depends on user-provided ordering.
                tenc_paths = [slot_to_path[slot] for slot in contract.tenc_slots]

            extras["tenc_path"] = tenc_paths[0] if len(tenc_paths) == 1 else tenc_paths

            # Flux.1/Kontext: translate sha-selected encoders into a loader override (paths stay server-side).
            if engine_id in ("flux1", "flux1_kontext"):
                if slot_to_path is None:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Internal error: slot mapping missing for engine '{engine_id}'",
                    )
                extras["text_encoder_override"] = {
                    "family": engine_id,
                    "label": f"{engine_id}/sha",
                    "components": [f"{slot}={slot_to_path[slot]}" for slot in contract.tenc_slots],
                }

        if lora_shas:
            try:
                semantic_engine = semantic_engine_for_engine_id(engine_id)
            except KeyError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown engine id for '{lora_field}': {engine_id!r}",
                ) from exc
            if not ENGINE_SURFACES[semantic_engine].supports_lora:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{lora_field}' is unsupported for engine '{engine_id}'.",
                )

            from apps.backend.inventory.scanners.loras import iter_lora_files

            known_lora_paths = {_normalize_path_for_compare(path) for path in iter_lora_files()}
            if not known_lora_paths:
                raise HTTPException(
                    status_code=409,
                    detail=f"'{lora_field}' was provided, but no LoRA assets are available in inventory.",
                )

            resolved_lora_paths: list[str] = []
            seen_lora_paths: set[str] = set()
            for sha in lora_shas:
                path = resolve_asset_by_sha(sha)
                if not path:
                    raise HTTPException(status_code=409, detail=f"Asset not found for sha: {sha}")
                canonical = _normalize_path_for_compare(path)
                if not canonical.lower().endswith(".safetensors"):
                    raise HTTPException(
                        status_code=409,
                        detail=f"'{lora_field}' must resolve to a .safetensors LoRA file: {sha}",
                    )
                if canonical not in known_lora_paths:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"'{lora_field}' resolved to a non-LoRA asset path: {path}. "
                            "Select a SHA from inventory.loras."
                        ),
                    )
                if canonical in seen_lora_paths:
                    continue
                resolved_lora_paths.append(path)
                seen_lora_paths.add(canonical)
            if resolved_lora_paths:
                extras["lora_path"] = resolved_lora_paths[0] if len(resolved_lora_paths) == 1 else resolved_lora_paths

    def _resolve_stage_model_selection_in_place(
        *,
        engine_id: str,
        selection: Dict[str, Any],
        field_prefix: str,
        models_api: Any,
        resolve_asset_by_sha,
        resolve_vae_path_by_sha,
    ) -> None:
        model_override = selection.get("model")
        resolved_model_ref, checkpoint_record = _resolve_checkpoint_selection(
            model_override=model_override,
            extras=selection,
            field_prefix=field_prefix,
            models_api=models_api,
        )
        selection["model"] = resolved_model_ref
        selection.pop("model_path", None)
        _apply_asset_contract_to_extras(
            engine_id=engine_id,
            checkpoint_record=checkpoint_record,
            extras=selection,
            field_prefix=field_prefix,
            require_explicit_checkpoint_contract=True,
            resolve_asset_by_sha=resolve_asset_by_sha,
            resolve_vae_path_by_sha=resolve_vae_path_by_sha,
        )


    def _resolve_nested_stage_model_payloads(
        *,
        engine_id: str,
        extras: Dict[str, Any],
        hires_cfg: Dict[str, Any] | None,
        models_api: Any,
        resolve_asset_by_sha,
        resolve_vae_path_by_sha,
    ) -> None:
        if isinstance(extras.get("swap_model"), dict):
            _resolve_stage_model_selection_in_place(
                engine_id=engine_id,
                selection=extras["swap_model"],
                field_prefix="extras.swap_model",
                models_api=models_api,
                resolve_asset_by_sha=resolve_asset_by_sha,
                resolve_vae_path_by_sha=resolve_vae_path_by_sha,
            )

        semantic_engine = semantic_engine_for_engine_id(engine_id)
        refiner_engine_id = "sdxl_refiner"
        if isinstance(extras.get("refiner"), dict):
            if not ENGINE_SURFACES[semantic_engine].supports_refiner:
                raise HTTPException(
                    status_code=400,
                    detail=f"'extras.refiner' is unsupported for engine '{engine_id}'.",
                )
            _resolve_stage_model_selection_in_place(
                engine_id=refiner_engine_id,
                selection=extras["refiner"],
                field_prefix="extras.refiner",
                models_api=models_api,
                resolve_asset_by_sha=resolve_asset_by_sha,
                resolve_vae_path_by_sha=resolve_vae_path_by_sha,
            )

        if not isinstance(hires_cfg, dict):
            return
        if isinstance(hires_cfg.get("swap_model"), dict):
            _resolve_stage_model_selection_in_place(
                engine_id=engine_id,
                selection=hires_cfg["swap_model"],
                field_prefix="extras.hires.swap_model",
                models_api=models_api,
                resolve_asset_by_sha=resolve_asset_by_sha,
                resolve_vae_path_by_sha=resolve_vae_path_by_sha,
            )
        if isinstance(hires_cfg.get("refiner"), dict):
            if not ENGINE_SURFACES[semantic_engine].supports_refiner:
                raise HTTPException(
                    status_code=400,
                    detail=f"'extras.hires.refiner' is unsupported for engine '{engine_id}'.",
                )
            _resolve_stage_model_selection_in_place(
                engine_id=refiner_engine_id,
                selection=hires_cfg["refiner"],
                field_prefix="extras.hires.refiner",
                models_api=models_api,
                resolve_asset_by_sha=resolve_asset_by_sha,
                resolve_vae_path_by_sha=resolve_vae_path_by_sha,
            )

    @dataclass(frozen=True, slots=True)
    class _Txt2ImgPayloadDTO:
        engine_key: str
        prompt: str
        negative_prompt: str
        width: int
        height: int
        steps: int
        cfg_scale: float
        distilled_cfg_scale: float
        sampler_name: str
        scheduler_name: str
        seed: int
        clip_skip: int | None

    @dataclass(frozen=True, slots=True)
    class _Img2ImgCoreDTO:
        engine_key: str
        model_ref: Any
        prompt: Any
        negative_prompt: Any
        styles: List[Any]
        batch_count: int
        batch_size: int
        steps: int
        cfg_scale: float
        distilled_cfg_scale: float | None
        image_cfg_scale: float | None
        denoise: float
        width: int
        height: int
        sampler_name: str
        scheduler_name: str
        seed: int
        clip_skip: int | None
        noise_source: str | None
        ensd_raw: Any

    @dataclass(frozen=True, slots=True)
    class _VideoCoreDTO:
        prompt: str
        negative_prompt: str
        width: int
        height: int
        steps: int
        fps: int
        num_frames: int
        sampler_name: str
        scheduler_name: str
        seed: int | None
        guidance_scale: float

    def _parse_txt2img_payload_dto(payload: Dict[str, Any]) -> _Txt2ImgPayloadDTO:
        _reject_public_qwen_image_variant(payload, context="txt2img")
        _reject_unknown_keys(payload, _TXT2IMG_ALLOWED_KEYS, "txt2img")
        engine_override = payload.get('engine')
        engine_key = _canonical_engine_key(engine_override)
        if not engine_key:
            raise HTTPException(status_code=400, detail="Missing engine key (engine)")
        _reject_not_implemented_engine(engine_key, field_name="engine")
        if _is_qwen_image_engine(engine_key):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Engine 'qwen_image' is Edit-2511 img2img-only. "
                    "Use POST /api/img2img with one init image."
                ),
            )

        prompt = _require_str_field(payload, 'prompt', allow_empty=True)
        negative_prompt = str(payload.get('negative_prompt') or '')
        width = _require_int_field(payload, 'width', minimum=8)
        height = _require_int_field(payload, 'height', minimum=8)
        steps_val = _require_int_field(payload, 'steps', minimum=1)
        if engine_key == "flux2":
            cfg_scale, distilled_cfg_scale = _parse_flux2_submitted_guidance(
                payload=payload,
                cfg_field="cfg",
                distilled_cfg_field="distilled_cfg",
                distilled_cfg_default=3.5,
            )
        elif not engine_supports_cfg(engine_key):
            if 'cfg' in payload:
                raise HTTPException(
                    status_code=400,
                    detail=f"Engine '{engine_key}' does not accept 'cfg'; use 'distilled_cfg'.",
                )
            if 'distilled_cfg' not in payload:
                raise HTTPException(status_code=400, detail=f"Engine '{engine_key}' requires 'distilled_cfg'.")
            # Flow models (Flux/Chroma) use distilled guidance (no classic CFG); keep cfg neutral.
            cfg_scale = 1.0
            distilled_cfg_scale = _require_float_field(payload, 'distilled_cfg')
        else:
            if 'distilled_cfg' in payload:
                raise HTTPException(
                    status_code=400,
                    detail=f"Engine '{engine_key}' does not support 'distilled_cfg'; use 'cfg'.",
                )
            if 'cfg' not in payload:
                raise HTTPException(status_code=400, detail="Missing 'cfg'")
            # Z-Image uses classic CFG semantics (diffusers parity).
            cfg_scale = _require_float_field(payload, 'cfg')
            distilled_cfg_scale = 3.5
        sampler_name = _require_str_field(payload, 'sampler', allow_empty=False)
        scheduler_name = _require_str_field(payload, 'scheduler', allow_empty=False)
        _validate_er_sde_release_scope(
            engine_key=engine_key,
            sampler=sampler_name,
            field_name="sampler",
        )
        try:
            from apps.backend.runtime.sampling.registry import get_sampler_spec
            from apps.backend.runtime.sampling.context import SchedulerName

            spec = get_sampler_spec(str(sampler_name))
            SchedulerName.from_string(str(scheduler_name))
            if not spec.is_supported_scheduler(str(scheduler_name)):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Scheduler '{scheduler_name}' is not supported by sampler '{sampler_name}'. "
                        f"Allowed: {sorted(spec.allowed_schedulers)}"
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            _router_log.warning("txt2img sampler/scheduler validation failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid sampler/scheduler configuration"),
            ) from exc
        seed_val = _require_int_field(payload, 'seed')
        clip_skip = _require_int_field(payload, 'clip_skip', minimum=0, maximum=12) if 'clip_skip' in payload else None

        return _Txt2ImgPayloadDTO(
            engine_key=engine_key,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps_val,
            cfg_scale=cfg_scale,
            distilled_cfg_scale=distilled_cfg_scale,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            seed=seed_val,
            clip_skip=clip_skip,
        )

    def _parse_img2img_core_dto(
        payload: Dict[str, Any],
        *,
        init_w: int,
        init_h: int,
    ) -> _Img2ImgCoreDTO:
        engine_override = payload.get('engine')
        model_override = payload.get('model')
        engine_key = _canonical_engine_key(engine_override)
        if not engine_key:
            raise HTTPException(status_code=400, detail="Missing engine key (engine)")
        _reject_not_implemented_engine(engine_key, field_name="engine")
        model_ref = model_override

        prompt = _require_str_field(payload, "img2img_prompt", allow_empty=True)
        negative_prompt = _require_str_field(payload, "img2img_neg_prompt", allow_empty=True)
        styles = _p.as_list(payload, 'img2img_styles') if 'img2img_styles' in payload else []
        batch_count = _require_int_field(payload, "img2img_batch_count", minimum=1) if "img2img_batch_count" in payload else 1
        batch_size = _require_int_field(payload, "img2img_batch_size", minimum=1) if "img2img_batch_size" in payload else 1
        if 'img2img_steps' in payload:
            steps_val = _require_int_field(payload, "img2img_steps", minimum=1)
        else:
            raise HTTPException(status_code=400, detail="'img2img_steps' is required")

        if engine_key == "flux2":
            cfg_scale, distilled_cfg_scale = _parse_flux2_submitted_guidance(
                payload=payload,
                cfg_field="img2img_cfg_scale",
                distilled_cfg_field="img2img_distilled_cfg_scale",
                distilled_cfg_default=None,
            )
        elif engine_supports_cfg(engine_key):
            if 'img2img_cfg_scale' not in payload:
                raise HTTPException(status_code=400, detail="'img2img_cfg_scale' is required")
            if 'img2img_distilled_cfg_scale' in payload:
                raise HTTPException(
                    status_code=400,
                    detail=f"Engine '{engine_key}' does not support 'img2img_distilled_cfg_scale'; use 'img2img_cfg_scale'.",
                )
            cfg_scale = _require_float_field(payload, 'img2img_cfg_scale')
            distilled_cfg_scale = None
        else:
            if 'img2img_cfg_scale' in payload:
                raise HTTPException(
                    status_code=400,
                    detail=f"Engine '{engine_key}' does not support 'img2img_cfg_scale'; use 'img2img_distilled_cfg_scale'.",
                )
            if 'img2img_distilled_cfg_scale' not in payload:
                raise HTTPException(status_code=400, detail="'img2img_distilled_cfg_scale' is required")
            cfg_scale = 1.0
            distilled_cfg_scale = _require_float_field(payload, 'img2img_distilled_cfg_scale')
        image_cfg_scale = _require_float_field(payload, 'img2img_image_cfg_scale') if 'img2img_image_cfg_scale' in payload else None
        denoise = _require_float_field(payload, 'img2img_denoising_strength', minimum=0.0, maximum=1.0)

        dimension_multiple = _img2img_dimension_multiple_for_engine(engine_key)

        if 'img2img_width' in payload:
            width_val = _require_int_field(payload, "img2img_width", minimum=8, maximum=8192)
        else:
            width_val = _snap_dimension(
                int(init_w) if init_w else 0,
                multiple=dimension_multiple,
                strategy="floor" if engine_key == "zimage" else "nearest",
            )
            if not width_val:
                raise HTTPException(status_code=400, detail="'img2img_width' is required")

        if 'img2img_height' in payload:
            height_val = _require_int_field(payload, "img2img_height", minimum=8, maximum=8192)
        else:
            height_val = _snap_dimension(
                int(init_h) if init_h else 0,
                multiple=dimension_multiple,
                strategy="floor" if engine_key == "zimage" else "nearest",
            )
            if not height_val:
                raise HTTPException(status_code=400, detail="'img2img_height' is required")
        width_val, height_val = _normalize_img2img_dimensions_for_engine(
            engine_key,
            width_val,
            height_val,
        )
        sampler_name = _require_str_field(payload, "img2img_sampling")
        scheduler_name = _require_str_field(payload, "img2img_scheduler")
        _validate_er_sde_release_scope(
            engine_key=engine_key,
            sampler=sampler_name,
            field_name="img2img_sampling",
        )
        try:
            from apps.backend.runtime.sampling.registry import get_sampler_spec
            from apps.backend.runtime.sampling.context import SchedulerName

            spec = get_sampler_spec(str(sampler_name))
            SchedulerName.from_string(str(scheduler_name))
            if not spec.is_supported_scheduler(str(scheduler_name)):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Scheduler '{scheduler_name}' is not supported by sampler '{sampler_name}'. "
                        f"Allowed: {sorted(spec.allowed_schedulers)}"
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            _router_log.warning("img2img sampler/scheduler validation failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid sampler/scheduler configuration"),
            ) from exc
        seed_val = _require_int_field(payload, "img2img_seed")
        clip_skip = _require_int_field(payload, "img2img_clip_skip", minimum=0, maximum=12) if "img2img_clip_skip" in payload else None
        noise_source: str | None = None
        if "img2img_noise_source" in payload:
            noise_source = _parse_noise_source_field(
                payload.get("img2img_noise_source"),
                field_name="img2img_noise_source",
            )
        ensd_raw = payload.get('img2img_eta_noise_seed_delta')

        return _Img2ImgCoreDTO(
            engine_key=engine_key,
            model_ref=model_ref,
            prompt=prompt,
            negative_prompt=negative_prompt,
            styles=styles,
            batch_count=batch_count,
            batch_size=batch_size,
            steps=steps_val,
            cfg_scale=cfg_scale,
            distilled_cfg_scale=distilled_cfg_scale,
            image_cfg_scale=image_cfg_scale,
            denoise=denoise,
            width=width_val,
            height=height_val,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            seed=seed_val,
            clip_skip=clip_skip,
            noise_source=noise_source,
            ensd_raw=ensd_raw,
        )

    def _parse_qwen_image_img2img_core_dto(
        payload: Dict[str, Any],
        *,
        init_w: int,
        init_h: int,
    ) -> _Img2ImgCoreDTO:
        engine_key = _canonical_engine_key(payload.get("engine"))
        if not _is_qwen_image_engine(engine_key):
            raise HTTPException(status_code=400, detail=f"Engine '{engine_key}' is not a Qwen Image engine.")
        _validate_qwen_image_img2img_core_fields(payload)

        prompt = _require_str_field(payload, "img2img_prompt", allow_empty=True)
        negative_prompt = _require_str_field(payload, "img2img_neg_prompt", allow_empty=True)
        styles = _p.as_list(payload, "img2img_styles") if "img2img_styles" in payload else []
        steps_val = _require_int_field(payload, "img2img_steps", minimum=2)
        if "img2img_cfg_scale" not in payload:
            raise HTTPException(status_code=400, detail="'img2img_cfg_scale' is required for Qwen Image Edit.")
        cfg_scale = _require_float_field(payload, "img2img_cfg_scale")
        sampler_name = _require_str_field(payload, "img2img_sampling")
        scheduler_name = _require_str_field(payload, "img2img_scheduler")
        _require_qwen_image_sampling_pair(
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            sampler_field="img2img_sampling",
            scheduler_field="img2img_scheduler",
        )
        seed_val = _require_int_field(payload, "img2img_seed")
        try:
            width_val, height_val = qwen_image_edit_vae_dimensions(init_w, init_h)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        return _Img2ImgCoreDTO(
            engine_key=engine_key,
            model_ref=None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            styles=styles,
            batch_count=1,
            batch_size=1,
            steps=steps_val,
            cfg_scale=cfg_scale,
            distilled_cfg_scale=None,
            image_cfg_scale=None,
            denoise=1.0,
            width=width_val,
            height=height_val,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            seed=seed_val,
            clip_skip=None,
            noise_source=None,
            ensd_raw=None,
        )

    def _extract_wan22_unipc_solver_hint(sampler_value: str | None) -> str | None:
        normalized = str(sampler_value or "").strip().lower()
        if not normalized:
            return None
        parts = normalized.split()
        if len(parts) == 2 and parts[0] == "uni-pc":
            return parts[1]
        return None

    def _validate_wan22_sampler_field(
        *,
        field_name: str,
        value: str,
        expected_unipc_solver_hint: str | None = None,
    ) -> str:
        if not isinstance(value, str):
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' must be a string.",
            )
        normalized = value.strip().lower()
        if not normalized:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' must not be empty when provided.",
            )
        from apps.backend.types.samplers import SamplerKind

        parts = normalized.split()
        sampler_name = parts[0]
        solver_hint = parts[1] if len(parts) == 2 else None

        if sampler_name == SamplerKind.UNI_PC.value:
            if len(parts) > 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{field_name}' must be 'uni-pc' or 'uni-pc <solver_hint>'; got {value!r}.",
                )
            if solver_hint is None:
                return SamplerKind.UNI_PC.value
            if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", solver_hint) is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{field_name}' has invalid UniPC solver hint {solver_hint!r}; "
                        "use lowercase [a-z0-9._-] tokens only."
                    ),
                )
            if expected_unipc_solver_hint is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{field_name}' solver hint {solver_hint!r} is unsupported by WAN metadata; "
                        "metadata scheduler has no solver_type."
                    ),
                )
            if solver_hint != expected_unipc_solver_hint:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{field_name}' solver hint {solver_hint!r} does not match "
                        f"WAN metadata solver_type {expected_unipc_solver_hint!r}."
                    ),
                )
            return f"{SamplerKind.UNI_PC.value} {solver_hint}"

        try:
            sampler_kind = SamplerKind.from_string(normalized)
        except Exception as exc:
            _router_log.warning("%s sampler validation failed: %s", field_name, exc)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{field_name}' must resolve to a WAN22 sampler lane "
                    "('uni-pc' with optional solver hint, 'euler', or 'euler a')."
                ),
            ) from exc

        if sampler_kind is SamplerKind.UNI_PC:
            return sampler_kind.value
        if sampler_kind is SamplerKind.EULER:
            return sampler_kind.value
        if sampler_kind is SamplerKind.EULER_A:
            return sampler_kind.value

        raise HTTPException(
            status_code=400,
            detail=(
                f"'{field_name}' must resolve to a WAN22 sampler lane "
                "('uni-pc' with optional solver hint, 'euler', or 'euler a')."
            ),
        )

    def _validate_wan22_scheduler_field(*, field_name: str, value: str) -> str:
        try:
            from apps.backend.runtime.sampling.context import SchedulerName

            parsed_scheduler = SchedulerName.from_string(value)
        except Exception as exc:
            _router_log.warning("%s scheduler validation failed: %s", field_name, exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid WAN22 scheduler configuration"),
            ) from exc
        if parsed_scheduler.value != SchedulerName.SIMPLE.value:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' must be 'simple' for WAN22 requests; got {parsed_scheduler.value!r}.",
            )
        return parsed_scheduler.value

    def _parse_video_core_dto(
        payload: Dict[str, Any],
        *,
        task_prefix: str,
        default_width: int,
        default_height: int,
        default_steps: int,
        default_fps: int,
        default_frames: int,
        default_sampler: str,
        default_scheduler: str,
        expected_unipc_solver_hint: str | None,
        sampler_validator: Callable[[str, str], str] | None = None,
        scheduler_validator: Callable[[str, str], str] | None = None,
        default_seed: int,
        default_cfg_scale: float,
    ) -> _VideoCoreDTO:
        prompt_key = f"{task_prefix}_prompt"
        negative_prompt_key = f"{task_prefix}_neg_prompt"
        width_key = f"{task_prefix}_width"
        height_key = f"{task_prefix}_height"
        steps_key = f"{task_prefix}_steps"
        fps_key = f"{task_prefix}_fps"
        frames_key = f"{task_prefix}_num_frames"
        sampler_key = f"{task_prefix}_sampler"
        scheduler_key = f"{task_prefix}_scheduler"
        seed_key = f"{task_prefix}_seed"
        cfg_key = f"{task_prefix}_cfg_scale"

        prompt = _require_str_field(payload, prompt_key, allow_empty=True) if prompt_key in payload else ""
        negative_prompt = _require_str_field(payload, negative_prompt_key, allow_empty=True) if negative_prompt_key in payload else ""
        width_val = _require_int_field(payload, width_key, minimum=16, maximum=8192) if width_key in payload else int(default_width)
        height_val = _require_int_field(payload, height_key, minimum=16, maximum=8192) if height_key in payload else int(default_height)
        _wan_require_dims_multiple_of_16(task=task_prefix, width=width_val, height=height_val)
        steps_val = _require_int_field(payload, steps_key, minimum=1) if steps_key in payload else int(default_steps)
        fps_val = _require_int_field(payload, fps_key, minimum=1) if fps_key in payload else int(default_fps)
        frames_val = _require_int_field(payload, frames_key, minimum=9, maximum=401) if frames_key in payload else int(default_frames)
        if frames_val < 9 or frames_val > 401:
            raise HTTPException(
                status_code=400,
                detail=f"'{task_prefix}_num_frames' must be within [9, 401] (4n+1 domain), got {frames_val}.",
            )
        if (frames_val - 1) % 4 != 0:
            raise HTTPException(
                status_code=400,
                detail=f"'{task_prefix}_num_frames' must satisfy 4n+1, got {frames_val}.",
            )
        sampler_name = _require_str_field(payload, sampler_key) if sampler_key in payload else str(default_sampler)
        scheduler_name = _require_str_field(payload, scheduler_key) if scheduler_key in payload else str(default_scheduler)
        if sampler_validator is None:
            sampler_name = _validate_wan22_sampler_field(
                field_name=sampler_key,
                value=sampler_name,
                expected_unipc_solver_hint=expected_unipc_solver_hint,
            )
        else:
            sampler_name = sampler_validator(sampler_key, sampler_name)
        if scheduler_validator is None:
            scheduler_name = _validate_wan22_scheduler_field(field_name=scheduler_key, value=scheduler_name)
        else:
            scheduler_name = scheduler_validator(scheduler_key, scheduler_name)
        seed_val = _require_int_field(payload, seed_key) if seed_key in payload else int(default_seed)
        guidance_scale = _require_float_field(payload, cfg_key, minimum=0.0) if cfg_key in payload else float(default_cfg_scale)

        return _VideoCoreDTO(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width_val,
            height=height_val,
            steps=steps_val,
            fps=fps_val,
            num_frames=frames_val,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            seed=seed_val,
            guidance_scale=guidance_scale,
        )

    def _parse_txt2vid_core_dto(
        payload: Dict[str, Any],
        *,
        default_sampler: str = "uni-pc",
        default_scheduler: str = "simple",
        expected_unipc_solver_hint: str | None = None,
    ) -> _VideoCoreDTO:
        _reject_legacy_wan_request_key_aliases(payload, context="txt2vid")
        _reject_unknown_keys(payload, _TXT2VID_ALLOWED_KEYS, "txt2vid")
        return _parse_video_core_dto(
            payload,
            task_prefix='txt2vid',
            default_width=768,
            default_height=432,
            default_steps=30,
            default_fps=24,
            default_frames=17,
            default_sampler=default_sampler,
            default_scheduler=default_scheduler,
            expected_unipc_solver_hint=expected_unipc_solver_hint,
            default_seed=-1,
            default_cfg_scale=7.0,
        )

    def _parse_img2vid_core_dto(
        payload: Dict[str, Any],
        *,
        default_sampler: str = "uni-pc",
        default_scheduler: str = "simple",
        expected_unipc_solver_hint: str | None = None,
    ) -> _VideoCoreDTO:
        _reject_legacy_wan_request_key_aliases(payload, context="img2vid")
        _reject_unknown_keys(payload, _IMG2VID_ALLOWED_KEYS, "img2vid")
        return _parse_video_core_dto(
            payload,
            task_prefix='img2vid',
            default_width=768,
            default_height=432,
            default_steps=30,
            default_fps=24,
            default_frames=17,
            default_sampler=default_sampler,
            default_scheduler=default_scheduler,
            expected_unipc_solver_hint=expected_unipc_solver_hint,
            default_seed=-1,
            default_cfg_scale=7.0,
        )

    def _ltx2_require_dims_multiple_of_32(*, task: str, width: int, height: int) -> None:
        if height % 32 == 0 and width % 32 == 0:
            return
        raise HTTPException(
            status_code=400,
            detail=f"LTX2 {task}: width/height must be divisible by 32. Got {int(width)}x{int(height)}.",
        )

    def _ltx2_require_dims_multiple_of_64(*, task: str, width: int, height: int) -> None:
        if height % 64 == 0 and width % 64 == 0:
            return
        raise HTTPException(
            status_code=400,
            detail=(
                f"LTX2 {task} two_stage: final width/height must be divisible by 64 because stage 1 runs at half resolution. "
                f"Got {int(width)}x{int(height)}."
            ),
        )

    def _ltx2_require_frames_8n_plus_1(*, task: str, frames: int) -> None:
        if frames < 9 or frames > 401:
            raise HTTPException(
                status_code=400,
                detail=f"'{task}_num_frames' must be within [9, 401] (8n+1 domain), got {frames}.",
            )
        if (frames - 1) % 8 != 0:
            raise HTTPException(
                status_code=400,
                detail=f"'{task}_num_frames' must satisfy 8n+1, got {frames}.",
            )

    def _resolve_ltx2_requested_execution_profile(
        *,
        payload: Dict[str, Any],
        checkpoint_record: Any,
        defaults: Any,
    ) -> tuple[str, Any | None]:
        from apps.backend.runtime.model_registry.ltx2_execution import (
            LTX2_PROFILE_TWO_STAGE,
            build_ltx2_execution_surface,
            resolve_ltx2_two_stage_assets,
        )

        if _LTX2_EXECUTION_PROFILE_KEY not in payload:
            raise HTTPException(
                status_code=400,
                detail=f"'{_LTX2_EXECUTION_PROFILE_KEY}' is required for engine 'ltx2'.",
            )
        raw_profile = _require_str_field(payload, _LTX2_EXECUTION_PROFILE_KEY)
        execution_profile = raw_profile.strip()

        allowed_surface_profiles = set(build_ltx2_execution_surface().allowed_execution_profiles)
        if execution_profile not in allowed_surface_profiles:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{_LTX2_EXECUTION_PROFILE_KEY}' for engine 'ltx2' must be one of "
                    f"{sorted(allowed_surface_profiles)!r}, got {execution_profile!r}."
                ),
            )

        allowed_checkpoint_profiles = tuple(getattr(defaults, "allowed_execution_profiles", ()) or ())
        if execution_profile not in allowed_checkpoint_profiles:
            if execution_profile == LTX2_PROFILE_TWO_STAGE:
                stage2_assets = resolve_ltx2_two_stage_assets(checkpoint_record)
                reason = stage2_assets.blocked_reason or "Selected LTX2 checkpoint does not allow the two_stage profile."
                raise HTTPException(
                    status_code=409,
                    detail=f"Execution profile 'two_stage' is unsupported for the selected LTX2 checkpoint. {reason}",
                )
            raise HTTPException(
                status_code=409,
                detail=f"Execution profile {execution_profile!r} is unsupported for the selected LTX2 checkpoint.",
            )

        if execution_profile == LTX2_PROFILE_TWO_STAGE:
            stage2_assets = resolve_ltx2_two_stage_assets(checkpoint_record)
            if not stage2_assets.available:
                raise HTTPException(
                    status_code=409,
                    detail=stage2_assets.blocked_reason or "Execution profile 'two_stage' is unavailable for the selected LTX2 checkpoint.",
                )
            return execution_profile, stage2_assets

        return execution_profile, None

    def _parse_ltx2_generic_video_core_dto(
        payload: Dict[str, Any],
        *,
        task_prefix: str,
        execution_profile: str,
        default_steps: int,
        default_cfg_scale: float,
    ) -> _VideoCoreDTO:
        if task_prefix == "txt2vid":
            _reject_unknown_keys(payload, _LTX2_TXT2VID_GENERIC_ALLOWED_KEYS, "txt2vid")
        elif task_prefix == "img2vid":
            _reject_unknown_keys(payload, _LTX2_IMG2VID_GENERIC_ALLOWED_KEYS, "img2vid")
        else:
            raise RuntimeError(f"Unsupported LTX2 generic video task_prefix: {task_prefix!r}")

        prompt_key = f"{task_prefix}_prompt"
        negative_prompt_key = f"{task_prefix}_neg_prompt"
        width_key = f"{task_prefix}_width"
        height_key = f"{task_prefix}_height"
        steps_key = f"{task_prefix}_steps"
        fps_key = f"{task_prefix}_fps"
        frames_key = f"{task_prefix}_num_frames"
        seed_key = f"{task_prefix}_seed"
        cfg_key = f"{task_prefix}_cfg_scale"

        prompt = _require_str_field(payload, prompt_key, allow_empty=True) if prompt_key in payload else ""
        negative_prompt = _require_str_field(payload, negative_prompt_key, allow_empty=True) if negative_prompt_key in payload else ""
        width_val = _require_int_field(payload, width_key, minimum=32, maximum=8192) if width_key in payload else 768
        height_val = _require_int_field(payload, height_key, minimum=32, maximum=8192) if height_key in payload else 512
        _ltx2_require_dims_multiple_of_32(task=task_prefix, width=width_val, height=height_val)
        if execution_profile == "two_stage":
            _ltx2_require_dims_multiple_of_64(task=task_prefix, width=width_val, height=height_val)
        steps_val = _require_int_field(payload, steps_key, minimum=1) if steps_key in payload else int(default_steps)
        fps_val = _require_int_field(payload, fps_key, minimum=1) if fps_key in payload else 24
        frames_val = _require_int_field(payload, frames_key, minimum=9, maximum=401) if frames_key in payload else 121
        _ltx2_require_frames_8n_plus_1(task=task_prefix, frames=frames_val)
        seed_raw = _require_int_field(payload, seed_key) if seed_key in payload else -1
        seed_val = None if seed_raw < 0 else seed_raw
        guidance_scale = _require_float_field(payload, cfg_key, minimum=0.0) if cfg_key in payload else float(default_cfg_scale)

        return _VideoCoreDTO(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width_val,
            height=height_val,
            steps=steps_val,
            fps=fps_val,
            num_frames=frames_val,
            sampler_name="euler",
            scheduler_name="simple",
            seed=seed_val,
            guidance_scale=guidance_scale,
        )

    def _parse_generic_txt2vid_core_dto(payload: Dict[str, Any], *, engine_key: str) -> _VideoCoreDTO:
        _reject_legacy_wan_request_key_aliases(payload, context="txt2vid")
        _reject_unknown_keys(payload, _TXT2VID_GENERIC_ALLOWED_KEYS, "txt2vid")
        if _canonical_engine_key(engine_key) == "ltx2":
            raise RuntimeError("LTX2 generic txt2vid parsing requires checkpoint-owned defaults before parse.")
        return _parse_video_core_dto(
            payload,
            task_prefix='txt2vid',
            default_width=768,
            default_height=432,
            default_steps=30,
            default_fps=24,
            default_frames=17,
            default_sampler="uni-pc",
            default_scheduler="simple",
            expected_unipc_solver_hint=None,
            default_seed=-1,
            default_cfg_scale=7.0,
        )

    def _parse_generic_img2vid_core_dto(payload: Dict[str, Any], *, engine_key: str) -> _VideoCoreDTO:
        _reject_legacy_wan_request_key_aliases(payload, context="img2vid")
        _reject_unknown_keys(payload, _IMG2VID_GENERIC_ALLOWED_KEYS, "img2vid")
        if _canonical_engine_key(engine_key) == "ltx2":
            raise RuntimeError("LTX2 generic img2vid parsing requires checkpoint-owned defaults before parse.")
        return _parse_video_core_dto(
            payload,
            task_prefix='img2vid',
            default_width=768,
            default_height=432,
            default_steps=30,
            default_fps=24,
            default_frames=17,
            default_sampler="uni-pc",
            default_scheduler="simple",
            expected_unipc_solver_hint=None,
            default_seed=-1,
            default_cfg_scale=7.0,
        )

    def _copy_generic_video_asset_selector_fields(*, payload: Mapping[str, Any], extras: Dict[str, Any]) -> None:
        for key in ("model_sha", "checkpoint_core_only", "model_format", "vae_sha", "tenc_sha", "lora_sha"):
            if key in payload and payload.get(key) is not None:
                extras[key] = payload.get(key)

    def _resolve_generic_video_checkpoint_contract(
        *,
        payload: Mapping[str, Any],
        extras: Dict[str, Any],
        engine_key: str,
    ) -> tuple[str, Any]:
        from apps.backend.inventory.cache import resolve_asset_by_sha, resolve_vae_path_by_sha
        from apps.backend.runtime.models import api as _models_api

        _copy_generic_video_asset_selector_fields(payload=payload, extras=extras)
        model_ref = _resolve_model_ref_from_sha_or_name(
            model_override=payload.get("model"),
            extras=extras,
            field_prefix="",
            models_api=_models_api,
        )
        checkpoint_record = _models_api.find_checkpoint(model_ref)
        if checkpoint_record is None:
            raise HTTPException(
                status_code=409,
                detail=f"Selected checkpoint not found for engine '{engine_key}': {model_ref}",
            )
        _apply_asset_contract_to_extras(
            engine_id=engine_key,
            checkpoint_record=checkpoint_record,
            extras=extras,
            field_prefix="",
            require_explicit_checkpoint_contract=False,
            resolve_asset_by_sha=resolve_asset_by_sha,
            resolve_vae_path_by_sha=resolve_vae_path_by_sha,
        )
        if "vae_source" not in extras:
            vae_path = extras.get("vae_path")
            extras["vae_source"] = "external" if isinstance(vae_path, str) and vae_path.strip() else "built_in"
        _apply_gguf_video_runtime_controls_from_payload(payload=payload, extras=extras)
        return model_ref, checkpoint_record

    def _require_ltx2_checkpoint_execution_defaults(*, checkpoint_record: Any) -> Any:
        from apps.backend.runtime.model_registry.ltx2_execution import (
            LTX2_KIND_UNKNOWN,
            resolve_ltx2_checkpoint_execution_defaults,
        )

        defaults = resolve_ltx2_checkpoint_execution_defaults(checkpoint_record)
        if defaults.checkpoint_kind == LTX2_KIND_UNKNOWN:
            detail = defaults.blocked_reason or (
                "The checkpoint classified as 'unknown' from local signals and is blocked until a truthful lane exists."
            )
            raise HTTPException(
                status_code=409,
                detail=(
                    "Selected LTX2 checkpoint is unsupported by the current executable tranche: "
                    f"{getattr(checkpoint_record, 'title', getattr(checkpoint_record, 'name', '<unknown>'))!r}. "
                    f"{detail}"
                ),
            )
        return defaults

    def _preflight_generic_video_route_checkpoint_contract(payload: Mapping[str, Any]) -> None:
        video_engine_key = _canonical_engine_key(payload.get("engine")) if payload.get("engine") is not None else ""
        if video_engine_key != "ltx2":
            return
        if _is_legacy_or_wan_video_route_engine(video_engine_key):
            return
        extras: Dict[str, Any] = {}
        _, checkpoint_record = _resolve_generic_video_checkpoint_contract(
            payload=payload,
            extras=extras,
            engine_key=video_engine_key,
        )
        ltx_defaults = _require_ltx2_checkpoint_execution_defaults(checkpoint_record=checkpoint_record)
        _resolve_ltx2_requested_execution_profile(
            payload=payload,
            checkpoint_record=checkpoint_record,
            defaults=ltx_defaults,
        )

    def prepare_txt2img(payload: Dict[str, Any]) -> Tuple["Txt2ImgRequest", str, Optional[str]]:
        settings_revision = _require_int_field(payload, "settings_revision", minimum=0)
        model_override = payload.get('model')
        parsed = _parse_txt2img_payload_dto(payload)
        engine_key = parsed.engine_key
        _reject_zimage_l2p_txt2img_payload(payload)
        family_name, family_capability = _resolve_image_family_capability_contract(engine_key)
        engine_id = engine_key
        prompt = parsed.prompt
        negative_prompt = parsed.negative_prompt
        width = parsed.width
        height = parsed.height
        steps_val = parsed.steps
        cfg_scale = parsed.cfg_scale
        distilled_cfg_scale = parsed.distilled_cfg_scale
        sampler_name = parsed.sampler_name
        scheduler_name = parsed.scheduler_name
        _enforce_family_sampler_scheduler_support(
            engine_key=engine_key,
            family_name=family_name,
            family_capability=family_capability,
            sampler_name=str(sampler_name),
            scheduler_name=str(scheduler_name),
            sampler_field_name="sampler",
            scheduler_field_name="scheduler",
        )
        seed_val = parsed.seed
        clip_skip = parsed.clip_skip

        styles = _parse_styles(payload)
        metadata = _parse_metadata(payload)
        extras, hires_cfg = _parse_txt2img_extras(payload)
        _enforce_txt2img_ip_adapter_stage_support(
            engine_key=engine_key,
            extras=extras,
            hires_cfg=hires_cfg,
        )
        if hires_cfg is not None:
            hires_sampler = _parse_optional_sampler_field(value=hires_cfg.get("sampler"), field_name="extras.hires.sampler")
            if hires_sampler is not None:
                hires_cfg["sampler"] = hires_sampler
                _validate_er_sde_release_scope(
                    engine_key=engine_key,
                    sampler=hires_sampler,
                    field_name="extras.hires.sampler",
                )
            hires_scheduler = _parse_optional_scheduler_field(
                value=hires_cfg.get("scheduler"),
                field_name="extras.hires.scheduler",
            )
            if hires_sampler is not None or hires_scheduler is not None:
                resolved_hires_sampler, resolved_hires_scheduler = _resolve_hires_sampler_scheduler_override(
                    base_sampler=str(sampler_name),
                    base_scheduler=str(scheduler_name),
                    sampler_override=hires_sampler,
                    scheduler_override=hires_scheduler,
                    sampler_field_name="extras.hires.sampler",
                    scheduler_field_name="extras.hires.scheduler",
                )
                if hires_sampler is not None:
                    hires_cfg["sampler"] = resolved_hires_sampler
                hires_cfg["scheduler"] = resolved_hires_scheduler
                _enforce_family_sampler_scheduler_support(
                    engine_key=engine_key,
                    family_name=family_name,
                    family_capability=family_capability,
                    sampler_name=str(resolved_hires_sampler),
                    scheduler_name=str(resolved_hires_scheduler),
                    sampler_field_name="extras.hires.sampler",
                    scheduler_field_name="extras.hires.scheduler",
                )
            hires_refiner_cfg = hires_cfg.get("refiner")
            if isinstance(hires_refiner_cfg, dict):
                hires_total_steps = int(hires_cfg.get("steps") or 0)
                if hires_total_steps <= 0:
                    hires_total_steps = int(steps_val)
                _validate_swap_at_step_pointer(
                    pointer=int(hires_refiner_cfg.get("switch_at_step", 0)),
                    total_steps=hires_total_steps,
                    field_name="extras.hires.refiner.switch_at_step",
                )
        global_swap_model_cfg = extras.get("swap_model")
        if isinstance(global_swap_model_cfg, dict):
            _validate_swap_at_step_pointer(
                pointer=int(global_swap_model_cfg.get("switch_at_step", 0)),
                total_steps=int(steps_val),
                field_name="extras.swap_model.switch_at_step",
            )
        global_refiner_cfg = extras.get("refiner")
        if isinstance(global_refiner_cfg, dict):
            _validate_swap_at_step_pointer(
                pointer=int(global_refiner_cfg.get("switch_at_step", 0)),
                total_steps=int(steps_val),
                field_name="extras.refiner.switch_at_step",
            )

        # Read batch params from extras (default to 1)
        batch_size = int(extras.pop('batch_size', 1)) if 'batch_size' in extras else 1
        batch_count = int(extras.pop('batch_count', 1)) if 'batch_count' in extras else 1

        metadata["styles"] = styles
        metadata["n_iter"] = batch_count
        metadata["batch_count"] = batch_count
        metadata["batch_size"] = batch_size
        metadata["hr"] = bool(hires_cfg)
        metadata["distilled_cfg_scale"] = distilled_cfg_scale

        smart_offload, smart_fallback, smart_cache = _resolve_smart_flags()

        # Resolve model assets from SHA (if provided in extras)
        from apps.backend.inventory.cache import resolve_asset_by_sha, resolve_vae_path_by_sha
        from apps.backend.runtime.models import api as _models_api
        model_override, checkpoint_record = _resolve_checkpoint_selection(
            model_override=model_override,
            extras=extras,
            field_prefix="extras",
            models_api=_models_api,
        )
        _apply_asset_contract_to_extras(
            engine_id=engine_id,
            checkpoint_record=checkpoint_record,
            extras=extras,
            field_prefix="extras",
            require_explicit_checkpoint_contract=True,
            resolve_asset_by_sha=resolve_asset_by_sha,
            resolve_vae_path_by_sha=resolve_vae_path_by_sha,
        )
        _resolve_nested_stage_model_payloads(
            engine_id=engine_id,
            extras=extras,
            hires_cfg=hires_cfg,
            models_api=_models_api,
            resolve_asset_by_sha=resolve_asset_by_sha,
            resolve_vae_path_by_sha=resolve_vae_path_by_sha,
        )

        req = Txt2ImgRequest(
            task=TaskType.TXT2IMG,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps_val,
            guidance_scale=cfg_scale,
            sampler=str(sampler_name),
            scheduler=str(scheduler_name),
            seed=seed_val,
            batch_size=batch_size,
            clip_skip=clip_skip,
            metadata=metadata,
            hires=_build_hires(hires_cfg, width, height, cfg_scale, distilled_cfg_scale) if hires_cfg is not None else None,
            extras=extras,
            smart_offload=smart_offload,
            smart_fallback=smart_fallback,
            smart_cache=smart_cache,
            settings_revision=settings_revision,
        )

        return req, engine_key, model_override

    def _prepare_qwen_image_img2img(
        payload: Dict[str, Any],
        *,
        init_image: Any,
        init_w: int,
        init_h: int,
        settings_revision: int,
    ) -> Tuple[Img2ImgRequest, str, Optional[str]]:
        core = _parse_qwen_image_img2img_core_dto(payload, init_w=init_w, init_h=init_h)
        engine_key = core.engine_key
        model_ref = core.model_ref
        extras = _parse_qwen_image_asset_extras(payload.get("img2img_extras"), field_prefix="img2img_extras")
        extras[_QWEN_IMAGE_VARIANT_KEY] = _QWEN_IMAGE_EDIT_VARIANT

        from apps.backend.inventory.cache import resolve_asset_by_sha, resolve_vae_path_by_sha
        from apps.backend.runtime.models import api as _models_api

        model_ref, checkpoint_record = _resolve_checkpoint_selection(
            model_override=model_ref,
            extras=extras,
            field_prefix="img2img_extras",
            models_api=_models_api,
        )
        _preflight_qwen_image_checkpoint_record(
            checkpoint_record,
            model_field="img2img_extras.model_sha",
        )
        _apply_asset_contract_to_extras(
            engine_id=engine_key,
            checkpoint_record=checkpoint_record,
            extras=extras,
            field_prefix="img2img_extras",
            require_explicit_checkpoint_contract=True,
            resolve_asset_by_sha=resolve_asset_by_sha,
            resolve_vae_path_by_sha=resolve_vae_path_by_sha,
        )

        try:
            condition_width, condition_height = qwen_image_edit_condition_dimensions(init_w, init_h)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        metadata = {
            "styles": core.styles,
            "distilled_cfg_scale": None,
            "image_cfg_scale": None,
            "batch_count": 1,
            "qwen_image_condition_width": condition_width,
            "qwen_image_condition_height": condition_height,
            "qwen_image_vae_width": core.width,
            "qwen_image_vae_height": core.height,
            _QWEN_IMAGE_VARIANT_KEY: _QWEN_IMAGE_EDIT_VARIANT,
        }
        smart_offload, smart_fallback, smart_cache = _resolve_smart_flags()
        req = Img2ImgRequest(
            task=TaskType.IMG2IMG,
            prompt=str(core.prompt),
            negative_prompt=str(core.negative_prompt),
            sampler=str(core.sampler_name),
            scheduler=str(core.scheduler_name),
            seed=core.seed,
            guidance_scale=core.cfg_scale,
            batch_size=1,
            clip_skip=None,
            metadata=metadata,
            init_image=init_image,
            mask=None,
            inpaint_mode=None,
            per_step_blend_strength=1.0,
            per_step_blend_steps=0,
            denoise_strength=1.0,
            width=core.width,
            height=core.height,
            steps=core.steps,
            extras=extras,
            hires=None,
            smart_offload=smart_offload,
            smart_fallback=smart_fallback,
            smart_cache=smart_cache,
            settings_revision=settings_revision,
        )
        return req, engine_key, model_ref

    def _parse_explicit_device(
        payload: Dict[str, Any],
        *,
        route_mode: GenerationRouteMode,
    ) -> str:
        """Parse/validate the per-request device selection (fail loud).

        Note: do not apply `switch_primary_device()` here; apply it only when the task actually starts running
        (single-flight-safe).
        """
        for legacy_key in ("codex_device", "codex_diffusion_device"):
            if legacy_key in payload:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported legacy device key: '{legacy_key}'. Use 'device'.",
                )
        policy = generation_route_device_policy(route_mode)
        try:
            return parse_device_from_payload(payload, route_policy=policy)
        except ValueError as exc:
            _router_log.warning("generation device selection validation failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid 'device' selection"),
            ) from None

    def _require_bool_value(value: object, *, field_name: str) -> bool:
        if not isinstance(value, bool):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be a boolean")
        return value

    def _parse_image_automation_request(payload: Dict[str, Any]) -> ImageAutomationRequest:
        from apps.backend.runtime.text_processing import default_wildcard_root

        _reject_unknown_keys(payload, _IMAGE_AUTOMATION_ALLOWED_KEYS, "image_automation")
        mode = _require_str_field(payload, "mode", allow_empty=False)
        if mode not in {"txt2img", "img2img"}:
            raise HTTPException(status_code=400, detail="'mode' must be one of: txt2img, img2img")

        template_raw = payload.get("template")
        if not isinstance(template_raw, dict):
            raise HTTPException(status_code=400, detail="'template' must be an object")
        template = dict(template_raw)
        if _is_zimage_l2p_engine(template.get("engine")):
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_ZIMAGE_L2P_ENGINE_ID}' does not support image automation.",
            )
        if _is_qwen_image_engine(template.get("engine")):
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{_QWEN_IMAGE_ENGINE_ID}' does not support image automation.",
            )
        _reject_lens_parked_request(template, route_label=mode)
        if mode == "txt2img":
            _validate_txt2img_hires_request_payload(template)
            _validate_pre_task_txt2img_payload(template)
        _enforce_generation_settings_contract(template)
        _validate_route_engine_capability(
            template,
            route_mode=GenerationRouteMode.TXT2IMG if mode == "txt2img" else GenerationRouteMode.IMG2IMG,
        )
        if mode == "txt2img":
            _reject_zimage_l2p_txt2img_payload(template)
        if mode == "img2img":
            _validate_pre_task_img2img_payload(template)

        batch_count_field = "img2img_batch_count" if mode == "img2img" else "batch_count"
        batch_size_field = "img2img_batch_size" if mode == "img2img" else "batch_size"
        batch_count = _require_int_field(template, batch_count_field, minimum=1) if batch_count_field in template else 1
        batch_size = _require_int_field(template, batch_size_field, minimum=1) if batch_size_field in template else 1
        if batch_count != 1 or batch_size != 1:
            raise HTTPException(
                status_code=400,
                detail="image automation requires batch count = 1 and batch size = 1.",
            )

        extras_key = "img2img_extras" if mode == "img2img" else "extras"
        extras_raw = template.get(extras_key)
        if extras_raw is not None and not isinstance(extras_raw, dict):
            raise HTTPException(status_code=400, detail=f"'{extras_key}' must be an object")
        ip_adapter_folder_selects_all = False
        if isinstance(extras_raw, dict):
            normalized_extras = dict(extras_raw)
            ip_adapter_payload = _parse_ip_adapter_payload(
                normalized_extras.get("ip_adapter"),
                field_name=f"{extras_key}.ip_adapter",
                allow_same_as_init=(mode == "img2img"),
                allow_server_folder=True,
            )
            if ip_adapter_payload is not None:
                template_engine_key = _canonical_engine_key(template.get("engine"))
                if template_engine_key:
                    if mode == "txt2img":
                        hires_cfg = normalized_extras.get("hires")
                        _enforce_txt2img_ip_adapter_stage_support(
                            engine_key=template_engine_key,
                            extras=normalized_extras,
                            hires_cfg=hires_cfg if isinstance(hires_cfg, dict) else None,
                        )
                    else:
                        _enforce_ip_adapter_engine_support(
                            engine_key=template_engine_key,
                            field_name=f"{extras_key}.ip_adapter",
                        )
                normalized_extras["ip_adapter"] = ip_adapter_payload
                source = ip_adapter_payload.get("source")
                if (
                    isinstance(source, dict)
                    and source.get("kind") == "server_folder"
                    and source.get("selection_mode") == "all"
                ):
                    ip_adapter_folder_selects_all = True
            template[extras_key] = normalized_extras

        loop_raw = payload.get("loop")
        if loop_raw is None:
            loop = ImageAutomationLoopConfig(mode="count", count=1)
        else:
            if not isinstance(loop_raw, dict):
                raise HTTPException(status_code=400, detail="'loop' must be an object")
            _reject_unknown_keys(loop_raw, _IMAGE_AUTOMATION_LOOP_KEYS, "loop")
            loop_mode = _require_str_field(loop_raw, "mode", allow_empty=False)
            if loop_mode not in {"count", "until_cancelled"}:
                raise HTTPException(status_code=400, detail="'loop.mode' must be one of: count, until_cancelled")
            loop_count = None
            if loop_mode == "count":
                raw_count = loop_raw.get("count")
                if raw_count is not None:
                    loop_count = _require_int_field(loop_raw, "count", minimum=1)
            elif loop_raw.get("count") is not None:
                raise HTTPException(status_code=400, detail="'loop.count' is only valid when loop.mode='count'")
            delay_ms = _require_int_field(loop_raw, "delay_ms", minimum=0) if "delay_ms" in loop_raw else 0
            stop_on_error = _require_bool_value(loop_raw.get("stop_on_error", False), field_name="loop.stop_on_error")
            loop = ImageAutomationLoopConfig(
                mode=loop_mode,
                count=loop_count,
                delay_ms=delay_ms,
                stop_on_error=stop_on_error,
            )

        seed_policy_raw = payload.get("seed_policy")
        if seed_policy_raw is None:
            seed_policy = ImageAutomationSeedPolicy(mode="fixed", increment_step=1)
        else:
            if not isinstance(seed_policy_raw, dict):
                raise HTTPException(status_code=400, detail="'seed_policy' must be an object")
            _reject_unknown_keys(seed_policy_raw, _IMAGE_AUTOMATION_SEED_POLICY_KEYS, "seed_policy")
            seed_mode = _require_str_field(seed_policy_raw, "mode", allow_empty=False)
            if seed_mode not in {"fixed", "increment", "random"}:
                raise HTTPException(status_code=400, detail="'seed_policy.mode' must be one of: fixed, increment, random")
            increment_step = _require_int_field(seed_policy_raw, "increment_step", minimum=1) if "increment_step" in seed_policy_raw else 1
            seed_policy = ImageAutomationSeedPolicy(mode=seed_mode, increment_step=increment_step)

        prompt_source_raw = payload.get("prompt_source")
        if prompt_source_raw is None:
            prompt_source = ImageAutomationPromptSource(kind="current")
        else:
            if not isinstance(prompt_source_raw, dict):
                raise HTTPException(status_code=400, detail="'prompt_source' must be an object")
            _reject_unknown_keys(prompt_source_raw, _IMAGE_AUTOMATION_PROMPT_SOURCE_KEYS, "prompt_source")
            prompt_kind = _require_str_field(prompt_source_raw, "kind", allow_empty=False)
            if prompt_kind not in {"current", "list"}:
                raise HTTPException(status_code=400, detail="'prompt_source.kind' must be one of: current, list")
            insert_position = str(prompt_source_raw.get("insert_position", "replace") or "").strip()
            if insert_position not in {"replace", "prepend", "append"}:
                raise HTTPException(
                    status_code=400,
                    detail="'prompt_source.insert_position' must be one of: replace, prepend, append",
                )
            wildcard_mode = str(prompt_source_raw.get("wildcard_mode", "disabled") or "").strip()
            if wildcard_mode not in {"disabled", "expand"}:
                raise HTTPException(
                    status_code=400,
                    detail="'prompt_source.wildcard_mode' must be one of: disabled, expand",
                )
            wildcard_root = None
            if wildcard_mode == "expand":
                wildcard_root_raw = prompt_source_raw.get("wildcard_root")
                wildcard_root = (
                    _path_from_api(wildcard_root_raw)
                    if isinstance(wildcard_root_raw, str) and wildcard_root_raw.strip()
                    else str(default_wildcard_root())
                )
            prompt_text = prompt_source_raw.get("text")
            if prompt_text is not None and not isinstance(prompt_text, str):
                raise HTTPException(status_code=400, detail="'prompt_source.text' must be a string when provided")
            if prompt_kind == "list":
                prompt_lines = [line.strip() for line in str(prompt_text or "").splitlines() if line.strip()]
                if not prompt_lines:
                    raise HTTPException(status_code=400, detail=_IMAGE_AUTOMATION_EMPTY_LIST_DETAIL)
            prompt_source = ImageAutomationPromptSource(
                kind=prompt_kind,
                text=prompt_text,
                insert_position=insert_position,
                wildcard_root=wildcard_root,
                wildcard_mode=wildcard_mode,
            )

        init_source: ImageAutomationInitSource | None = None
        init_source_raw = payload.get("init_source")
        if mode == "txt2img":
            if init_source_raw is not None:
                raise HTTPException(status_code=400, detail="'init_source' is only supported for img2img automation")
        else:
            if init_source_raw is None:
                init_source = ImageAutomationInitSource(kind="uploaded_current")
            else:
                if not isinstance(init_source_raw, dict):
                    raise HTTPException(status_code=400, detail="'init_source' must be an object")
                _reject_unknown_keys(init_source_raw, _IMAGE_AUTOMATION_INIT_SOURCE_KEYS, "init_source")
                init_kind = _require_str_field(init_source_raw, "kind", allow_empty=False)
                if init_kind not in {"uploaded_current", "server_folder"}:
                    raise HTTPException(
                        status_code=400,
                        detail="'init_source.kind' must be one of: uploaded_current, server_folder",
                    )
                if init_kind == "uploaded_current":
                    for folder_only_key in ("folder_path", "selection_mode", "count", "order", "sort_by", "use_crop"):
                        if folder_only_key in init_source_raw:
                            raise HTTPException(
                                status_code=400,
                                detail=f"'init_source.{folder_only_key}' is only valid when init_source.kind='server_folder'",
                            )
                    init_source = ImageAutomationInitSource(kind="uploaded_current")
                else:
                    selection_mode = init_source_raw.get("selection_mode")
                    if selection_mode is None:
                        selection_mode = "all"
                    elif not isinstance(selection_mode, str):
                        raise HTTPException(
                            status_code=400,
                            detail="'init_source.selection_mode' must be one of: all, count",
                        )
                    else:
                        selection_mode = selection_mode.strip()
                    if selection_mode not in {"all", "count"}:
                        raise HTTPException(
                            status_code=400,
                            detail="'init_source.selection_mode' must be one of: all, count",
                        )
                    order = str(init_source_raw.get("order", "sorted") or "").strip()
                    if order not in {"random", "sorted"}:
                        raise HTTPException(status_code=400, detail="'init_source.order' must be one of: random, sorted")
                    sort_by = init_source_raw.get("sort_by")
                    if sort_by is None:
                        sort_by = "name"
                    elif not isinstance(sort_by, str):
                        raise HTTPException(
                            status_code=400,
                            detail="'init_source.sort_by' must be one of: name, size, created_at, modified_at",
                        )
                    else:
                        sort_by = sort_by.strip()
                    if sort_by not in {"name", "size", "created_at", "modified_at"}:
                        raise HTTPException(
                            status_code=400,
                            detail="'init_source.sort_by' must be one of: name, size, created_at, modified_at",
                        )
                    selection_count = None
                    if selection_mode == "count":
                        raw_count = init_source_raw.get("count")
                        if raw_count is None:
                            raise HTTPException(
                                status_code=400,
                                detail="'init_source.count' is required when init_source.selection_mode='count'",
                            )
                        selection_count = _require_int_field(init_source_raw, "count", minimum=1)
                    elif init_source_raw.get("count") is not None:
                        raise HTTPException(
                            status_code=400,
                            detail="'init_source.count' is only valid when init_source.selection_mode='count'",
                        )
                    use_crop = _require_bool_value(init_source_raw.get("use_crop", False), field_name="init_source.use_crop")
                    folder_raw = init_source_raw.get("folder_path")
                    if not isinstance(folder_raw, str) or not folder_raw.strip():
                        raise HTTPException(
                            status_code=400,
                            detail="'init_source.folder_path' is required when init_source.kind='server_folder'",
                        )
                    folder_path = _path_from_api(folder_raw)
                    mask_value = template.get("img2img_mask")
                    if isinstance(mask_value, str) and mask_value.strip():
                        raise HTTPException(
                            status_code=400,
                            detail="img2img folder automation does not support masks. Set the initial-image source back to IMG or clear the mask.",
                        )
                    init_source = ImageAutomationInitSource(
                        kind="server_folder",
                        folder_path=_path_from_api(folder_raw),
                        selection_mode=str(selection_mode),
                        count=selection_count,
                        order=order,
                        sort_by=str(sort_by),
                        use_crop=use_crop,
                    )
            if init_source.kind == "uploaded_current":
                init_image_data = template.get("img2img_init_image")
                if not isinstance(init_image_data, str) or not init_image_data.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="img2img automation with init_source.kind='uploaded_current' requires 'template.img2img_init_image'.",
                    )

        if loop.mode == "count" and loop.count is None:
            init_folder_selects_all = (
                mode == "img2img"
                and init_source is not None
                and init_source.kind == "server_folder"
                and init_source.selection_mode == "all"
            )
            if not (init_folder_selects_all or ip_adapter_folder_selects_all):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "'loop.count' is required for loop.mode='count' unless a folder-backed init source "
                        "or IP-Adapter source is selecting all images."
                    ),
                )

        return ImageAutomationRequest(
            mode=mode,
            template=template,
            loop=loop,
            seed_policy=seed_policy,
            prompt_source=prompt_source,
            init_source=init_source,
        )

    _ORCH = InferenceOrchestrator()


    def run_txt2img_task(task_id: str, payload: Dict[str, Any], entry: TaskEntry, *, device: str) -> None:
        from apps.backend.interfaces.api.tasks.generation_tasks import run_image_task as _run_image_task

        try:
            _run_image_task(
                task_id=task_id,
                payload=payload,
                entry=entry,
                device=device,
                task_type=TaskType.TXT2IMG,
                prepare=prepare_txt2img,
                orch=_ORCH,
                ensure_default_engines_registered=_ensure_default_engines_registered,
                live_preview=live_preview,
                opts_get=_opts_get,
                opts_snapshot=_opts_snapshot,
                generation_provenance=_GENERATION_PROVENANCE,
                save_generated_images=_save_generated_images,
            )
        except HTTPException:
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid txt2img payload configuration"),
            ) from None

    def prepare_img2img(payload: Dict[str, Any]) -> Tuple[Img2ImgRequest, str, Optional[str]]:
        _reject_removed_img2img_second_pass_keys(payload)
        _reject_public_qwen_image_variant(payload, context="img2img")
        _reject_unknown_keys(payload, _IMG2IMG_ALLOWED_KEYS, "img2img")
        settings_revision = _require_int_field(payload, "settings_revision", minimum=0)
        if "img2img_init_image" not in payload:
            raise HTTPException(status_code=400, detail="Missing 'img2img_init_image'")
        init_image_data = payload.get("img2img_init_image")
        try:
            if not isinstance(init_image_data, str) or not init_image_data.strip():
                raise ValueError("'img2img_init_image' must be a non-empty string")
            init_image = media.decode_image(init_image_data)
        except Exception as exc:
            _router_log.warning("img2img init image validation failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid 'img2img_init_image' payload"),
            ) from None
        init_w, init_h = 0, 0
        try:
            init_w, init_h = init_image.size  # type: ignore[attr-defined]
        except Exception:
            init_w, init_h = 0, 0
        engine_key_for_preflight = _canonical_engine_key(payload.get("engine"))
        if _is_qwen_image_engine(engine_key_for_preflight):
            return _prepare_qwen_image_img2img(
                payload,
                init_image=init_image,
                init_w=init_w,
                init_h=init_h,
                settings_revision=settings_revision,
            )
        mask_data = payload.get('img2img_mask')
        mask_image = None
        if mask_data:
            try:
                if not isinstance(mask_data, str) or not mask_data.strip():
                    raise ValueError("'img2img_mask' must be a non-empty string")
                mask_image = media.decode_image(mask_data)
            except Exception as exc:
                _router_log.warning("img2img mask validation failed: %s", exc)
                raise HTTPException(
                    status_code=400,
                    detail=public_http_error_detail(exc, fallback="Invalid 'img2img_mask' payload"),
                ) from None

        inpaint_mode = None
        inpainting_fill = 1
        inpaint_full_res_padding = 32
        inpainting_mask_invert = 0
        mask_blur = 4
        mask_blur_x = 4
        mask_blur_y = 4
        mask_round = True
        mask_region_split = False
        per_step_blend_strength = 1.0
        per_step_blend_steps = 0

        if mask_image is not None:
            if "img2img_mask_enforcement" in payload:
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_mask_enforcement' was removed; use 'img2img_inpaint_mode'.",
                )
            raw_inpaint_mode = payload.get("img2img_inpaint_mode")
            if not isinstance(raw_inpaint_mode, str) or not raw_inpaint_mode.strip():
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_inpaint_mode' is required when 'img2img_mask' is provided",
                )
            inpaint_mode = raw_inpaint_mode.strip()
            if inpaint_mode not in ("post_sample_blend", "per_step_blend", "fooocus_inpaint", "brushnet"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid 'img2img_inpaint_mode' (allowed: per_step_blend, post_sample_blend, fooocus_inpaint, brushnet)",
                )
            if "img2img_inpainting_fill" in payload:
                inpainting_fill = _require_int_field(payload, "img2img_inpainting_fill")
            if inpainting_fill not in (0, 1, 2, 3):
                raise HTTPException(status_code=400, detail="'img2img_inpainting_fill' must be 0,1,2,3")

            if "img2img_inpaint_full_res_padding" in payload:
                inpaint_full_res_padding = _require_int_field(payload, "img2img_inpaint_full_res_padding")
            if inpaint_full_res_padding < 0:
                raise HTTPException(status_code=400, detail="'img2img_inpaint_full_res_padding' must be >= 0")

            if "img2img_inpainting_mask_invert" in payload:
                inpainting_mask_invert = _require_int_field(payload, "img2img_inpainting_mask_invert")
            if inpainting_mask_invert not in (0, 1):
                raise HTTPException(status_code=400, detail="'img2img_inpainting_mask_invert' must be 0 or 1")

            if "img2img_mask_blur" in payload:
                mask_blur = _require_int_field(payload, "img2img_mask_blur")
                mask_blur_x = mask_blur
                mask_blur_y = mask_blur
            if "img2img_mask_blur_x" in payload:
                mask_blur_x = _require_int_field(payload, "img2img_mask_blur_x")
            if "img2img_mask_blur_y" in payload:
                mask_blur_y = _require_int_field(payload, "img2img_mask_blur_y")
            if mask_blur_x < 0 or mask_blur_y < 0:
                raise HTTPException(status_code=400, detail="'img2img_mask_blur' must be >= 0")

            if "img2img_per_step_blend_strength" in payload:
                if inpaint_mode != "per_step_blend":
                    raise HTTPException(
                        status_code=400,
                        detail="'img2img_per_step_blend_strength' requires 'img2img_inpaint_mode' = 'per_step_blend'",
                    )
                per_step_blend_strength = _require_float_field(
                    payload,
                    "img2img_per_step_blend_strength",
                    minimum=0.0,
                    maximum=1.0,
                )
            if "img2img_per_step_blend_steps" in payload:
                if inpaint_mode != "per_step_blend":
                    raise HTTPException(
                        status_code=400,
                        detail="'img2img_per_step_blend_steps' requires 'img2img_inpaint_mode' = 'per_step_blend'",
                    )
                per_step_blend_steps = _require_int_field(
                    payload,
                    "img2img_per_step_blend_steps",
                    minimum=0,
                )

            if "img2img_mask_round" in payload:
                mask_round = _require_bool_field(payload, "img2img_mask_round")
            if "img2img_mask_region_split" in payload:
                mask_region_split = _require_bool_field(payload, "img2img_mask_region_split")
        else:
            if "img2img_inpaint_mode" in payload:
                raise HTTPException(status_code=400, detail="'img2img_inpaint_mode' requires 'img2img_mask'")
            if "img2img_mask_enforcement" in payload:
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_mask_enforcement' was removed; use 'img2img_inpaint_mode'.",
                )
            if "img2img_per_step_blend_strength" in payload:
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_per_step_blend_strength' requires 'img2img_mask'",
                )
            if "img2img_per_step_blend_steps" in payload:
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_per_step_blend_steps' requires 'img2img_mask'",
                )
            if "img2img_mask_region_split" in payload:
                raise HTTPException(status_code=400, detail="'img2img_mask_region_split' requires 'img2img_mask'")

        core = _parse_img2img_core_dto(payload, init_w=init_w, init_h=init_h)
        engine_key = core.engine_key
        model_ref = core.model_ref
        prompt = core.prompt
        negative_prompt = core.negative_prompt
        styles = core.styles
        batch_count = core.batch_count
        batch_size = core.batch_size
        steps_val = core.steps
        cfg_scale = core.cfg_scale
        distilled_cfg_scale = core.distilled_cfg_scale
        image_cfg_scale = core.image_cfg_scale
        denoise = core.denoise
        width_val = core.width
        height_val = core.height
        sampler_name = core.sampler_name
        scheduler_name = core.scheduler_name
        family_name, family_capability = _resolve_image_family_capability_contract(engine_key)
        _enforce_family_sampler_scheduler_support(
            engine_key=engine_key,
            family_name=family_name,
            family_capability=family_capability,
            sampler_name=str(sampler_name),
            scheduler_name=str(scheduler_name),
            sampler_field_name="img2img_sampling",
            scheduler_field_name="img2img_scheduler",
        )
        if mask_image is not None and inpaint_mode is not None:
            _enforce_img2img_inpaint_mode_support(engine_key=engine_key, mode=inpaint_mode)
            if inpaint_mode in {"fooocus_inpaint", "brushnet"}:
                extras_raw = payload.get("img2img_extras")
                if isinstance(extras_raw, Mapping):
                    supir_cfg = extras_raw.get("supir")
                    if isinstance(supir_cfg, Mapping) and bool(supir_cfg.get("enabled")):
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"'img2img_inpaint_mode' = '{inpaint_mode}' cannot be combined with "
                                "'img2img_extras.supir'."
                            ),
                        )
                try:
                    if inpaint_mode == "fooocus_inpaint":
                        from apps.backend.runtime.families.sd.fooocus_inpaint import resolve_fooocus_inpaint_assets

                        resolve_fooocus_inpaint_assets()
                    else:
                        from apps.backend.runtime.families.sd.brushnet import resolve_brushnet_assets

                        resolve_brushnet_assets()
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from None
        seed_val = core.seed
        clip_skip = core.clip_skip
        noise_source = core.noise_source
        ensd_raw = core.ensd_raw
        hires_cfg: Optional[Dict[str, Any]] = None
        hires_data = {"enable": False}

        resize_mode = _parse_img2img_resize_mode(payload)
        if resize_mode is not None and engine_key != "zimage":
            raise HTTPException(
                status_code=400,
                detail=f"Engine '{engine_key}' does not support top-level 'img2img_resize_mode'.",
            )
        if resize_mode is not None and mask_image is not None and engine_key == "zimage":
            raise HTTPException(
                status_code=400,
                detail="Engine 'zimage' does not support 'img2img_resize_mode' when 'img2img_mask' is provided.",
            )
        extras: Dict[str, Any] = {}
        if resize_mode is not None:
            extras["resize_mode"] = resize_mode
        raw_extras = payload.get("img2img_extras")
        supir_config = None
        if raw_extras is not None:
            if not isinstance(raw_extras, dict):
                raise HTTPException(status_code=400, detail="'img2img_extras' must be an object")
            _reject_unknown_keys(raw_extras, _IMG2IMG_EXTRAS_KEYS, "img2img_extras")
            raw_extras = dict(raw_extras)
            raw_supir_payload = raw_extras.get("supir")
            hires_cfg = _parse_img2img_nested_hires_config(
                raw_extras.get("hires"),
                default_cfg=cfg_scale,
                default_distilled=distilled_cfg_scale or 3.5,
            )
            raw_extras.pop("hires", None)
            ip_adapter_payload = _parse_ip_adapter_payload(
                raw_extras.get("ip_adapter"),
                field_name="img2img_extras.ip_adapter",
                allow_same_as_init=True,
                allow_server_folder=False,
            )
            checkpoint_core_only = _parse_optional_bool_selector(
                payload=raw_extras,
                key="checkpoint_core_only",
                field_name="img2img_extras.checkpoint_core_only",
            )
            model_format = _parse_optional_model_format_selector(
                payload=raw_extras,
                key="model_format",
                field_name="img2img_extras.model_format",
            )
            vae_source = _parse_optional_vae_source_selector(
                payload=raw_extras,
                key="vae_source",
                field_name="img2img_extras.vae_source",
            )
            raw_extras.pop("checkpoint_core_only", None)
            raw_extras.pop("model_format", None)
            raw_extras.pop("vae_source", None)
            raw_extras.pop("ip_adapter", None)
            try:
                supir_config = parse_supir_mode_config(raw_extras.get("supir"))
            except SupirConfigError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=public_http_error_detail(exc, fallback="Invalid 'img2img_extras.supir' configuration"),
                ) from None
            raw_extras.pop("supir", None)

            te_override = raw_extras.get("text_encoder_override")
            if te_override is not None:
                if not isinstance(te_override, dict):
                    raise HTTPException(status_code=400, detail="'img2img_extras.text_encoder_override' must be an object")
                _reject_unknown_keys(te_override, {"family", "label", "components"}, "img2img_extras.text_encoder_override")
                family_raw = te_override.get("family")
                label_raw = te_override.get("label")
                if not isinstance(family_raw, str) or not family_raw.strip():
                    raise HTTPException(status_code=400, detail="'img2img_extras.text_encoder_override.family' must be a non-empty string")
                if not isinstance(label_raw, str) or not label_raw.strip():
                    raise HTTPException(status_code=400, detail="'img2img_extras.text_encoder_override.label' must be a non-empty string")
                family = family_raw.strip()
                label = label_raw.strip()
                if "/" in label and not label.startswith(f"{family}/"):
                    raise HTTPException(
                        status_code=400,
                        detail="img2img_extras.text_encoder_override.label must start with '<family>/'",
                    )
                components_val = te_override.get("components")
                components: list[str] | None = None
                if components_val is not None:
                    if not isinstance(components_val, list) or any(not isinstance(c, str) for c in components_val):
                        raise HTTPException(status_code=400, detail="'img2img_extras.text_encoder_override.components' must be an array of strings")
                    components = [c.strip() for c in components_val if isinstance(c, str) and c.strip()]
                te_cfg: Dict[str, Any] = {"family": family, "label": label}
                if components:
                    te_cfg["components"] = components
                raw_extras["text_encoder_override"] = te_cfg

            if "er_sde" in raw_extras:
                raw_extras["er_sde"] = _parse_er_sde_options(
                    raw_extras["er_sde"],
                    field_name="img2img_extras.er_sde",
                )
            if "guidance" in raw_extras:
                raw_extras["guidance"] = _parse_guidance_options(
                    raw_extras["guidance"],
                    field_name="img2img_extras.guidance",
                )
            if ip_adapter_payload is not None:
                raw_extras["ip_adapter"] = ip_adapter_payload

            extras.update(raw_extras)
            if supir_config is not None and isinstance(raw_supir_payload, Mapping):
                extras["supir"] = dict(raw_supir_payload)
            if checkpoint_core_only is not None:
                extras["checkpoint_core_only"] = checkpoint_core_only
            if model_format is not None:
                extras["model_format"] = model_format
            if vae_source is not None:
                extras["vae_source"] = vae_source
        if hires_cfg is not None:
            hires_sampler = _parse_optional_sampler_field(
                value=hires_cfg.get("sampler"),
                field_name="img2img_extras.hires.sampler",
            )
            if hires_sampler is not None:
                hires_cfg["sampler"] = hires_sampler
                _validate_er_sde_release_scope(
                    engine_key=engine_key,
                    sampler=hires_sampler,
                    field_name="img2img_extras.hires.sampler",
                )
            hires_scheduler = _parse_optional_scheduler_field(
                value=hires_cfg.get("scheduler"),
                field_name="img2img_extras.hires.scheduler",
            )
            if hires_sampler is not None or hires_scheduler is not None:
                resolved_hires_sampler, resolved_hires_scheduler = _resolve_hires_sampler_scheduler_override(
                    base_sampler=str(sampler_name),
                    base_scheduler=str(scheduler_name),
                    sampler_override=hires_sampler,
                    scheduler_override=hires_scheduler,
                    sampler_field_name="img2img_extras.hires.sampler",
                    scheduler_field_name="img2img_extras.hires.scheduler",
                )
                if hires_sampler is not None:
                    hires_cfg["sampler"] = resolved_hires_sampler
                hires_cfg["scheduler"] = resolved_hires_scheduler
                _enforce_family_sampler_scheduler_support(
                    engine_key=engine_key,
                    family_name=family_name,
                    family_capability=family_capability,
                    sampler_name=str(resolved_hires_sampler),
                    scheduler_name=str(resolved_hires_scheduler),
                    sampler_field_name="img2img_extras.hires.sampler",
                    scheduler_field_name="img2img_extras.hires.scheduler",
                )
            hires_data = _build_hires(
                hires_cfg,
                width_val,
                height_val,
                cfg_scale,
                distilled_cfg_scale or 3.5,
            )
        if bool(hires_data.get("enable")) and mask_image is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "img2img hires does not support masks/inpaint in this backend seam yet. "
                    "Disable hires or remove 'img2img_mask'."
                ),
            )
        if isinstance(extras.get("ip_adapter"), dict):
            _enforce_ip_adapter_engine_support(
                engine_key=engine_key,
                field_name="img2img_extras.ip_adapter",
            )
        if supir_config is not None:
            _enforce_supir_engine_support(
                engine_key=engine_key,
                field_name="img2img_extras.supir",
            )
            if isinstance(extras.get("guidance"), dict):
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_extras.supir' cannot be combined with 'img2img_extras.guidance'.",
                )
            if bool(hires_data.get("enable")):
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_extras.supir' cannot be combined with img2img hires in tranche 1.",
                )
            if mask_image is not None and inpainting_fill in {2, 3}:
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_extras.supir' with masks supports only 'img2img_inpainting_fill' values 0 or 1.",
                )
            if isinstance(extras.get("swap_model"), dict):
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_extras.supir' cannot be combined with 'img2img_extras.swap_model'.",
                )
            if isinstance(extras.get("ip_adapter"), dict):
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_extras.supir' cannot be combined with 'img2img_extras.ip_adapter'.",
                )
        # Z-Image variant selection (Turbo/Base) for img2img runs.
        if "zimage_variant" in extras:
            val = extras.get("zimage_variant")
            if val is None:
                extras.pop("zimage_variant", None)
            elif not isinstance(val, str):
                raise HTTPException(status_code=400, detail="'img2img_extras.zimage_variant' must be a string")
            else:
                variant = val.strip().lower()
                if not variant:
                    extras.pop("zimage_variant", None)
                elif variant not in {"turbo", "base"}:
                    raise HTTPException(
                        status_code=400,
                        detail="'img2img_extras.zimage_variant' must be one of: turbo, base",
                    )
                else:
                    extras["zimage_variant"] = variant
        if noise_source:
            extras['noise_source'] = str(noise_source)
        if ensd_raw is not None:
            try:
                extras['eta_noise_seed_delta'] = int(float(ensd_raw))
            except Exception:
                raise HTTPException(status_code=400, detail="img2img_eta_noise_seed_delta must be numeric")

        # Resolve SHA-based assets (if provided in img2img_extras)
        from apps.backend.inventory.cache import resolve_asset_by_sha, resolve_vae_path_by_sha
        from apps.backend.runtime.models import api as _models_api
        engine_id = engine_key

        if "vae_path" in extras or "tenc_path" in extras:
            raise HTTPException(status_code=400, detail="img2img_extras must not include raw '*_path' fields; use sha256 via '*_sha'")

        model_ref, checkpoint_record = _resolve_checkpoint_selection(
            model_override=model_ref,
            extras=extras,
            field_prefix="img2img_extras",
            models_api=_models_api,
        )
        if inpaint_mode == "fooocus_inpaint":
            try:
                from apps.backend.runtime.families.sd.fooocus_inpaint import ensure_fooocus_checkpoint_supported

                ensure_fooocus_checkpoint_supported(checkpoint_record)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None

        _apply_asset_contract_to_extras(
            engine_id=engine_id,
            checkpoint_record=checkpoint_record,
            extras=extras,
            field_prefix="img2img_extras",
            require_explicit_checkpoint_contract=True,
            resolve_asset_by_sha=resolve_asset_by_sha,
            resolve_vae_path_by_sha=resolve_vae_path_by_sha,
        )
        if supir_config is not None:
            try:
                resolve_supir_assets(
                    checkpoint_record=checkpoint_record,
                    variant=supir_config.variant,
                    supir_models_roots=[Path(path) for path in get_paths_for("supir_models")],
                )
            except (SupirBaseModelError, SupirWeightsError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=public_http_error_detail(exc, fallback="SUPIR mode asset validation failed"),
                ) from None
            _reject_supir_prompt_loras(prompt=prompt, negative_prompt=negative_prompt)
            if extras.get("lora_path") is not None:
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_extras.supir' cannot be combined with LoRA selections in tranche 1.",
                )

        metadata = {
            "styles": styles,
            "distilled_cfg_scale": distilled_cfg_scale,
            "image_cfg_scale": image_cfg_scale,
            "batch_count": batch_count,
        }
        if noise_source:
            metadata["noise_source"] = str(noise_source)
        if 'eta_noise_seed_delta' in extras:
            metadata["eta_noise_seed_delta"] = extras['eta_noise_seed_delta']

        smart_offload, smart_fallback, smart_cache = _resolve_smart_flags()
        req = Img2ImgRequest(
            task=TaskType.IMG2IMG,
            prompt=prompt,
            negative_prompt=negative_prompt,
            sampler=str(sampler_name),
            scheduler=str(scheduler_name),
            seed=seed_val,
            guidance_scale=cfg_scale,
            batch_size=batch_size,
            clip_skip=clip_skip,
            metadata=metadata,
            init_image=init_image,
            mask=mask_image,
            inpaint_mode=inpaint_mode,
            per_step_blend_strength=per_step_blend_strength,
            per_step_blend_steps=per_step_blend_steps,
            mask_region_split=mask_region_split,
            inpainting_fill=inpainting_fill,
            inpaint_full_res_padding=inpaint_full_res_padding,
            inpainting_mask_invert=inpainting_mask_invert,
            mask_blur=mask_blur,
            mask_blur_x=mask_blur_x,
            mask_blur_y=mask_blur_y,
            mask_round=mask_round,
            denoise_strength=denoise,
            width=width_val,
            height=height_val,
            steps=steps_val,
            extras=extras,
            hires=hires_data if hires_data.get("enable") else None,
            smart_offload=smart_offload,
            smart_fallback=smart_fallback,
            smart_cache=smart_cache,
            settings_revision=settings_revision,
        )

        return req, engine_key, model_ref

    def run_img2img_task(task_id: str, payload: Dict[str, Any], entry: TaskEntry, *, device: str) -> None:
        from apps.backend.interfaces.api.tasks.generation_tasks import run_image_task as _run_image_task

        try:
            _run_image_task(
                task_id=task_id,
                payload=payload,
                entry=entry,
                device=device,
                task_type=TaskType.IMG2IMG,
                prepare=prepare_img2img,
                orch=_ORCH,
                ensure_default_engines_registered=_ensure_default_engines_registered,
                live_preview=live_preview,
                opts_get=_opts_get,
                opts_snapshot=_opts_snapshot,
                generation_provenance=_GENERATION_PROVENANCE,
                save_generated_images=_save_generated_images,
            )
        except HTTPException:
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid img2img payload configuration"),
            ) from None

    def _wan_require_dims_multiple_of_16(*, task: str, width: int, height: int) -> None:
        """WAN video geometry guard (Diffusers parity).

        WAN requires width/height divisible by 16; otherwise the latent patch grid silently crops.
        The frontend rounds up, but the backend must fail loud for direct API callers.
        """

        if height % 16 == 0 and width % 16 == 0:
            return
        w_up = ((int(width) + 15) // 16) * 16
        h_up = ((int(height) + 15) // 16) * 16
        raise HTTPException(
            status_code=400,
            detail=(
                f"WAN22 {task}: width/height must be divisible by 16 (Diffusers parity). "
                f"Got {int(width)}x{int(height)}. Suggested: {w_up}x{h_up} (rounded up)."
            ),
        )

    def prepare_txt2vid(payload: Dict[str, Any]) -> Tuple[Txt2VidRequest, str, Optional[str]]:
        settings_revision = _require_int_field(payload, "settings_revision", minimum=0)
        video_engine_key = _canonical_engine_key(payload.get("engine")) if payload.get("engine") is not None else ""
        use_generic_video_route = not _is_legacy_or_wan_video_route_engine(video_engine_key)
        wan_metadata_dir: str | None = None
        expected_unipc_solver_hint: str | None = None
        extras: Dict[str, Any] = {}
        model_ref: str | None = None
        if use_generic_video_route and video_engine_key == "ltx2":
            model_ref, checkpoint_record = _resolve_generic_video_checkpoint_contract(
                payload=payload,
                extras=extras,
                engine_key=video_engine_key,
            )
            ltx_defaults = _require_ltx2_checkpoint_execution_defaults(checkpoint_record=checkpoint_record)
            extras["ltx_checkpoint_kind"] = ltx_defaults.checkpoint_kind
            ltx_execution_profile, ltx_two_stage_assets = _resolve_ltx2_requested_execution_profile(
                payload=payload,
                checkpoint_record=checkpoint_record,
                defaults=ltx_defaults,
            )
            extras["ltx_execution_profile"] = ltx_execution_profile
            if ltx_two_stage_assets is not None:
                extras["ltx_two_stage_distilled_lora_path"] = str(ltx_two_stage_assets.distilled_lora_path)
                extras["ltx_two_stage_spatial_upsampler_path"] = str(ltx_two_stage_assets.spatial_upsampler_path)
            parsed = _parse_ltx2_generic_video_core_dto(
                payload,
                task_prefix="txt2vid",
                execution_profile=ltx_execution_profile,
                default_steps=int(ltx_defaults.default_steps),
                default_cfg_scale=float(ltx_defaults.default_guidance_scale),
            )
        elif use_generic_video_route:
            parsed = _parse_generic_txt2vid_core_dto(payload, engine_key=video_engine_key)
        else:
            wan_metadata_dir = _resolve_wan_metadata_dir(payload)
            default_sampler, default_scheduler = _resolve_wan_sampler_scheduler_defaults_from_assets(wan_metadata_dir)
            expected_unipc_solver_hint = _extract_wan22_unipc_solver_hint(default_sampler)
            parsed = _parse_txt2vid_core_dto(
                payload,
                default_sampler=default_sampler,
                default_scheduler=default_scheduler,
                expected_unipc_solver_hint=expected_unipc_solver_hint,
            )
        prompt = parsed.prompt
        negative_prompt = parsed.negative_prompt
        width_val = parsed.width
        height_val = parsed.height
        steps_val = parsed.steps
        fps_val = parsed.fps
        frames_val = parsed.num_frames
        sampler_name = parsed.sampler_name
        scheduler_name = parsed.scheduler_name
        seed_val = parsed.seed
        cfg_val = parsed.guidance_scale

        video_options, video_extras_updates = _parse_video_output_options(payload, route_label="txt2vid")
        extras.update(video_extras_updates)
        if use_generic_video_route:
            if model_ref is None:
                model_ref, _checkpoint_record = _resolve_generic_video_checkpoint_contract(
                    payload=payload,
                    extras=extras,
                    engine_key=video_engine_key,
                )
            smart_offload, smart_fallback, smart_cache = _resolve_smart_flags()
            req = Txt2VidRequest(
                task=TaskType.TXT2VID,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width_val,
                height=height_val,
                steps=steps_val,
                fps=fps_val,
                num_frames=frames_val,
                sampler=sampler_name,
                scheduler=scheduler_name,
                seed=seed_val,
                guidance_scale=cfg_val,
                video_options=video_options,
                extras=extras,
                smart_offload=smart_offload,
                smart_fallback=smart_fallback,
                smart_cache=smart_cache,
                settings_revision=settings_revision,
                metadata={
                    "styles": payload.get('txt2vid_styles', []),
                },
            )
            return req, video_engine_key, model_ref
        # WAN (GGUF-only): strict sha-only selection for model parts (no raw paths).
        from apps.backend.inventory.cache import resolve_asset_by_sha, resolve_vae_path_by_sha

        def _require_sha_field(key: str) -> str:
            return _require_sha256_field(payload, key)

        has_wan_single = isinstance(payload.get("wan_single"), dict)
        has_wan_high = isinstance(payload.get("wan_high"), dict)
        has_wan_low = isinstance(payload.get("wan_low"), dict)
        if has_wan_single:
            if has_wan_high or has_wan_low:
                raise HTTPException(
                    status_code=400,
                    detail="WAN22 requests must use either 'wan_single' or ('wan_high' + 'wan_low'), not both.",
                )
        elif has_wan_high or has_wan_low:
            if not (has_wan_high and has_wan_low):
                missing_stage = "wan_high" if not has_wan_high else "wan_low"
                raise HTTPException(
                    status_code=400,
                    detail=f"WAN22 14B requests must provide both 'wan_high' and 'wan_low' (missing '{missing_stage}').",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="WAN22 requests must include either 'wan_single' or both 'wan_high' and 'wan_low'.",
            )

        def _resolve_wan_stage_model_path(*, stage_key: str, raw: Mapping[str, Any]) -> str:
            if isinstance(raw.get("model_dir"), str) and str(raw.get("model_dir")).strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"'{stage_key}.model_dir' is unsupported; use '{stage_key}.model_sha'",
                )
            sha = _require_sha256_field(raw, "model_sha")
            model_path = resolve_asset_by_sha(sha)
            if not model_path:
                raise HTTPException(status_code=409, detail=f"WAN stage model not found for sha: {sha}")
            if not str(model_path).lower().endswith(".gguf"):
                raise HTTPException(status_code=409, detail=f"WAN stage sha does not resolve to a .gguf file: {sha}")
            return str(model_path)

        if has_wan_single:
            prompt, negative_prompt = _parse_wan_request_prompt_loras(
                prompt_field_name="txt2vid_prompt",
                negative_prompt_field_name="txt2vid_neg_prompt",
                lora_owner_field_name="wan_single",
                prompt=str(prompt or "").strip(),
                negative_prompt=str(negative_prompt or "").strip(),
            )
            if not prompt:
                raise HTTPException(status_code=400, detail="'txt2vid_prompt' resolved empty after WAN LoRA parsing")

            raw_single = payload.get("wan_single")
            assert isinstance(raw_single, dict)
            _reject_legacy_wan_stage_lora_keys(stage_key="wan_single", stage_raw=raw_single)
            _reject_unknown_keys(raw_single, _WAN_SINGLE_ALLOWED_KEYS, "wan_single")
            explicit_single_loras = _normalize_wan_stage_loras(
                stage_raw=raw_single,
                stage_key="wan_single",
                resolve_asset_by_sha_fn=resolve_asset_by_sha,
            )
            extras["wan_single"] = {
                "model_dir": _resolve_wan_stage_model_path(stage_key="wan_single", raw=raw_single),
                "loras": _merge_wan_stage_loras(explicit_single_loras),
                **(
                    {"flow_shift": raw_single.get("flow_shift")}
                    if raw_single.get("flow_shift") is not None
                    else {}
                ),
            }
        else:
            prompt, negative_prompt = _parse_wan_request_prompt_loras(
                prompt_field_name="txt2vid_prompt",
                negative_prompt_field_name="txt2vid_neg_prompt",
                lora_owner_field_name="wan_high",
                prompt=str(prompt or "").strip(),
                negative_prompt=str(negative_prompt or "").strip(),
            )
            if not prompt:
                raise HTTPException(status_code=400, detail="'txt2vid_prompt' resolved empty after WAN LoRA parsing")

            def _resolve_wan_stage(stage_key: str) -> dict[str, object]:
                raw = payload.get(stage_key)
                if not isinstance(raw, dict):
                    raise HTTPException(status_code=400, detail=f"'{stage_key}' is required and must be an object")
                _reject_legacy_wan_stage_lora_keys(stage_key=stage_key, stage_raw=raw)
                allowed_stage_keys = _WAN_HIGH_ALLOWED_KEYS if stage_key == "wan_high" else _WAN_LOW_ALLOWED_KEYS
                _reject_unknown_keys(raw, allowed_stage_keys, stage_key)
                out: dict[str, object] = dict(raw)
                out.pop("model_sha", None)
                out["model_dir"] = _resolve_wan_stage_model_path(stage_key=stage_key, raw=raw)
                if stage_key == "wan_high":
                    stage_prompt = str(prompt or "").strip()
                    stage_negative_prompt = str(negative_prompt or "").strip()
                else:
                    raw_stage_prompt = out.get("prompt")
                    if not isinstance(raw_stage_prompt, str):
                        raise HTTPException(status_code=400, detail=f"'{stage_key}.prompt' is required and must be a string")
                    stage_prompt = str(raw_stage_prompt).strip()
                    if not stage_prompt:
                        raise HTTPException(status_code=400, detail=f"'{stage_key}.prompt' must be a non-empty string")
                    raw_stage_negative_prompt = out.get("negative_prompt")
                    if raw_stage_negative_prompt is not None and not isinstance(raw_stage_negative_prompt, str):
                        raise HTTPException(
                            status_code=400,
                            detail=f"'{stage_key}.negative_prompt' must be a string when provided",
                        )
                    stage_negative_prompt = (
                        str(raw_stage_negative_prompt).strip()
                        if isinstance(raw_stage_negative_prompt, str)
                        else None
                    )
                stage_prompt, stage_negative_prompt, prompt_stage_loras = _parse_wan_stage_prompt_loras(
                    stage_key=stage_key,
                    prompt=stage_prompt,
                    negative_prompt=stage_negative_prompt,
                )
                if stage_key != "wan_high":
                    out["prompt"] = stage_prompt
                    out["negative_prompt"] = stage_negative_prompt
                raw_stage_sampler = out.get("sampler")
                if raw_stage_sampler is not None:
                    if not isinstance(raw_stage_sampler, str):
                        raise HTTPException(
                            status_code=400,
                            detail=f"'{stage_key}.sampler' must be a string when provided",
                        )
                    stage_sampler = raw_stage_sampler.strip()
                    if stage_sampler:
                        out["sampler"] = _validate_wan22_sampler_field(
                            field_name=f"{stage_key}.sampler",
                            value=stage_sampler,
                            expected_unipc_solver_hint=expected_unipc_solver_hint,
                        )
                    else:
                        out.pop("sampler", None)
                raw_stage_scheduler = out.get("scheduler")
                if raw_stage_scheduler is not None:
                    if not isinstance(raw_stage_scheduler, str):
                        raise HTTPException(
                            status_code=400,
                            detail=f"'{stage_key}.scheduler' must be a string when provided",
                        )
                    stage_scheduler = raw_stage_scheduler.strip()
                    if stage_scheduler:
                        out["scheduler"] = _validate_wan22_scheduler_field(
                            field_name=f"{stage_key}.scheduler",
                            value=stage_scheduler,
                        )
                    else:
                        out.pop("scheduler", None)
                explicit_stage_loras = _normalize_wan_stage_loras(
                    stage_raw=raw,
                    stage_key=stage_key,
                    resolve_asset_by_sha_fn=resolve_asset_by_sha,
                )
                out["loras"] = _merge_wan_stage_loras(prompt_stage_loras, explicit_stage_loras)
                return out

            extras["wan_high"] = _resolve_wan_stage("wan_high")
            extras["wan_low"] = _resolve_wan_stage("wan_low")

        # Resolve sha-selected WAN assets
        if payload.get("wan_vae_path") or payload.get("wan_text_encoder_path") or payload.get("wan_text_encoder_dir"):
            raise HTTPException(status_code=400, detail="WAN sha-only mode: do not send wan_*_path fields; send wan_vae_sha/wan_tenc_sha instead.")

        wan_vae_sha = _require_sha_field("wan_vae_sha")
        wan_tenc_sha = _require_sha_field("wan_tenc_sha")

        extras["wan_vae_path"] = _resolve_wan_vae_path_from_sha(
            wan_vae_sha=wan_vae_sha,
            metadata_dir=wan_metadata_dir,
            resolve_asset_by_sha=resolve_asset_by_sha,
            resolve_vae_path_by_sha=resolve_vae_path_by_sha,
        )

        wan_tenc_path = resolve_asset_by_sha(wan_tenc_sha)
        if not wan_tenc_path:
            raise HTTPException(status_code=409, detail=f"WAN text encoder not found for sha: {wan_tenc_sha}")
        te_lower = str(wan_tenc_path).lower()
        if not (te_lower.endswith(".safetensors") or te_lower.endswith(".gguf")):
            raise HTTPException(
                status_code=409,
                detail=f"WAN text encoder sha must resolve to a .safetensors or .gguf file: {wan_tenc_sha}",
            )
        extras["wan_text_encoder_path"] = wan_tenc_path

        extras["wan_metadata_dir"] = wan_metadata_dir

        # Pass-through of runtime controls (non-model-part config)
        for key in (
            'gguf_offload',
            'gguf_offload_level',
            'gguf_sdpa_policy',
            'gguf_attention_mode',
            'gguf_attn_chunk',
            'gguf_cache_policy',
            'gguf_cache_limit_mb',
            'gguf_log_mem_interval',
            'gguf_te_device',
        ):
            if key in payload and payload.get(key) is not None:
                extras[key] = payload.get(key)
        if 'gguf_attention_mode' in extras:
            attn_mode = str(extras.get('gguf_attention_mode') or '').strip().lower()
            if attn_mode not in {'global', 'sliding'}:
                raise HTTPException(status_code=400, detail=f"Invalid gguf_attention_mode: {extras.get('gguf_attention_mode')!r}")
            extras['gguf_attention_mode'] = attn_mode
        if 'gguf_sdpa_policy' in extras:
            sdpa_policy = str(extras.get('gguf_sdpa_policy') or '').strip().lower()
            if sdpa_policy not in {'auto', 'mem_efficient', 'flash', 'math'}:
                raise HTTPException(status_code=400, detail=f"Invalid gguf_sdpa_policy: {extras.get('gguf_sdpa_policy')!r}")
            extras['gguf_sdpa_policy'] = sdpa_policy
        _normalize_gguf_runtime_controls(extras)
        _normalize_gguf_te_device(extras)
        _normalize_gguf_cache_controls(extras)

        resolved_stage_paths = (
            (str(extras["wan_single"]["model_dir"]),)
            if "wan_single" in extras
            else (str(extras["wan_high"]["model_dir"]), str(extras["wan_low"]["model_dir"]))
        )
        engine_key, wan_engine_variant = _resolve_wan22_engine_key(
            payload,
            metadata_dir=wan_metadata_dir,
            task_type=TaskType.TXT2VID,
            requested_engine_key=video_engine_key,
            resolved_stage_paths=resolved_stage_paths,
        )
        extras["wan_engine_variant"] = wan_engine_variant
        extras["wan_engine_dispatch"] = engine_key
        smart_offload, smart_fallback, smart_cache = _resolve_smart_flags()
        req = Txt2VidRequest(
            task=TaskType.TXT2VID,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width_val,
            height=height_val,
            steps=steps_val,
            fps=fps_val,
            num_frames=frames_val,
            sampler=sampler_name,
            scheduler=scheduler_name,
            seed=seed_val,
            guidance_scale=cfg_val,
            video_options=video_options,
            extras=extras,
            smart_offload=smart_offload,
            smart_fallback=smart_fallback,
            smart_cache=smart_cache,
            settings_revision=settings_revision,
            metadata={
                "styles": payload.get('txt2vid_styles', []),
            },
        )

        model_ref = str(resolved_stage_paths[0])
        return req, engine_key, model_ref

    def prepare_img2vid(payload: Dict[str, Any]) -> Tuple[Img2VidRequest, str, Optional[str]]:
        get_backend_logger('backend.api').info('[api] DEBUG: enter prepare_img2vid')
        settings_revision = _require_int_field(payload, "settings_revision", minimum=0)
        video_engine_key = _canonical_engine_key(payload.get("engine")) if payload.get("engine") is not None else ""
        use_generic_video_route = not _is_legacy_or_wan_video_route_engine(video_engine_key)
        wan_metadata_dir: str | None = None
        expected_unipc_solver_hint: str | None = None
        init_image_data = payload.get('img2vid_init_image')
        if not isinstance(init_image_data, str) or not init_image_data.strip():
            raise HTTPException(
                status_code=400,
                detail="'img2vid_init_image' is required for img2vid requests.",
            )
        extras: Dict[str, Any] = {}
        model_ref: str | None = None
        if use_generic_video_route and video_engine_key == "ltx2":
            model_ref, checkpoint_record = _resolve_generic_video_checkpoint_contract(
                payload=payload,
                extras=extras,
                engine_key=video_engine_key,
            )
            ltx_defaults = _require_ltx2_checkpoint_execution_defaults(checkpoint_record=checkpoint_record)
            extras["ltx_checkpoint_kind"] = ltx_defaults.checkpoint_kind
            ltx_execution_profile, ltx_two_stage_assets = _resolve_ltx2_requested_execution_profile(
                payload=payload,
                checkpoint_record=checkpoint_record,
                defaults=ltx_defaults,
            )
            extras["ltx_execution_profile"] = ltx_execution_profile
            if ltx_two_stage_assets is not None:
                extras["ltx_two_stage_distilled_lora_path"] = str(ltx_two_stage_assets.distilled_lora_path)
                extras["ltx_two_stage_spatial_upsampler_path"] = str(ltx_two_stage_assets.spatial_upsampler_path)
            parsed = _parse_ltx2_generic_video_core_dto(
                payload,
                task_prefix="img2vid",
                execution_profile=ltx_execution_profile,
                default_steps=int(ltx_defaults.default_steps),
                default_cfg_scale=float(ltx_defaults.default_guidance_scale),
            )
        elif use_generic_video_route:
            parsed = _parse_generic_img2vid_core_dto(payload, engine_key=video_engine_key)
        else:
            wan_metadata_dir = _resolve_wan_metadata_dir(payload)
            default_sampler, default_scheduler = _resolve_wan_sampler_scheduler_defaults_from_assets(wan_metadata_dir)
            expected_unipc_solver_hint = _extract_wan22_unipc_solver_hint(default_sampler)
            parsed = _parse_img2vid_core_dto(
                payload,
                default_sampler=default_sampler,
                default_scheduler=default_scheduler,
                expected_unipc_solver_hint=expected_unipc_solver_hint,
            )
        prompt = parsed.prompt
        negative_prompt = parsed.negative_prompt
        width_val = parsed.width
        height_val = parsed.height
        steps_val = parsed.steps
        fps_val = parsed.fps
        frames_val = parsed.num_frames
        sampler_name = parsed.sampler_name
        scheduler_name = parsed.scheduler_name
        seed_val = parsed.seed
        cfg_val = parsed.guidance_scale

        try:
            init_image = media.decode_image(init_image_data)
        except Exception as exc:
            _router_log.warning("img2vid init image validation failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="Invalid 'img2vid_init_image' payload"),
            ) from None

        video_options, video_extras_updates = _parse_video_output_options(payload, route_label="img2vid")
        extras.update(video_extras_updates)
        if use_generic_video_route:
            if model_ref is None:
                model_ref, _checkpoint_record = _resolve_generic_video_checkpoint_contract(
                    payload=payload,
                    extras=extras,
                    engine_key=video_engine_key,
                )
            smart_offload, smart_fallback, smart_cache = _resolve_smart_flags()
            req = Img2VidRequest(
                task=TaskType.IMG2VID,
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=init_image,
                width=width_val,
                height=height_val,
                steps=steps_val,
                fps=fps_val,
                num_frames=frames_val,
                sampler=sampler_name,
                scheduler=scheduler_name,
                seed=seed_val,
                guidance_scale=cfg_val,
                video_options=video_options,
                extras=extras,
                smart_offload=smart_offload,
                smart_fallback=smart_fallback,
                smart_cache=smart_cache,
                settings_revision=settings_revision,
                metadata={
                    "styles": payload.get('img2vid_styles', []),
                },
            )
            return req, video_engine_key, model_ref
        # WAN (GGUF-only): strict sha-only selection for model parts (no raw paths).
        from apps.backend.inventory.cache import resolve_asset_by_sha, resolve_vae_path_by_sha

        def _require_sha_field(key: str) -> str:
            return _require_sha256_field(payload, key)

        has_wan_single = isinstance(payload.get("wan_single"), dict)
        has_wan_high = isinstance(payload.get("wan_high"), dict)
        has_wan_low = isinstance(payload.get("wan_low"), dict)
        if has_wan_single:
            if has_wan_high or has_wan_low:
                raise HTTPException(
                    status_code=400,
                    detail="WAN22 requests must use either 'wan_single' or ('wan_high' + 'wan_low'), not both.",
                )
        elif has_wan_high or has_wan_low:
            if not (has_wan_high and has_wan_low):
                missing_stage = "wan_high" if not has_wan_high else "wan_low"
                raise HTTPException(
                    status_code=400,
                    detail=f"WAN22 14B requests must provide both 'wan_high' and 'wan_low' (missing '{missing_stage}').",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="WAN22 requests must include either 'wan_single' or both 'wan_high' and 'wan_low'.",
            )

        def _resolve_wan_stage_model_path(*, stage_key: str, raw: Mapping[str, Any]) -> str:
            if isinstance(raw.get("model_dir"), str) and str(raw.get("model_dir")).strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"'{stage_key}.model_dir' is unsupported; use '{stage_key}.model_sha'",
                )
            sha = _require_sha256_field(raw, "model_sha")
            model_path = resolve_asset_by_sha(sha)
            if not model_path:
                raise HTTPException(status_code=409, detail=f"WAN stage model not found for sha: {sha}")
            if not str(model_path).lower().endswith(".gguf"):
                raise HTTPException(status_code=409, detail=f"WAN stage sha does not resolve to a .gguf file: {sha}")
            return str(model_path)

        if has_wan_single:
            prompt, negative_prompt = _parse_wan_request_prompt_loras(
                prompt_field_name="img2vid_prompt",
                negative_prompt_field_name="img2vid_neg_prompt",
                lora_owner_field_name="wan_single",
                prompt=str(prompt or "").strip(),
                negative_prompt=str(negative_prompt or "").strip(),
            )
            if not prompt:
                raise HTTPException(status_code=400, detail="'img2vid_prompt' resolved empty after WAN LoRA parsing")

            raw_single = payload.get("wan_single")
            assert isinstance(raw_single, dict)
            _reject_legacy_wan_stage_lora_keys(stage_key="wan_single", stage_raw=raw_single)
            _reject_unknown_keys(raw_single, _WAN_SINGLE_ALLOWED_KEYS, "wan_single")
            explicit_single_loras = _normalize_wan_stage_loras(
                stage_raw=raw_single,
                stage_key="wan_single",
                resolve_asset_by_sha_fn=resolve_asset_by_sha,
            )
            extras["wan_single"] = {
                "model_dir": _resolve_wan_stage_model_path(stage_key="wan_single", raw=raw_single),
                "loras": _merge_wan_stage_loras(explicit_single_loras),
                **(
                    {"flow_shift": raw_single.get("flow_shift")}
                    if raw_single.get("flow_shift") is not None
                    else {}
                ),
            }
        else:
            prompt, negative_prompt = _parse_wan_request_prompt_loras(
                prompt_field_name="img2vid_prompt",
                negative_prompt_field_name="img2vid_neg_prompt",
                lora_owner_field_name="wan_high",
                prompt=str(prompt or "").strip(),
                negative_prompt=str(negative_prompt or "").strip(),
            )
            if not prompt:
                raise HTTPException(status_code=400, detail="'img2vid_prompt' resolved empty after WAN LoRA parsing")

            def _resolve_wan_stage(stage_key: str) -> dict[str, object]:
                raw = payload.get(stage_key)
                if not isinstance(raw, dict):
                    raise HTTPException(status_code=400, detail=f"'{stage_key}' is required and must be an object")
                _reject_legacy_wan_stage_lora_keys(stage_key=stage_key, stage_raw=raw)
                allowed_stage_keys = _WAN_HIGH_ALLOWED_KEYS if stage_key == "wan_high" else _WAN_LOW_ALLOWED_KEYS
                _reject_unknown_keys(raw, allowed_stage_keys, stage_key)
                out: dict[str, object] = dict(raw)
                out.pop("model_sha", None)
                out["model_dir"] = _resolve_wan_stage_model_path(stage_key=stage_key, raw=raw)
                if stage_key == "wan_high":
                    stage_prompt = str(prompt or "").strip()
                    stage_negative_prompt = str(negative_prompt or "").strip()
                else:
                    raw_stage_prompt = out.get("prompt")
                    if not isinstance(raw_stage_prompt, str):
                        raise HTTPException(status_code=400, detail=f"'{stage_key}.prompt' is required and must be a string")
                    stage_prompt = str(raw_stage_prompt).strip()
                    if not stage_prompt:
                        raise HTTPException(status_code=400, detail=f"'{stage_key}.prompt' must be a non-empty string")
                    raw_stage_negative_prompt = out.get("negative_prompt")
                    if raw_stage_negative_prompt is not None and not isinstance(raw_stage_negative_prompt, str):
                        raise HTTPException(
                            status_code=400,
                            detail=f"'{stage_key}.negative_prompt' must be a string when provided",
                        )
                    stage_negative_prompt = (
                        str(raw_stage_negative_prompt).strip()
                        if isinstance(raw_stage_negative_prompt, str)
                        else None
                    )
                stage_prompt, stage_negative_prompt, prompt_stage_loras = _parse_wan_stage_prompt_loras(
                    stage_key=stage_key,
                    prompt=stage_prompt,
                    negative_prompt=stage_negative_prompt,
                )
                if stage_key != "wan_high":
                    out["prompt"] = stage_prompt
                    out["negative_prompt"] = stage_negative_prompt
                raw_stage_sampler = out.get("sampler")
                if raw_stage_sampler is not None:
                    if not isinstance(raw_stage_sampler, str):
                        raise HTTPException(
                            status_code=400,
                            detail=f"'{stage_key}.sampler' must be a string when provided",
                        )
                    stage_sampler = raw_stage_sampler.strip()
                    if stage_sampler:
                        out["sampler"] = _validate_wan22_sampler_field(
                            field_name=f"{stage_key}.sampler",
                            value=stage_sampler,
                            expected_unipc_solver_hint=expected_unipc_solver_hint,
                        )
                    else:
                        out.pop("sampler", None)
                raw_stage_scheduler = out.get("scheduler")
                if raw_stage_scheduler is not None:
                    if not isinstance(raw_stage_scheduler, str):
                        raise HTTPException(
                            status_code=400,
                            detail=f"'{stage_key}.scheduler' must be a string when provided",
                        )
                    stage_scheduler = raw_stage_scheduler.strip()
                    if stage_scheduler:
                        out["scheduler"] = _validate_wan22_scheduler_field(
                            field_name=f"{stage_key}.scheduler",
                            value=stage_scheduler,
                        )
                    else:
                        out.pop("scheduler", None)
                explicit_stage_loras = _normalize_wan_stage_loras(
                    stage_raw=raw,
                    stage_key=stage_key,
                    resolve_asset_by_sha_fn=resolve_asset_by_sha,
                )
                out["loras"] = _merge_wan_stage_loras(prompt_stage_loras, explicit_stage_loras)
                return out

            extras["wan_high"] = _resolve_wan_stage("wan_high")
            extras["wan_low"] = _resolve_wan_stage("wan_low")

        # Resolve sha-selected WAN assets
        if payload.get("wan_vae_path") or payload.get("wan_text_encoder_path") or payload.get("wan_text_encoder_dir"):
            raise HTTPException(status_code=400, detail="WAN sha-only mode: do not send wan_*_path fields; send wan_vae_sha/wan_tenc_sha instead.")

        wan_vae_sha = _require_sha_field("wan_vae_sha")
        wan_tenc_sha = _require_sha_field("wan_tenc_sha")

        extras["wan_vae_path"] = _resolve_wan_vae_path_from_sha(
            wan_vae_sha=wan_vae_sha,
            metadata_dir=wan_metadata_dir,
            resolve_asset_by_sha=resolve_asset_by_sha,
            resolve_vae_path_by_sha=resolve_vae_path_by_sha,
        )

        wan_tenc_path = resolve_asset_by_sha(wan_tenc_sha)
        if not wan_tenc_path:
            raise HTTPException(status_code=409, detail=f"WAN text encoder not found for sha: {wan_tenc_sha}")
        te_lower = str(wan_tenc_path).lower()
        if not (te_lower.endswith(".safetensors") or te_lower.endswith(".gguf")):
            raise HTTPException(
                status_code=409,
                detail=f"WAN text encoder sha must resolve to a .safetensors or .gguf file: {wan_tenc_sha}",
            )
        extras["wan_text_encoder_path"] = wan_tenc_path

        extras["wan_metadata_dir"] = wan_metadata_dir

        # Pass-through of runtime controls (non-model-part config)
        for key in (
            'gguf_offload',
            'gguf_offload_level',
            'gguf_sdpa_policy',
            'gguf_attention_mode',
            'gguf_attn_chunk',
            'gguf_cache_policy',
            'gguf_cache_limit_mb',
            'gguf_log_mem_interval',
            'gguf_te_device',
        ):
            if key in payload and payload.get(key) is not None:
                extras[key] = payload.get(key)
        if 'gguf_attention_mode' in extras:
            attn_mode = str(extras.get('gguf_attention_mode') or '').strip().lower()
            if attn_mode not in {'global', 'sliding'}:
                raise HTTPException(status_code=400, detail=f"Invalid gguf_attention_mode: {extras.get('gguf_attention_mode')!r}")
            extras['gguf_attention_mode'] = attn_mode
        if 'gguf_sdpa_policy' in extras:
            sdpa_policy = str(extras.get('gguf_sdpa_policy') or '').strip().lower()
            if sdpa_policy not in {'auto', 'mem_efficient', 'flash', 'math'}:
                raise HTTPException(status_code=400, detail=f"Invalid gguf_sdpa_policy: {extras.get('gguf_sdpa_policy')!r}")
            extras['gguf_sdpa_policy'] = sdpa_policy
        _normalize_gguf_runtime_controls(extras)
        _normalize_gguf_te_device(extras)
        _normalize_gguf_cache_controls(extras)

        img2vid_mode = str(payload.get('img2vid_mode') or '').strip().lower()
        if img2vid_mode == 'chunk':
            raise HTTPException(
                status_code=400,
                detail="img2vid_mode='chunk' is no longer supported (expected 'solo'|'sliding'|'svi2'|'svi2_pro').",
            )
        if img2vid_mode not in {'solo', 'sliding', 'svi2', 'svi2_pro'}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid img2vid_mode: {payload.get('img2vid_mode')!r} (expected 'solo'|'sliding'|'svi2'|'svi2_pro').",
            )
        extras['img2vid_mode'] = img2vid_mode
        if payload.get('img2vid_image_scale') not in (None, ''):
            image_scale = _require_float_field(payload, 'img2vid_image_scale')
            if not math.isfinite(image_scale) or image_scale <= 0.0:
                raise HTTPException(status_code=400, detail="'img2vid_image_scale' must be a finite number > 0.")
            extras['img2vid_image_scale'] = float(image_scale)
        if payload.get('img2vid_crop_offset_x') in (None, ''):
            extras['img2vid_crop_offset_x'] = 0.5
        else:
            crop_offset_x = _require_float_field(payload, 'img2vid_crop_offset_x')
            if crop_offset_x < 0.0 or crop_offset_x > 1.0:
                raise HTTPException(status_code=400, detail="'img2vid_crop_offset_x' must be within [0, 1].")
            extras['img2vid_crop_offset_x'] = crop_offset_x
        if payload.get('img2vid_crop_offset_y') in (None, ''):
            extras['img2vid_crop_offset_y'] = 0.5
        else:
            crop_offset_y = _require_float_field(payload, 'img2vid_crop_offset_y')
            if crop_offset_y < 0.0 or crop_offset_y > 1.0:
                raise HTTPException(status_code=400, detail="'img2vid_crop_offset_y' must be within [0, 1].")
            extras['img2vid_crop_offset_y'] = crop_offset_y

        has_chunk_frames = payload.get('img2vid_chunk_frames') not in (None, '')
        has_overlap_frames = payload.get('img2vid_overlap_frames') not in (None, '')
        has_anchor_alpha = payload.get('img2vid_anchor_alpha') not in (None, '')
        has_reset_anchor_to_base = payload.get('img2vid_reset_anchor_to_base') not in (None, '')
        has_chunk_seed_mode = payload.get('img2vid_chunk_seed_mode') not in (None, '')
        has_chunk_buffer_mode = payload.get('img2vid_chunk_buffer_mode') not in (None, '')
        has_window_frames = payload.get('img2vid_window_frames') not in (None, '')
        has_window_stride = payload.get('img2vid_window_stride') not in (None, '')
        has_window_commit = payload.get('img2vid_window_commit_frames') not in (None, '')

        has_temporal_fields = any(
            (
                has_chunk_frames,
                has_overlap_frames,
                has_anchor_alpha,
                has_reset_anchor_to_base,
                has_chunk_seed_mode,
                has_chunk_buffer_mode,
                has_window_frames,
                has_window_stride,
                has_window_commit,
            )
        )

        if img2vid_mode == 'solo':
            if has_temporal_fields:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "img2vid_mode='solo' does not allow temporal fields "
                        "(chunk/window/anchor/reset/seed/buffer)."
                    ),
                )
        else:
            mode_label = str(img2vid_mode)
            if has_chunk_frames or has_overlap_frames:
                raise HTTPException(
                    status_code=400,
                    detail=f"img2vid_mode='{mode_label}' does not allow 'img2vid_chunk_frames'/'img2vid_overlap_frames'.",
                )
            if not (has_window_frames and has_window_stride and has_window_commit):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"img2vid_mode='{mode_label}' requires 'img2vid_window_frames', "
                        "'img2vid_window_stride', and 'img2vid_window_commit_frames'."
                    ),
                )
            window_frames = _require_int_field(payload, 'img2vid_window_frames', minimum=9, maximum=401)
            if (window_frames - 1) % 4 != 0:
                raise HTTPException(status_code=400, detail=f"'img2vid_window_frames' must satisfy 4n+1, got {window_frames}.")
            if int(window_frames) >= int(frames_val):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "'img2vid_window_frames' must be smaller than 'img2vid_num_frames' "
                        f"(window={int(window_frames)} total={int(frames_val)})."
                    ),
                )
            window_stride = _require_int_field(payload, 'img2vid_window_stride', minimum=1, maximum=400)
            if int(window_stride) >= int(window_frames):
                raise HTTPException(
                    status_code=400,
                    detail="'img2vid_window_stride' must be smaller than 'img2vid_window_frames'.",
                )
            if int(window_stride) % 4 != 0:
                raise HTTPException(
                    status_code=400,
                    detail="'img2vid_window_stride' must be aligned to temporal scale=4.",
                )
            window_commit = _require_int_field(payload, 'img2vid_window_commit_frames', minimum=1, maximum=401)
            if int(window_commit) < int(window_stride) or int(window_commit) > int(window_frames):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "'img2vid_window_commit_frames' must be within "
                        "[img2vid_window_stride, img2vid_window_frames]."
                    ),
                )
            if (int(window_commit) - int(window_stride)) < 4:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "'img2vid_window_commit_frames' must keep at least 4 committed overlap frames "
                        "beyond 'img2vid_window_stride'."
                    ),
                )
            extras['img2vid_window_frames'] = window_frames
            extras['img2vid_window_stride'] = window_stride
            extras['img2vid_window_commit_frames'] = window_commit

        if img2vid_mode in {'sliding', 'svi2', 'svi2_pro'} and has_anchor_alpha:
            anchor_alpha = _require_float_field(payload, 'img2vid_anchor_alpha')
            if anchor_alpha < 0.0 or anchor_alpha > 1.0:
                raise HTTPException(status_code=400, detail="'img2vid_anchor_alpha' must be within [0, 1].")
            extras['img2vid_anchor_alpha'] = anchor_alpha
        if img2vid_mode in {'sliding'} and has_reset_anchor_to_base:
            extras['img2vid_reset_anchor_to_base'] = _require_bool_field(payload, 'img2vid_reset_anchor_to_base')
        if img2vid_mode in {'svi2', 'svi2_pro'} and has_reset_anchor_to_base:
            reset_anchor_to_base = _require_bool_field(payload, 'img2vid_reset_anchor_to_base')
            if reset_anchor_to_base:
                raise HTTPException(
                    status_code=400,
                    detail=f"img2vid_mode='{img2vid_mode}' requires 'img2vid_reset_anchor_to_base=false'.",
                )
            extras['img2vid_reset_anchor_to_base'] = False
        if img2vid_mode in {'sliding', 'svi2', 'svi2_pro'} and has_chunk_seed_mode:
            seed_mode = str(payload.get('img2vid_chunk_seed_mode') or '').strip().lower()
            if seed_mode not in {'fixed', 'increment', 'random'}:
                raise HTTPException(status_code=400, detail=f"Invalid img2vid_chunk_seed_mode: {payload.get('img2vid_chunk_seed_mode')!r}")
            extras['img2vid_chunk_seed_mode'] = seed_mode
        if img2vid_mode in {'sliding', 'svi2', 'svi2_pro'} and has_chunk_buffer_mode:
            chunk_buffer_mode = str(payload.get('img2vid_chunk_buffer_mode') or '').strip().lower()
            if chunk_buffer_mode not in {'hybrid', 'ram', 'ram+hd'}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid img2vid_chunk_buffer_mode: {payload.get('img2vid_chunk_buffer_mode')!r}",
                )
            extras['img2vid_chunk_buffer_mode'] = chunk_buffer_mode

        resolved_stage_paths = (
            (str(extras["wan_single"]["model_dir"]),)
            if "wan_single" in extras
            else (str(extras["wan_high"]["model_dir"]), str(extras["wan_low"]["model_dir"]))
        )
        engine_key, wan_engine_variant = _resolve_wan22_engine_key(
            payload,
            metadata_dir=wan_metadata_dir,
            task_type=TaskType.IMG2VID,
            requested_engine_key=video_engine_key,
            resolved_stage_paths=resolved_stage_paths,
        )
        extras["wan_engine_variant"] = wan_engine_variant
        extras["wan_engine_dispatch"] = engine_key
        smart_offload, smart_fallback, smart_cache = _resolve_smart_flags()
        req = Img2VidRequest(
            task=TaskType.IMG2VID,
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            width=width_val,
            height=height_val,
            steps=steps_val,
            fps=fps_val,
            num_frames=frames_val,
            sampler=sampler_name,
            scheduler=scheduler_name,
            seed=seed_val,
            guidance_scale=cfg_val,
            video_options=video_options,
            extras=extras,
            smart_offload=smart_offload,
            smart_fallback=smart_fallback,
            smart_cache=smart_cache,
            settings_revision=settings_revision,
            metadata={
                "styles": payload.get('img2vid_styles', []),
            },
        )

        model_ref = str(resolved_stage_paths[0])
        get_backend_logger('backend.api').info('[api] DEBUG: exit prepare_img2vid engine=%s model_ref=%s size=%dx%d frames=%d', engine_key, model_ref, width_val, height_val, frames_val)
        return req, engine_key, model_ref

    def validate_pre_task_img2vid_payload(payload: Dict[str, Any]) -> None:
        init_image_data = payload.get("img2vid_init_image")
        if not isinstance(init_image_data, str) or not init_image_data.strip():
            raise HTTPException(
                status_code=400,
                detail="'img2vid_init_image' is required for img2vid requests.",
            )

    def run_video_task(task_id: str, payload: Dict[str, Any], entry: TaskEntry, task_type: TaskType, *, device: str) -> None:
        from apps.backend.runtime.diagnostics.contract_trace import error_meta
        from apps.backend.runtime.diagnostics.contract_trace import emit_event as emit_contract_trace
        from apps.backend.runtime.diagnostics.contract_trace import hash_request_prompt
        from apps.backend.runtime.diagnostics.fallback_state import fallback_used as fallback_state_used
        from apps.backend.runtime.diagnostics.fallback_state import reset_fallback_state

        def push(event: Dict[str, Any]) -> None:
            entry.push_event(event)

        push({"type": "status", "stage": "queued"})
        try:
            _ensure_default_engines_registered()
            if task_type == TaskType.TXT2VID:
                req, engine_key, model_ref = prepare_txt2vid(payload)
            elif task_type == TaskType.IMG2VID:
                req, engine_key, model_ref = prepare_img2vid(payload)
            else:
                raise RuntimeError(f"Unsupported video task: {task_type}")
            options_snapshot = _opts_snapshot()
            storage_dtype, compute_dtype = _resolve_core_dtype_overrides(options_snapshot)
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
        fallback_enabled = bool(getattr(req, "smart_fallback", False))
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

                from apps.backend.interfaces.api.tasks.generation_tasks import (
                    build_engine_options as _build_engine_options,
                    encode_images as _encode_images,
                    resolve_request_smart_flags as _resolve_request_smart_flags,
                )
                from apps.backend.runtime.memory.smart_offload import smart_runtime_overrides

                engine_opts: dict[str, object] = {
                    "export_video": _require_options_bool(options_snapshot, "codex_export_video")
                }
                if _require_options_bool(options_snapshot, "codex_core_streaming"):
                    engine_opts["core_streaming_enabled"] = True
                request_extras = getattr(req, "extras", {}) or {}
                if any(
                    key in request_extras
                    for key in (
                        "checkpoint_core_only",
                        "model_format",
                        "text_encoder_override",
                        "tenc_path",
                        "vae_path",
                        "vae_source",
                    )
                ):
                    engine_opts.update(
                        _build_engine_options(
                            req=req,
                            engine_key=engine_key,
                            opts_snapshot=lambda: options_snapshot,
                        )
                    )
                if compute_dtype is not None:
                    engine_opts["dtype"] = compute_dtype

                smart_offload, smart_fallback, smart_cache = _resolve_request_smart_flags(req)

                cancelled_immediate = False
                with smart_runtime_overrides(
                    smart_offload=smart_offload,
                    smart_fallback=smart_fallback,
                    smart_cache=smart_cache,
                ):
                    for ev in _ORCH.run(task_type, engine_key, req, model_ref=model_ref, engine_options=engine_opts):
                        if entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE:
                            if not cancelled_immediate:
                                entry.error = build_cancelled_task_error()
                            cancelled_immediate = True
                            # Keep draining orchestrator events so teardown/finalizers complete
                            # before this worker marks done + releases inference gate.
                            continue
                        if isinstance(ev, ProgressEvent):
                            progress_payload: Dict[str, Any] = {
                                "type": "progress",
                                "stage": ev.stage,
                                "percent": ev.percent,
                                "step": ev.step,
                                "total_steps": ev.total_steps,
                                "eta_seconds": ev.eta_seconds,
                            }
                            if ev.message is not None:
                                progress_payload["message"] = ev.message
                            if ev.data:
                                progress_payload["data"] = dict(ev.data)
                            push(
                                progress_payload
                            )
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
                                meta={
                                    "step": ev.step,
                                    "total_steps": ev.total_steps,
                                    "percent": ev.percent,
                                    "message": ev.message,
                                    "data_keys": sorted(str(key) for key in ev.data.keys()) if ev.data else [],
                                },
                            )
                        elif isinstance(ev, ResultEvent):
                            payload_obj = ev.payload or {}
                            info_raw = payload_obj.get("info", "{}")
                            try:
                                info_obj = json.loads(info_raw)
                            except Exception:
                                info_obj = info_raw
                            encoded = _encode_images(payload_obj.get("images", []))
                            result = {"images": encoded, "info": info_obj}
                            if isinstance(payload_obj.get("video"), dict):
                                result["video"] = payload_obj.get("video")
                            entry.result = {"status": "completed", "result": result}
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
                                meta={
                                    "image_count": len(payload_obj.get("images", []) or []),
                                    "has_video": isinstance(payload_obj.get("video"), dict),
                                },
                            )
                success = not cancelled_immediate
            except Exception as err:
                engine_execution_error = False
                try:
                    from apps.backend.core.exceptions import EngineExecutionError, EngineLoadError

                    engine_execution_error = isinstance(err, (EngineExecutionError, EngineLoadError))
                except Exception:
                    pass
                if not engine_execution_error:
                    try:
                        from apps.backend.runtime.diagnostics.exception_hook import dump_exception as _dump_exc

                        _dump_exc(type(err), err, err.__traceback__, where=f"{label}_worker", context={"task_id": task_id})
                    except Exception:
                        pass
                cleanup_err: Exception | None = None
                try:
                    from apps.backend.interfaces.api.tasks.generation_tasks import (
                        force_runtime_memory_cleanup as _force_runtime_memory_cleanup,
                    )

                    _force_runtime_memory_cleanup(
                        reason=f"{mode}:worker_error",
                        orch=_ORCH,
                    )
                except Exception as cleanup_exc:
                    cleanup_err = cleanup_exc
                    _router_log.error(
                        "Runtime memory cleanup failed after %s worker error (task_id=%s): %s",
                        label,
                        task_id,
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
                        _router_log.warning(
                            "inference gate release failed in %s_worker (task_id=%s): %s",
                            label,
                            task_id,
                            exc,
                            exc_info=False,
                        )

        if task_type == TaskType.TXT2VID:
            label = "txt2vid"
        elif task_type == TaskType.IMG2VID:
            label = "img2vid"
        else:
            raise RuntimeError(f"Unsupported video task: {task_type}")
        thread = threading.Thread(target=worker, name=f"{label}-task-{task_id}", daemon=True)
        thread.start()

    @router.post('/api/txt2img')
    async def txt2img(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")
        _validate_txt2img_hires_request_payload(payload)
        _validate_pre_task_txt2img_payload(payload)
        _enforce_generation_settings_contract(payload)
        _validate_route_engine_capability(payload, route_mode=GenerationRouteMode.TXT2IMG)
        _reject_zimage_l2p_txt2img_payload(payload)

        device = _parse_explicit_device(
            payload,
            route_mode=GenerationRouteMode.TXT2IMG,
        )
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-{uuid4().hex})"
        register_task(task_id, entry)
        run_txt2img_task(task_id, payload, entry, device=device)
        return {"task_id": task_id}

    @router.post('/api/img2img')
    async def img2img(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")
        _validate_pre_task_img2img_payload(payload)
        _enforce_generation_settings_contract(payload)
        _validate_route_engine_capability(payload, route_mode=GenerationRouteMode.IMG2IMG)

        device = _parse_explicit_device(
            payload,
            route_mode=GenerationRouteMode.IMG2IMG,
        )
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-img2img-{uuid4().hex})"
        register_task(task_id, entry)
        run_img2img_task(task_id, payload, entry, device=device)
        return {"task_id": task_id}

    @router.post('/api/image-automation')
    async def image_automation(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")
        request = _parse_image_automation_request(payload)
        route_mode = GenerationRouteMode.TXT2IMG if request.mode == "txt2img" else GenerationRouteMode.IMG2IMG
        device = _parse_explicit_device(dict(request.template), route_mode=route_mode)
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-image-automation-{uuid4().hex})"
        register_task(task_id, entry)
        from apps.backend.interfaces.api.tasks.generation_tasks import run_image_automation_task as _run_image_automation_task

        _run_image_automation_task(
            task_id=task_id,
            request=request,
            entry=entry,
            device=device,
            prepare_txt2img=prepare_txt2img,
            prepare_img2img=prepare_img2img,
            orch=_ORCH,
            ensure_default_engines_registered=_ensure_default_engines_registered,
            live_preview=live_preview,
            opts_get=_opts_get,
            opts_snapshot=_opts_snapshot,
            generation_provenance=_GENERATION_PROVENANCE,
            save_generated_images=_save_generated_images,
        )
        return {"task_id": task_id}

    @router.post('/api/txt2vid')
    async def txt2vid(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")
        _enforce_generation_settings_contract(payload)
        _validate_route_engine_capability(payload, route_mode=GenerationRouteMode.TXT2VID)
        _reject_legacy_wan_request_key_aliases(payload, context="txt2vid")
        _validate_pre_task_txt2vid_payload(payload)
        _preflight_generic_video_route_checkpoint_contract(payload)

        device = _parse_explicit_device(payload, route_mode=GenerationRouteMode.TXT2VID)
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-txt2vid-{uuid4().hex})"
        register_task(task_id, entry)
        run_video_task(task_id, payload, entry, TaskType.TXT2VID, device=device)
        return {"task_id": task_id}

    @router.post('/api/img2vid')
    async def img2vid(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        get_backend_logger('backend.api').info('[api] DEBUG: POST /api/img2vid received')
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")
        _enforce_generation_settings_contract(payload)
        _validate_route_engine_capability(payload, route_mode=GenerationRouteMode.IMG2VID)
        _reject_legacy_wan_request_key_aliases(payload, context="img2vid")
        validate_pre_task_img2vid_payload(payload)
        _preflight_generic_video_route_checkpoint_contract(payload)

        device = _parse_explicit_device(payload, route_mode=GenerationRouteMode.IMG2VID)
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-img2vid-{uuid4().hex})"
        register_task(task_id, entry)
        get_backend_logger('backend.api').info('[api] DEBUG: scheduling img2vid task_id=%s', task_id)
        run_video_task(task_id, payload, entry, TaskType.IMG2VID, device=device)
        return {"task_id": task_id}

    @router.post('/api/vid2vid')
    async def vid2vid(
        video: UploadFile | None = File(default=None),
        reference_image: UploadFile | None = File(default=None),
        pose_video: UploadFile | None = File(default=None),
        face_video: UploadFile | None = File(default=None),
        background_video: UploadFile | None = File(default=None),
        mask_video: UploadFile | None = File(default=None),
        payload: str = Form(default="{}"),
    ) -> Dict[str, Any]:
        """Video-to-video endpoint.

        The route is intentionally parked until a native vid2vid family lands end-to-end.
        """
        try:
            data = json.loads(payload) if payload else {}
        except Exception as exc:
            _router_log.warning("vid2vid payload JSON parse failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail=public_http_error_detail(exc, fallback="payload must be valid JSON"),
            ) from None
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="payload must be JSON object")

        del reference_image, pose_video, face_video, background_video, video, mask_video
        raise HTTPException(status_code=400, detail=_PARKED_VID2VID_ROUTE_DETAIL)

    return router

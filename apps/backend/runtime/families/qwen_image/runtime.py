"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native Qwen Image Edit-2511 single-image component runtime.
Owns condition/reference preprocessing, shared-vision multimodal prompt encoding, request-static transformer context,
FlowMatch Euler denoising with one terminal device-validity read, final VAE decode, atomic stage-exclusive patcher
lifecycle, streamed-core memory admission, and primary-stage exception preservation while delegating tensor contracts
and latent math to `runtime_latents.py`.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageRuntime` (class): Exact Edit-2511 conditioning, reference encode, denoise, and decode owner.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from apps.backend.core.rng import ImageRNG, NoiseSettings
from apps.backend.core.state import state as backend_state
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.memory.streamed_residency import StreamedResidencyPhase
from apps.backend.runtime.processing.conditioners import normalize_torch_manual_seed
from apps.backend.runtime.logging import get_backend_logger
from apps.backend.runtime.sampling.block_progress import (
    BLOCK_PROGRESS_CALLBACK_KEY,
    validate_block_progress_payload,
)

from .config import (
    QWEN_IMAGE_CONTEXT_DIM,
    QWEN_IMAGE_DEFAULT_TRUE_CFG,
    QWEN_IMAGE_EDIT_VARIANT,
    QWEN_IMAGE_LATENT_CHANNELS,
    qwen_image_edit_condition_dimensions,
    qwen_image_edit_vae_dimensions,
    require_qwen_image_variant,
)
from .runtime_latents import (
    QwenImageConditioning,
    QwenImageDenoisedLatents,
    QwenImageReferenceLatents,
    _QwenImageDenoiseError,
    _device_error_flag,
    qwen_image_pack_latents,
    qwen_image_true_cfg,
    qwen_image_unpack_latents,
)
from .scheduler import (
    QwenImageLatentGrid,
    qwen_image_flow_euler_step,
    qwen_image_flow_schedule,
    qwen_image_latent_grid,
)
from .streaming import QwenImageStreamedCoreRuntime
from .text_encoder import (
    QwenImageProcessorBatch,
    QwenImageTextEncoderRuntime,
    qwen_image_prompt_plan,
)
from .transformer import (
    QWEN_IMAGE_NUM_LAYERS,
    QwenImageTransformer2DModel,
    QwenImageTransformerRequestContext,
)
from .vae import (
    QwenImageVaeConfig,
    qwen_image_denormalize_latents,
    qwen_image_normalize_latents,
)

if TYPE_CHECKING:
    from apps.backend.patchers.base import ModelPatcher
    from apps.backend.patchers.denoiser import DenoiserPatcher
    from apps.backend.patchers.vae import VAE

    from .loader import QwenImageComponentAssembly


_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

_CONDITIONING_SOURCE = "runtime.families.qwen_image.runtime.QwenImageRuntime.encode_conditioning"
_CONDITIONING_STAGE = "qwen_image_conditioning"
_CONDITIONING_COMPONENT = "text_encoder:qwen2_5_vl_7b"

_REFERENCE_SOURCE = "runtime.families.qwen_image.runtime.QwenImageRuntime.encode_reference"
_REFERENCE_STAGE = "qwen_image_vae_encode"
_REFERENCE_COMPONENT = "vae"

_DENOISE_SOURCE = "runtime.families.qwen_image.runtime.QwenImageRuntime.denoise"
_DENOISE_STAGE = "qwen_image_denoise"
_DENOISE_COMPONENT = "denoiser"

_DECODE_SOURCE = "runtime.families.qwen_image.runtime.QwenImageRuntime.decode"
_DECODE_STAGE = "qwen_image_vae_decode"
_DECODE_COMPONENT = "vae"

logger = get_backend_logger("backend.runtime.families.qwen_image.runtime")


@dataclass(frozen=True, slots=True)
class _QwenImageDenoiseMemoryBudget:
    image_tokens: int
    active_text_tokens: int
    working_sequence_tokens: int
    working_set_bytes: int
    persistent_input_bytes: int
    memory_required: int
    hard_memory_preservation: int


def _resized_rgb_image(image: Image.Image, *, width: int, height: int, label: str) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"{label} must be a PIL.Image.Image; got {type(image).__name__}.")
    rgb = image.convert("RGB")
    if rgb.size == (int(width), int(height)):
        return rgb
    return rgb.resize((int(width), int(height)), resample=_RESAMPLE_LANCZOS)


def _vae_pixel_batch(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy()
    if array.ndim != 3 or tuple(array.shape[2:]) != (3,):
        raise RuntimeError(f"Qwen Image VAE preprocessing expected HWC RGB data; got shape={array.shape}.")
    return torch.from_numpy(array).div_(255.0).unsqueeze(0).contiguous()


def _masked_prompt_features(
    hidden_states: torch.Tensor,
    batch: QwenImageProcessorBatch,
    *,
    template_start_idx: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(
            f"{label} hidden states must be a torch.Tensor; got {type(hidden_states).__name__}."
        )
    bool_mask = batch.attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    valid_lengths = bool_mask.sum(dim=1)
    if tuple(valid_lengths.shape) != (1,):
        raise RuntimeError(f"{label} conditioning requires exactly one processor sample.")
    selected = hidden_states[bool_mask]
    expected_valid = int(valid_lengths[0].item())
    if int(selected.shape[0]) != expected_valid:
        raise RuntimeError(
            f"{label} masked hidden-state extraction mismatch: got={int(selected.shape[0])} expected={expected_valid}."
        )
    retained = selected[int(template_start_idx) :]
    if retained.ndim != 2 or int(retained.shape[0]) <= 0 or int(retained.shape[1]) != QWEN_IMAGE_CONTEXT_DIM:
        raise RuntimeError(
            f"{label} retained hidden states must have shape [S,{QWEN_IMAGE_CONTEXT_DIM}] with S>0; "
            f"got {tuple(retained.shape)}."
        )
    embeddings = retained.unsqueeze(0).detach().to(device=memory_management.manager.cpu_device).contiguous()
    mask = torch.ones(
        (1, int(retained.shape[0])),
        device=memory_management.manager.cpu_device,
        dtype=torch.long,
    )
    return embeddings, mask


def _target_tensor_bytes(tensor: torch.Tensor, *, dtype: torch.dtype) -> int:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Qwen Image memory accounting requires torch.Tensor; got {type(tensor).__name__}.")
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Qwen Image memory accounting requires torch.dtype; got {type(dtype).__name__}.")
    return int(tensor.numel()) * int(torch.empty((), dtype=dtype).element_size())


@contextmanager
def _managed_component_stage(
    target: object,
    *,
    source: str,
    stage: str,
    component_hint: str,
    memory_required: int = 0,
    hard_memory_preservation: int = 0,
) -> Iterator[None]:
    manager = memory_management.manager
    manager.load_model(
        target,
        source=source,
        stage=stage,
        component_hint=component_hint,
        memory_required=memory_required,
        hard_memory_preservation=hard_memory_preservation,
    )
    try:
        yield
    except BaseException as stage_error:
        cleanup_errors: list[tuple[str, BaseException]] = []
        try:
            manager.unload_model(
                target,
                source=source,
                stage=stage,
                component_hint=component_hint,
            )
        except BaseException as cleanup_error:
            cleanup_errors.append(("unload_model", cleanup_error))
        try:
            manager.soft_empty_cache(force=True)
        except BaseException as cleanup_error:
            cleanup_errors.append(("soft_empty_cache", cleanup_error))
        for cleanup_operation, cleanup_error in cleanup_errors:
            stage_error.add_note(
                "Qwen Image secondary cleanup failure after the primary stage error: "
                f"operation={cleanup_operation} error={type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    else:
        unload_error: BaseException | None = None
        try:
            manager.unload_model(
                target,
                source=source,
                stage=stage,
                component_hint=component_hint,
            )
        except BaseException as cleanup_error:
            unload_error = cleanup_error
        try:
            manager.soft_empty_cache(force=True)
        except BaseException as cache_error:
            if unload_error is None:
                raise
            unload_error.add_note(
                "Qwen Image secondary cleanup failure after unload_model failed: "
                f"operation=soft_empty_cache error={type(cache_error).__name__}: {cache_error}"
            )
        if unload_error is not None:
            raise unload_error


class QwenImageRuntime:
    """Exact single-image Edit-2511 tensor and component lifecycle owner."""

    def __init__(self, assembly: QwenImageComponentAssembly) -> None:
        self.variant = require_qwen_image_variant(
            assembly.variant,
            context="Qwen Image runtime assembly variant",
        )
        if self.variant != QWEN_IMAGE_EDIT_VARIANT:
            raise RuntimeError(
                f"Qwen Image runtime supports only {QWEN_IMAGE_EDIT_VARIANT!r}; got {self.variant!r}."
            )
        if not isinstance(assembly.transformer, QwenImageTransformer2DModel):
            raise TypeError(
                "Qwen Image runtime transformer must be QwenImageTransformer2DModel; "
                f"got {type(assembly.transformer).__name__}."
            )
        if not isinstance(assembly.text_encoder, QwenImageTextEncoderRuntime):
            raise TypeError(
                "Qwen Image runtime text encoder must be QwenImageTextEncoderRuntime; "
                f"got {type(assembly.text_encoder).__name__}."
            )
        self.transformer = assembly.transformer
        self.denoiser: DenoiserPatcher = assembly.denoiser
        if not isinstance(assembly.streamed_core_runtime, QwenImageStreamedCoreRuntime):
            raise TypeError(
                "Qwen Image runtime assembly streamed core must be QwenImageStreamedCoreRuntime; "
                f"got {type(assembly.streamed_core_runtime).__name__}."
            )
        if self.denoiser.streamed_core_runtime is not assembly.streamed_core_runtime:
            raise RuntimeError("Qwen Image denoiser and assembly must share one streamed core runtime owner.")
        self.streamed_core_runtime = assembly.streamed_core_runtime
        self.text_encoder = assembly.text_encoder
        self.text_encoder_patcher: ModelPatcher = assembly.text_encoder_patcher
        self.vae: VAE = assembly.vae
        self.vae_config: QwenImageVaeConfig = assembly.vae_config
        self.scheduler_config = assembly.scheduler_config
        self._component_stage_lock = threading.RLock()

    @contextmanager
    def _component_stage_lease(
        self,
        *,
        stage: str,
        forbidden: Sequence[tuple[str, object]],
    ) -> Iterator[None]:
        with self._component_stage_lock:
            resident = [label for label, target in forbidden if memory_management.manager.is_model_loaded(target)]
            if resident:
                raise RuntimeError(
                    f"Qwen Image stage {stage!r} requires exclusive component residency; "
                    f"still_loaded={resident}."
                )
            self._require_streamed_phase(StreamedResidencyPhase.OFFLOADED, stage=stage)
            yield

    def _require_streamed_phase(
        self,
        expected: StreamedResidencyPhase,
        *,
        stage: str,
    ) -> None:
        snapshot = self.streamed_core_runtime.verify_residency()
        if snapshot.phase is not expected:
            raise RuntimeError(
                f"Qwen Image stage {stage!r} requires streamed core phase {expected.value!r}; "
                f"got {snapshot.phase.value!r}."
            )

    def _denoise_memory_budget(
        self,
        *,
        conditioning: QwenImageConditioning,
        reference: QwenImageReferenceLatents,
        grid: QwenImageLatentGrid,
        cfg_scale: float,
        compute_dtype: torch.dtype,
    ) -> _QwenImageDenoiseMemoryBudget:
        dtype_bytes = int(torch.empty((), dtype=compute_dtype).element_size())
        image_tokens = int(grid.sequence_length) + int(reference.grid.sequence_length)
        positive_text_tokens = int(conditioning.positive_mask.to(dtype=torch.bool).sum().item())
        negative_text_tokens = int(conditioning.negative_mask.to(dtype=torch.bool).sum().item())
        use_true_cfg = cfg_scale > 1.0
        active_text_tokens = max(positive_text_tokens, negative_text_tokens) if use_true_cfg else positive_text_tokens
        working_sequence_tokens = image_tokens + active_text_tokens
        working_set_bytes = (
            64
            * working_sequence_tokens
            * int(self.transformer.inner_dim)
            * dtype_bytes
        )
        generated_elements = (
            int(grid.sequence_length)
            * int(self.transformer.config.in_channels)
        )
        persistent_input_bytes = generated_elements * dtype_bytes
        persistent_input_bytes += _target_tensor_bytes(reference.packed_latents, dtype=compute_dtype)
        persistent_input_bytes += _target_tensor_bytes(conditioning.positive_embeddings, dtype=compute_dtype)
        persistent_input_bytes += _target_tensor_bytes(conditioning.positive_mask, dtype=torch.long)
        if use_true_cfg:
            persistent_input_bytes += _target_tensor_bytes(conditioning.negative_embeddings, dtype=compute_dtype)
            persistent_input_bytes += _target_tensor_bytes(conditioning.negative_mask, dtype=torch.long)
        memory_required = working_set_bytes + persistent_input_bytes
        budgets = memory_management.manager.config.budgets
        configured_hard_bytes = max(0, int(budgets.hard_reservation_mb)) * 1024 * 1024
        configured_safety_bytes = max(0, int(budgets.safety_margin_mb)) * 1024 * 1024
        hard_memory_preservation = max(1 << 30, configured_hard_bytes) + configured_safety_bytes
        return _QwenImageDenoiseMemoryBudget(
            image_tokens=image_tokens,
            active_text_tokens=active_text_tokens,
            working_sequence_tokens=working_sequence_tokens,
            working_set_bytes=working_set_bytes,
            persistent_input_bytes=persistent_input_bytes,
            memory_required=memory_required,
            hard_memory_preservation=hard_memory_preservation,
        )

    @torch.inference_mode()
    def encode_conditioning(
        self,
        *,
        prompt: object,
        negative_prompt: object,
        image: Image.Image,
    ) -> QwenImageConditioning:
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Qwen Image condition image must be a PIL.Image.Image; got {type(image).__name__}."
            )
        condition_width, condition_height = qwen_image_edit_condition_dimensions(*image.size)
        condition_image = _resized_rgb_image(
            image,
            width=condition_width,
            height=condition_height,
            label="Qwen Image condition image",
        )
        positive_plan = qwen_image_prompt_plan(prompt, variant=self.variant)
        negative_plan = qwen_image_prompt_plan(negative_prompt, variant=self.variant)
        positive_batch, negative_batch = self.text_encoder.prepare_conditioning_batches(
            positive_prompt_plan=positive_plan,
            negative_prompt_plan=negative_plan,
            image=condition_image,
        )

        try:
            with (
                self._component_stage_lease(
                    stage=_CONDITIONING_STAGE,
                    forbidden=(("vae", self.vae.patcher), ("denoiser", self.denoiser)),
                ),
                _managed_component_stage(
                    self.text_encoder_patcher,
                    source=_CONDITIONING_SOURCE,
                    stage=_CONDITIONING_STAGE,
                    component_hint=_CONDITIONING_COMPONENT,
                ),
            ):
                positive_hidden_states, negative_hidden_states = (
                    self.text_encoder.forward_conditioning_batches(
                        positive_batch,
                        negative_batch,
                    )
                )
                positive_embeddings, positive_mask = _masked_prompt_features(
                    positive_hidden_states,
                    positive_batch,
                    template_start_idx=positive_plan.template_start_idx,
                    label="Qwen Image positive",
                )
                negative_embeddings, negative_mask = _masked_prompt_features(
                    negative_hidden_states,
                    negative_batch,
                    template_start_idx=negative_plan.template_start_idx,
                    label="Qwen Image negative",
                )
        except Exception as exc:
            raise RuntimeError(
                f"Qwen Image stage {_CONDITIONING_STAGE!r} component {_CONDITIONING_COMPONENT!r} failed: {exc}"
            ) from exc

        return QwenImageConditioning(
            positive_embeddings=positive_embeddings,
            positive_mask=positive_mask,
            negative_embeddings=negative_embeddings,
            negative_mask=negative_mask,
            condition_width=condition_width,
            condition_height=condition_height,
        )

    @torch.inference_mode()
    def encode_reference(self, *, image: Image.Image) -> QwenImageReferenceLatents:
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Qwen Image VAE reference image must be a PIL.Image.Image; got {type(image).__name__}."
            )
        reference_width, reference_height = qwen_image_edit_vae_dimensions(*image.size)
        reference_image = _resized_rgb_image(
            image,
            width=reference_width,
            height=reference_height,
            label="Qwen Image VAE reference image",
        )
        pixels = _vae_pixel_batch(reference_image)
        grid = qwen_image_latent_grid(reference_width, reference_height)

        try:
            with (
                self._component_stage_lease(
                    stage=_REFERENCE_STAGE,
                    forbidden=(("text_encoder", self.text_encoder_patcher), ("denoiser", self.denoiser)),
                ),
                _managed_component_stage(
                    self.vae.patcher,
                    source=_REFERENCE_SOURCE,
                    stage=_REFERENCE_STAGE,
                    component_hint=_REFERENCE_COMPONENT,
                ),
            ):
                raw_latents = self.vae.encode(pixels)
                expected_shape = (
                    1,
                    QWEN_IMAGE_LATENT_CHANNELS,
                    int(grid.latent_height),
                    int(grid.latent_width),
                )
                if tuple(raw_latents.shape) != expected_shape:
                    raise RuntimeError(
                        "Qwen Image reference VAE latent shape mismatch: "
                        f"got={tuple(raw_latents.shape)} expected={expected_shape}."
                    )
                normalized = qwen_image_normalize_latents(raw_latents, self.vae_config)
                packed = qwen_image_pack_latents(normalized)
                packed_cpu = packed.detach().to(device=memory_management.manager.cpu_device).contiguous()
        except Exception as exc:
            raise RuntimeError(
                f"Qwen Image stage {_REFERENCE_STAGE!r} component {_REFERENCE_COMPONENT!r} failed: {exc}"
            ) from exc

        return QwenImageReferenceLatents(packed_latents=packed_cpu, grid=grid)

    def _transformer_prediction(
        self,
        *,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        request_context: QwenImageTransformerRequestContext,
        progress_owner_token: str,
        block_offset: int,
        total_blocks: int,
    ) -> torch.Tensor:
        def _on_block_progress(block_index: int, inner_total: int) -> None:
            normalized_index, normalized_total = validate_block_progress_payload(block_index, inner_total)
            if normalized_total != QWEN_IMAGE_NUM_LAYERS:
                raise RuntimeError(
                    "Qwen Image transformer block count changed during denoise: "
                    f"got={normalized_total} expected={QWEN_IMAGE_NUM_LAYERS}."
                )
            if backend_state.should_stop:
                raise RuntimeError("cancelled")
            backend_state.update_sampling_block(
                block_index=int(block_offset) + int(normalized_index),
                total_blocks=int(total_blocks),
                owner_token=progress_owner_token,
            )

        transformer_options: Mapping[str, object] = {
            BLOCK_PROGRESS_CALLBACK_KEY: _on_block_progress,
        }
        output = self.transformer(
            latent_model_input,
            encoder_hidden_states=embeddings,
            encoder_hidden_states_mask=mask,
            timestep=timestep,
            request_context=request_context,
            transformer_options=transformer_options,
            return_dict=True,
        ).sample
        return output

    @staticmethod
    def _raise_denoise_error_mask(error_mask: int) -> None:
        if type(error_mask) is not int or error_mask <= 0:
            raise RuntimeError(
                "Qwen Image denoise terminal error mask must be a positive exact int; "
                f"got {error_mask!r}."
            )
        descriptions = (
            (
                _QwenImageDenoiseError.POSITIVE_PREDICTION_NONFINITE,
                "positive prediction contains non-finite values",
            ),
            (
                _QwenImageDenoiseError.CFG_NORM_NONFINITE,
                "true-CFG prediction norms contain non-finite values",
            ),
            (
                _QwenImageDenoiseError.CFG_COMBINED_ZERO_NORM,
                "true-CFG combined prediction has zero norm",
            ),
            (
                _QwenImageDenoiseError.CFG_RESULT_NONFINITE,
                "true-CFG output contains non-finite values",
            ),
            (
                _QwenImageDenoiseError.EULER_RESULT_NONFINITE,
                "Euler update contains non-finite values",
            ),
        )
        causes = [description for flag, description in descriptions if error_mask & int(flag)]
        known_mask = sum(int(flag) for flag, _description in descriptions)
        unknown_bits = error_mask & ~known_mask
        if unknown_bits:
            causes.append(f"unknown validity bits 0x{unknown_bits:x}")
        raise RuntimeError(
            "Qwen Image denoise device validation failed after the final step: "
            f"mask=0x{error_mask:x} causes={'; '.join(causes)}."
        )

    @staticmethod
    def _read_denoise_error_flags(error_flags: torch.Tensor) -> int:
        if error_flags.ndim != 0 or error_flags.dtype is not torch.int32:
            raise RuntimeError(
                "Qwen Image denoise terminal validity flags must be a zero-dimensional int32 tensor; "
                f"shape={tuple(error_flags.shape)} dtype={error_flags.dtype}."
            )
        return int(error_flags.item())

    def _run_loaded_denoise(
        self,
        *,
        conditioning: QwenImageConditioning,
        reference: QwenImageReferenceLatents,
        grid: QwenImageLatentGrid,
        step_count: int,
        normalized_seed: int,
        owner_token: str,
        cfg_scale: float,
        noise_settings: NoiseSettings,
        compute_device: torch.device,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        rng = ImageRNG(
            shape=(QWEN_IMAGE_LATENT_CHANNELS, int(grid.latent_height), int(grid.latent_width)),
            seeds=(normalized_seed,),
            subseeds=(),
            subseed_strength=0.0,
            settings=noise_settings,
            device=compute_device,
        )
        generated = qwen_image_pack_latents(rng.next().to(device=compute_device, dtype=compute_dtype))
        reference_tokens = reference.packed_latents.to(device=compute_device, dtype=compute_dtype)
        positive_embeddings = conditioning.positive_embeddings.to(device=compute_device, dtype=compute_dtype)
        positive_mask = conditioning.positive_mask
        use_true_cfg = cfg_scale > 1.0
        negative_embeddings = (
            conditioning.negative_embeddings.to(device=compute_device, dtype=compute_dtype)
            if use_true_cfg
            else None
        )
        negative_mask = conditioning.negative_mask if use_true_cfg else None
        schedule = qwen_image_flow_schedule(
            step_count,
            image_seq_len=grid.sequence_length,
            scheduler_config=self.scheduler_config,
            device=compute_device,
        )
        image_shapes = (
            (
                (1, int(grid.packed_height), int(grid.packed_width)),
                (1, int(reference.grid.packed_height), int(reference.grid.packed_width)),
            ),
        )
        request_contexts: dict[int, QwenImageTransformerRequestContext] = {}
        positive_text_length = int(positive_embeddings.shape[1])
        positive_context = self.transformer.prepare_request_context(
            image_shapes,
            batch_size=int(positive_embeddings.shape[0]),
            text_sequence_length=positive_text_length,
            compute_device=compute_device,
        )
        request_contexts[positive_text_length] = positive_context
        negative_context: QwenImageTransformerRequestContext | None = None
        if use_true_cfg:
            if negative_embeddings is None:
                raise RuntimeError("Qwen Image true CFG negative embeddings were not staged on the compute device.")
            negative_text_length = int(negative_embeddings.shape[1])
            negative_context = request_contexts.get(negative_text_length)
            if negative_context is None:
                negative_context = self.transformer.prepare_request_context(
                    image_shapes,
                    batch_size=int(negative_embeddings.shape[0]),
                    text_sequence_length=negative_text_length,
                    compute_device=compute_device,
                )
                request_contexts[negative_text_length] = negative_context
        blocks_per_step = QWEN_IMAGE_NUM_LAYERS * (2 if use_true_cfg else 1)
        error_flags = torch.zeros((), device=compute_device, dtype=torch.int32)

        backend_state.start(
            job_count=1,
            sampling_steps=step_count,
            progress_owner_token=owner_token,
        )
        try:
            backend_state.reset_sampling_blocks(owner_token=owner_token)
            for step_index in range(step_count):
                if backend_state.should_stop:
                    raise RuntimeError("cancelled")
                backend_state.update_sampling(
                    step=step_index,
                    total=step_count,
                    owner_token=owner_token,
                )
                latent_model_input = torch.cat((generated, reference_tokens), dim=1)
                timestep = schedule.timesteps[step_index].expand(1).to(dtype=compute_dtype)
                model_timestep = timestep / float(self.scheduler_config.num_train_timesteps)
                positive_prediction = self._transformer_prediction(
                    latent_model_input=latent_model_input,
                    timestep=model_timestep,
                    embeddings=positive_embeddings,
                    mask=positive_mask,
                    request_context=positive_context,
                    progress_owner_token=owner_token,
                    block_offset=0,
                    total_blocks=blocks_per_step,
                )[:, : int(grid.sequence_length)]
                if use_true_cfg:
                    if negative_embeddings is None or negative_mask is None or negative_context is None:
                        raise RuntimeError("Qwen Image true CFG inputs were not staged on the compute device.")
                    negative_prediction = self._transformer_prediction(
                        latent_model_input=latent_model_input,
                        timestep=model_timestep,
                        embeddings=negative_embeddings,
                        mask=negative_mask,
                        request_context=negative_context,
                        progress_owner_token=owner_token,
                        block_offset=QWEN_IMAGE_NUM_LAYERS,
                        total_blocks=blocks_per_step,
                    )[:, : int(grid.sequence_length)]
                    prediction, cfg_error_flags = qwen_image_true_cfg(
                        positive_prediction,
                        negative_prediction,
                        scale=cfg_scale,
                    )
                    error_flags = torch.bitwise_or(error_flags, cfg_error_flags)
                else:
                    prediction = positive_prediction
                    positive_nonfinite = torch.logical_not(torch.isfinite(prediction).all())
                    error_flags = torch.bitwise_or(
                        error_flags,
                        _device_error_flag(
                            positive_nonfinite,
                            _QwenImageDenoiseError.POSITIVE_PREDICTION_NONFINITE,
                        ),
                    )
                generated, euler_error_flags = qwen_image_flow_euler_step(
                    prediction,
                    generated,
                    current_sigma=schedule.sigmas[step_index],
                    next_sigma=schedule.sigmas[step_index + 1],
                )
                error_flags = torch.bitwise_or(error_flags, euler_error_flags)
                backend_state.tick(
                    sampling_step=step_index + 1,
                    owner_token=owner_token,
                )
                backend_state.reset_sampling_blocks(owner_token=owner_token)
        finally:
            backend_state.end()
            backend_state.clear_flags()

        terminal_error_mask = self._read_denoise_error_flags(error_flags)
        if terminal_error_mask:
            self._raise_denoise_error_mask(terminal_error_mask)
        return generated.detach().to(device=memory_management.manager.cpu_device).contiguous()

    @torch.inference_mode()
    def denoise(
        self,
        *,
        conditioning: QwenImageConditioning,
        reference: QwenImageReferenceLatents,
        width: object,
        height: object,
        steps: object,
        seed: object,
        progress_owner_token: object,
        true_cfg_scale: object = QWEN_IMAGE_DEFAULT_TRUE_CFG,
        noise_settings: NoiseSettings | None = None,
    ) -> QwenImageDenoisedLatents:
        if not isinstance(conditioning, QwenImageConditioning):
            raise TypeError(
                f"Qwen Image denoise conditioning must be QwenImageConditioning; got {type(conditioning).__name__}."
            )
        if not isinstance(reference, QwenImageReferenceLatents):
            raise TypeError(
                f"Qwen Image denoise reference must be QwenImageReferenceLatents; got {type(reference).__name__}."
            )
        try:
            step_count = int(steps)
        except Exception as exc:  # noqa: BLE001 - strict runtime validation
            raise RuntimeError("Qwen Image denoise steps must be an integer.") from exc
        if step_count < 2:
            raise RuntimeError(f"Qwen Image denoise requires at least 2 steps; got {step_count}.")
        try:
            normalized_seed = normalize_torch_manual_seed(int(seed))
        except Exception as exc:  # noqa: BLE001 - strict runtime validation
            raise RuntimeError("Qwen Image denoise seed must be an integer.") from exc
        owner_token = str(progress_owner_token or "").strip()
        if not owner_token:
            raise RuntimeError("Qwen Image denoise requires a non-empty progress owner token.")
        try:
            cfg_scale = float(true_cfg_scale)
        except Exception as exc:  # noqa: BLE001 - strict runtime validation
            raise RuntimeError("Qwen Image true CFG scale must be numeric.") from exc
        if not math.isfinite(cfg_scale) or cfg_scale <= 0.0:
            raise RuntimeError(f"Qwen Image true CFG scale must be finite and positive; got {cfg_scale!r}.")

        grid = qwen_image_latent_grid(width, height)
        compute_device = memory_management.manager.get_device(DeviceRole.CORE)
        if compute_device.type != "cuda":
            raise RuntimeError(
                "Qwen Image Edit-2511 denoise requires the CORE role on CUDA; "
                f"got {compute_device}."
            )
        compute_dtype = getattr(self.transformer, "computation_dtype", None)
        if compute_dtype is not torch.bfloat16:
            raise RuntimeError(
                "Qwen Image Edit-2511 transformer computation dtype must be torch.bfloat16; "
                f"got {compute_dtype!r}."
            )

        active_noise_settings = noise_settings if noise_settings is not None else NoiseSettings()
        if not isinstance(active_noise_settings, NoiseSettings):
            raise TypeError(
                "Qwen Image noise_settings must be NoiseSettings or None; "
                f"got {type(active_noise_settings).__name__}."
            )

        memory_budget = self._denoise_memory_budget(
            conditioning=conditioning,
            reference=reference,
            grid=grid,
            cfg_scale=cfg_scale,
            compute_dtype=compute_dtype,
        )
        logger.info(
            "Qwen Image denoise memory admission: image_tokens=%d active_text_tokens=%d "
            "working_sequence_tokens=%d working_set_bytes=%d persistent_input_bytes=%d "
            "memory_required=%d hard_memory_preservation=%d",
            memory_budget.image_tokens,
            memory_budget.active_text_tokens,
            memory_budget.working_sequence_tokens,
            memory_budget.working_set_bytes,
            memory_budget.persistent_input_bytes,
            memory_budget.memory_required,
            memory_budget.hard_memory_preservation,
        )

        try:
            with self._component_stage_lease(
                stage=_DENOISE_STAGE,
                forbidden=(("text_encoder", self.text_encoder_patcher), ("vae", self.vae.patcher)),
            ):
                with _managed_component_stage(
                    self.denoiser,
                    source=_DENOISE_SOURCE,
                    stage=_DENOISE_STAGE,
                    component_hint=_DENOISE_COMPONENT,
                    memory_required=memory_budget.memory_required,
                    hard_memory_preservation=memory_budget.hard_memory_preservation,
                ):
                    self._require_streamed_phase(StreamedResidencyPhase.READY, stage=_DENOISE_STAGE)
                    packed_cpu = self._run_loaded_denoise(
                        conditioning=conditioning,
                        reference=reference,
                        grid=grid,
                        step_count=step_count,
                        normalized_seed=normalized_seed,
                        owner_token=owner_token,
                        cfg_scale=cfg_scale,
                        noise_settings=active_noise_settings,
                        compute_device=compute_device,
                        compute_dtype=compute_dtype,
                    )
                self._require_streamed_phase(StreamedResidencyPhase.OFFLOADED, stage=_DENOISE_STAGE)
        except Exception as exc:
            raise RuntimeError(
                f"Qwen Image stage {_DENOISE_STAGE!r} component {_DENOISE_COMPONENT!r} failed: {exc}"
            ) from exc

        return QwenImageDenoisedLatents(
            packed_latents=packed_cpu,
            grid=grid,
            seed=normalized_seed,
            steps=step_count,
        )

    @torch.inference_mode()
    def decode(self, *, denoised: QwenImageDenoisedLatents) -> torch.Tensor:
        if not isinstance(denoised, QwenImageDenoisedLatents):
            raise TypeError(
                f"Qwen Image decode input must be QwenImageDenoisedLatents; got {type(denoised).__name__}."
            )
        unpacked_5d = qwen_image_unpack_latents(denoised.packed_latents, denoised.grid)
        if int(unpacked_5d.shape[2]) != 1:
            raise RuntimeError(
                "Qwen Image decode requires one explicit temporal latent frame; "
                f"got {int(unpacked_5d.shape[2])}."
            )
        unpacked_4d = unpacked_5d[:, :, 0].contiguous()
        denormalized = qwen_image_denormalize_latents(unpacked_4d, self.vae_config)

        try:
            with (
                self._component_stage_lease(
                    stage=_DECODE_STAGE,
                    forbidden=(("text_encoder", self.text_encoder_patcher), ("denoiser", self.denoiser)),
                ),
                _managed_component_stage(
                    self.vae.patcher,
                    source=_DECODE_SOURCE,
                    stage=_DECODE_STAGE,
                    component_hint=_DECODE_COMPONENT,
                ),
            ):
                decoded = self.vae.decode(denormalized)
                expected_shape = (1, 3, int(denoised.grid.height), int(denoised.grid.width))
                if tuple(decoded.shape) != expected_shape:
                    raise RuntimeError(
                        "Qwen Image decoded image shape mismatch: "
                        f"got={tuple(decoded.shape)} expected={expected_shape}."
                    )
                if not bool(torch.isfinite(decoded).all().item()):
                    raise RuntimeError("Qwen Image VAE decode produced non-finite pixels.")
                decoded_cpu = decoded.detach().to(device=memory_management.manager.cpu_device).contiguous()
        except Exception as exc:
            raise RuntimeError(
                f"Qwen Image stage {_DECODE_STAGE!r} component {_DECODE_COMPONENT!r} failed: {exc}"
            ) from exc

        return decoded_cpu


__all__ = [
    "QwenImageRuntime",
]

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from modules import devices, sd_samplers
from modules import sd_models
from modules.sd_models import apply_token_merging, SkipWritingToConfig
from modules.sd_samplers_common import (
    approximation_indexes,
    decode_first_stage,
    images_tensor_to_samples,
)
from modules.shared import opts, shared
from modules_forge import main_entry


def _decode_latent_batch(model, batch, target_device=None) -> torch.Tensor:
    """Mirror modules.processing.decode_latent_batch without importing the module."""
    decoded = decode_first_stage(model, batch)
    if target_device is not None:
        decoded = decoded.to(target_device)
    return decoded


def _prepare_first_pass_from_image(processing) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    image = processing.firstpass_image
    if image is None or not processing.enable_hr:
        return None, None

    if processing.latent_scale_mode is None:
        array = np.array(image).astype(np.float32) / 255.0
        array = array * 2.0 - 1.0
        array = np.moveaxis(array, 2, 0)
        decoded_samples = torch.from_numpy(np.expand_dims(array, 0))
        return None, decoded_samples

    array = np.array(image).astype(np.float32) / 255.0
    array = np.moveaxis(array, 2, 0)
    tensor = torch.from_numpy(np.expand_dims(array, axis=0))
    tensor = tensor.to(shared.device, dtype=torch.float32)

    if opts.sd_vae_encode_method != "Full":
        processing.extra_generation_params["VAE Encoder"] = opts.sd_vae_encode_method

    samples = images_tensor_to_samples(
        tensor,
        approximation_indexes.get(opts.sd_vae_encode_method),
        processing.sd_model,
    )
    devices.torch_gc()
    return samples, None


def _reload_for_hires(processing) -> None:
    with SkipWritingToConfig():
        checkpoint_before = getattr(opts, "sd_model_checkpoint")
        modules_before = getattr(opts, "forge_additional_modules")

        reload_required = False
        if (
            getattr(processing, "hr_additional_modules", None) is not None
            and "Use same choices" not in processing.hr_additional_modules
        ):
            modules_changed = main_entry.modules_change(
                processing.hr_additional_modules, save=False, refresh=False
            )
            reload_required = reload_required or modules_changed

        if (
            processing.hr_checkpoint_name
            and processing.hr_checkpoint_name != "Use same checkpoint"
        ):
            checkpoint_changed = main_entry.checkpoint_change(
                processing.hr_checkpoint_name, save=False, refresh=False
            )
            if checkpoint_changed:
                processing.firstpass_use_distilled_cfg_scale = (
                    processing.sd_model.use_distilled_cfg_scale
                )
                reload_required = True

        if reload_required:
            try:
                main_entry.refresh_model_loading_parameters()
                sd_models.forge_model_reload()
            finally:
                main_entry.modules_change(modules_before, save=False, refresh=False)
                main_entry.checkpoint_change(checkpoint_before, save=False, refresh=False)
                main_entry.refresh_model_loading_parameters()

        if processing.sd_model.use_distilled_cfg_scale:
            processing.extra_generation_params["Hires Distilled CFG Scale"] = (
                processing.hr_distilled_cfg
            )


class Txt2ImgRuntime:
    """Encapsulates txt2img sampling so that the orchestration can be tested in isolation."""

    def __init__(
        self,
        processing,
        conditioning,
        unconditional_conditioning,
        seeds: Sequence[int],
        subseeds: Sequence[int],
        subseed_strength: float,
        prompts: Sequence[str],
    ) -> None:
        self.processing = processing
        self.conditioning = conditioning
        self.unconditional_conditioning = unconditional_conditioning
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.prompts = prompts

    def generate(self):
        self._ensure_sampler()

        samples, decoded_samples = _prepare_first_pass_from_image(self.processing)

        if samples is None and decoded_samples is None:
            samples = self._run_base_sampling()
            decoded_samples = self._maybe_decode_for_hr(samples)

        if not self.processing.enable_hr:
            return samples

        _reload_for_hires(self.processing)

        return self.processing.sample_hr_pass(
            samples,
            decoded_samples,
            self.seeds,
            self.subseeds,
            self.subseed_strength,
            self.prompts,
        )

    def _ensure_sampler(self) -> None:
        self.processing.sampler = sd_samplers.create_sampler(
            self.processing.sampler_name, self.processing.sd_model
        )

    def _run_base_sampling(self):
        noise = self.processing.rng.next()

        self.processing.sd_model.forge_objects = (
            self.processing.sd_model.forge_objects_after_applying_lora.shallow_copy()
        )
        apply_token_merging(
            self.processing.sd_model, self.processing.get_token_merging_ratio()
        )

        if self.processing.scripts is not None:
            self.processing.scripts.process_before_every_sampling(
                self.processing,
                x=noise,
                noise=noise,
                c=self.conditioning,
                uc=self.unconditional_conditioning,
            )

        if self.processing.modified_noise is not None:
            noise = self.processing.modified_noise
            self.processing.modified_noise = None

        samples = self.processing.sampler.sample(
            self.processing,
            noise,
            self.conditioning,
            self.unconditional_conditioning,
            image_conditioning=self.processing.txt2img_image_conditioning(noise),
        )

        del noise
        return samples

    def _maybe_decode_for_hr(self, samples):
        if not self.processing.enable_hr:
            return None

        devices.torch_gc()

        if self.processing.latent_scale_mode is None:
            return _decode_latent_batch(
                self.processing.sd_model, samples, target_device=devices.cpu
            ).to(dtype=torch.float32)

        return None


def generate_txt2img(
    processing,
    conditioning,
    unconditional_conditioning,
    seeds: Sequence[int],
    subseeds: Sequence[int],
    subseed_strength: float,
    prompts: Sequence[str],
):
    runtime = Txt2ImgRuntime(
        processing,
        conditioning,
        unconditional_conditioning,
        seeds,
        subseeds,
        subseed_strength,
        prompts,
    )

    return runtime.generate()

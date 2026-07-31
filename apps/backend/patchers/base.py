"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Core patcher primitives (model options hooks + LoRA patch registry + object patching + full/streamed ModelPatcher residency bridge).
Provides the shared patch/registry structures that engines use to apply LoRA patches, inject CFG/UNet/VAE wrappers, and manage
device placement, explicit streamed-core residency, and smart offload interactions.
Host-memory pinning during smart-offload CPU unpatch now also emits canonical INFO audit events via `backend.smart_offload`.
That audit event is tagged through the canonical `SmartOffloadAction.PIN_HOST_MEMORY` enum action.

Symbols (top-level; keep in sync; no ghosts):
- `LoraPatchRegistry` (dataclass): Tracks LoRA patch bundles keyed by filename/strength (clone/merge helpers; supports “online” LoRA mode).
- `ObjectPatchRegistry` (dataclass): Tracks object attribute patches and backups (register/apply/restore against a model object).
- `_ensure_transformer_options` (function): Ensures model options include the nested transformer-options mapping.
- `_copy_transformer_options` (function): Copies transformer options mapping to avoid shared-mutation across clones.
- `set_model_options_patch_replace` (function): Adds a “patch replace” entry into model options (block/name/indices keyed).
- `_coerce_patch_replace_block` (function): Validates and normalizes a patch-replace block key tuple before registration.
- `set_model_options_post_cfg_function` (function): Registers a post-CFG callback in model options.
- `set_model_options_pre_cfg_function` (function): Registers a pre-CFG callback in model options.
- `ModelPatcher` (class): Main patcher wrapper around a model; owns LoRA/object patch registries, optional typed streamed-core
  runtime registration, wrapper/patch hooks, and full-versus-streamed apply/unapply integration with the memory manager.
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import copy
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

import torch

from apps.backend.runtime import trace as _trace
from apps.backend.runtime import utils
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.streamed_residency import (
    ResidencyMode,
    StreamedCoreRuntime,
    StreamedFootprint,
    StreamedResidencyPhase,
)
from apps.backend.runtime.memory.smart_offload import (
    SmartOffloadAction,
    log_smart_offload_action,
    smart_offload_enabled,
)
from .lora import CodexLoraLoader

logger = get_backend_logger("backend.patchers.base")

LoraPatchKey = Tuple[str, float, float, bool]
PatchEntry = List[Any]


@dataclass
class LoraPatchRegistry:
    """Tracks LoRA patch bundles keyed by filename/strength."""

    data: Dict[LoraPatchKey, Dict[str, List[PatchEntry]]] = field(default_factory=dict)

    def clone(self) -> "LoraPatchRegistry":
        return LoraPatchRegistry(copy.deepcopy(self.data))

    def add_patch(
        self,
        key: LoraPatchKey,
        parameter: str,
        strength_patch: float,
        tensor: Any,
        strength_model: float,
        offset: Any,
        function: Any,
    ) -> None:
        bucket = self.data.setdefault(key, {})
        entries = bucket.setdefault(parameter, [])
        entry = [strength_patch, tensor, strength_model, offset, function]
        entries.append(entry)
        logger.debug(
            "Registered LoRA patch bundle=%s parameter=%s entries=%d",
            key[0],
            parameter,
            len(entries),
        )

    def has_online_lora(self) -> bool:
        return any(key[3] for key in self.data)

    @classmethod
    def from_mapping(cls, mapping: MutableMapping[LoraPatchKey, Dict[str, List[PatchEntry]]]) -> "LoraPatchRegistry":
        registry = cls()
        for key, target_map in mapping.items():
            if not isinstance(key, tuple) or len(key) != 4:
                raise ValueError(f"Invalid LoRA patch key {key!r}")
            filename, strength_patch, strength_model, online_mode = key
            for parameter, entries in target_map.items():
                if not isinstance(entries, Iterable):
                    raise TypeError("LoRA patch entries must be iterable")
                registry.data.setdefault(key, {})[parameter] = [list(entry) for entry in entries]
        return registry


@dataclass
class ObjectPatchRegistry:
    """Maintains object attribute patches and their backups."""

    active: Dict[str, Any] = field(default_factory=dict)
    backup: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "ObjectPatchRegistry":
        return ObjectPatchRegistry(active=self.active.copy(), backup=self.backup.copy())

    def register(self, name: str, value: Any) -> None:
        logger.debug("Registered object patch %s", name)
        self.active[name] = value

    def get(self, name: str, model: Any) -> Any:
        if name in self.active:
            return self.active[name]
        if name in self.backup:
            return self.backup[name]
        return utils.get_attr(model, name)

    def apply_to_model(self, model: Any) -> None:
        for name, value in self.active.items():
            previous = utils.get_attr(model, name)
            if name not in self.backup:
                self.backup[name] = previous
            utils.set_attr_raw(model, name, value)
            logger.debug("Applied object patch %s", name)

    def restore(self, model: Any) -> None:
        for name, original in self.backup.items():
            utils.set_attr_raw(model, name, original)
            logger.debug("Restored object patch %s", name)
        self.backup.clear()


def _ensure_transformer_options(model_options: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    transformer_options = model_options.setdefault("transformer_options", {})
    if not isinstance(transformer_options, MutableMapping):
        raise TypeError("transformer_options must be a mapping")
    return transformer_options


def _copy_transformer_options(model_options: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    transformer_options = _ensure_transformer_options(model_options)
    copied = transformer_options.copy()
    model_options["transformer_options"] = copied
    return copied


def set_model_options_patch_replace(
    model_options: MutableMapping[str, Any],
    patch: Any,
    name: str,
    block_name: str,
    number: int,
    transformer_index: Optional[int] = None,
) -> MutableMapping[str, Any]:
    transformer_options = _copy_transformer_options(model_options)

    patches_replace = transformer_options.get("patches_replace")
    if patches_replace is None:
        patches_replace = {}
    else:
        patches_replace = patches_replace.copy()

    target = patches_replace.get(name)
    if target is None:
        target = {}
    else:
        target = target.copy()

    if transformer_index is not None:
        block = (block_name, number, transformer_index)
    else:
        block = (block_name, number)

    target[block] = patch
    patches_replace[name] = target
    transformer_options["patches_replace"] = patches_replace

    logger.debug(
        "Registered transformer replace patch name=%s block=%s number=%s transformer_index=%s",
        name,
        block_name,
        number,
        transformer_index,
    )
    return model_options


def _coerce_patch_replace_block(block: Any) -> tuple[tuple[str, int] | tuple[str, int, int], str, int, Optional[int]]:
    if not isinstance(block, tuple):
        raise TypeError("Patch-replace block key must be a tuple.")
    if len(block) == 2:
        block_name, number = block
        transformer_index = None
    elif len(block) == 3:
        block_name, number, transformer_index = block
    else:
        raise TypeError("Patch-replace block key must have length 2 or 3.")
    if not isinstance(block_name, str):
        raise TypeError("Patch-replace block name must be a string.")
    if not isinstance(number, int):
        raise TypeError("Patch-replace block index must be an int.")
    if transformer_index is not None and not isinstance(transformer_index, int):
        raise TypeError("Patch-replace transformer_index must be an int when provided.")
    normalized = (block_name, number) if transformer_index is None else (block_name, number, transformer_index)
    return normalized, block_name, number, transformer_index


def set_model_options_post_cfg_function(
    model_options: MutableMapping[str, Any],
    post_cfg_function: Any,
    disable_cfg1_optimization: bool = False,
) -> MutableMapping[str, Any]:
    funcs = list(model_options.get("sampler_post_cfg_function", []))
    funcs.append(post_cfg_function)
    model_options["sampler_post_cfg_function"] = funcs
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    logger.debug("Registered sampler post-cfg function %s", getattr(post_cfg_function, "__name__", post_cfg_function))
    return model_options


def set_model_options_pre_cfg_function(
    model_options: MutableMapping[str, Any],
    pre_cfg_function: Any,
    disable_cfg1_optimization: bool = False,
) -> MutableMapping[str, Any]:
    funcs = list(model_options.get("sampler_pre_cfg_function", []))
    funcs.append(pre_cfg_function)
    model_options["sampler_pre_cfg_function"] = funcs
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    logger.debug("Registered sampler pre-cfg function %s", getattr(pre_cfg_function, "__name__", pre_cfg_function))
    return model_options


class ModelPatcher:
    """Codex-native model patcher with typed registries and lifecycle hooks."""

    def __init__(self, model, load_device, offload_device, size=0, current_device=None, **kwargs):
        self.size = size
        self.model = model
        self._lora_registry = LoraPatchRegistry()
        self._object_registry = ObjectPatchRegistry()
        self._model_options: Dict[str, Any] = {"transformer_options": {}}
        self._streamed_core_runtime: StreamedCoreRuntime | None = None
        self.patches: Dict[str, List[Any]] = {}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device

        if not hasattr(model, "lora_loader") or not isinstance(model.lora_loader, CodexLoraLoader):
            model.lora_loader = CodexLoraLoader(model)

        self.lora_loader: CodexLoraLoader = model.lora_loader

        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

    @property
    def lora_patches(self) -> Dict[LoraPatchKey, Dict[str, List[PatchEntry]]]:
        return self._lora_registry.data

    @lora_patches.setter
    def lora_patches(self, value: Dict[LoraPatchKey, Dict[str, List[PatchEntry]]]) -> None:
        self._lora_registry = LoraPatchRegistry.from_mapping(value)

    @property
    def object_patches(self) -> Dict[str, Any]:
        return self._object_registry.active

    @object_patches.setter
    def object_patches(self, value: Dict[str, Any]) -> None:
        self._object_registry = ObjectPatchRegistry(active=value.copy(), backup={})

    @property
    def object_patches_backup(self) -> Dict[str, Any]:
        return self._object_registry.backup

    @object_patches_backup.setter
    def object_patches_backup(self, value: Dict[str, Any]) -> None:
        self._object_registry.backup = value.copy()

    @property
    def model_options(self) -> Dict[str, Any]:
        return self._model_options

    @model_options.setter
    def model_options(self, value: Dict[str, Any]) -> None:
        if "transformer_options" not in value:
            value = {**value, "transformer_options": {}}
        self._model_options = copy.deepcopy(value)

    @property
    def streamed_core_runtime(self) -> StreamedCoreRuntime | None:
        return self._streamed_core_runtime

    @property
    def residency_mode(self) -> ResidencyMode:
        if self._streamed_core_runtime is None:
            return ResidencyMode.FULL
        return ResidencyMode.STREAMED

    def set_streamed_core_runtime(self, runtime: StreamedCoreRuntime) -> None:
        if not isinstance(runtime, StreamedCoreRuntime):
            raise TypeError(
                "streamed core runtime must implement StreamedCoreRuntime, "
                f"got {type(runtime).__name__}."
            )
        footprint = runtime.footprint
        if not isinstance(footprint, StreamedFootprint):
            raise TypeError(
                "streamed core runtime footprint must be StreamedFootprint, "
                f"got {type(footprint).__name__}."
            )
        if self._streamed_core_runtime is not None and self._streamed_core_runtime is not runtime:
            raise RuntimeError("ModelPatcher already owns a different streamed core runtime.")
        self._streamed_core_runtime = runtime

    def model_size(self):
        if self.size > 0:
            return self.size
        self.size = memory_management.manager.module_size(self.model)
        return self.size

    def clone(self):
        clone = self.__class__(self.model, self.load_device, self.offload_device, self.size, self.current_device)
        clone._lora_registry = self._lora_registry.clone()
        clone._object_registry = self._object_registry.clone()
        clone._model_options = copy.deepcopy(self._model_options)
        clone.lora_loader = self.lora_loader
        clone.patches = copy.deepcopy(self.patches)
        if self._streamed_core_runtime is not None:
            clone.set_streamed_core_runtime(self._streamed_core_runtime)
        logger.debug("Cloned ModelPatcher %s -> %s", id(self), id(clone))
        return clone

    def is_clone(self, other):
        if hasattr(other, "model") and self.model is other.model:
            return True
        return False

    def add_patches(
        self,
        *,
        filename: str,
        patches: MutableMapping[str, Any],
        strength_patch: float = 1.0,
        strength_model: float = 1.0,
        online_mode: bool = False,
    ) -> set:
        lora_identifier: LoraPatchKey = (filename, strength_patch, strength_model, online_mode)
        matched = set()
        model_keys = set(k for k, _ in self.model.named_parameters())

        for key in patches:
            offset = None
            function = None

            if isinstance(key, str):
                parameter = key
            else:
                parameter = key[0]
                offset = key[1]
                if len(key) > 2:
                    function = key[2]

            if parameter in model_keys:
                matched.add(key)
                tensor = patches[key]
                self._lora_registry.add_patch(
                    lora_identifier,
                    parameter,
                    strength_patch,
                    tensor,
                    strength_model,
                    offset,
                    function,
                )

        logger.debug(
            "Registered %d LoRA patches from %s (online=%s)",
            len(matched),
            filename,
            online_mode,
        )
        return matched

    def has_online_lora(self):
        return self._lora_registry.has_online_lora()

    def refresh_loras(self):
        logger.debug("Refreshing LoRA loader with %d bundles", len(self._lora_registry.data))
        self.lora_loader.refresh(lora_patches=self.lora_patches, offload_device=self.offload_device)
        return

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(
                args["cond"],
                args["uncond"],
                args["cond_scale"],
            )
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True
        logger.debug("Registered sampler cfg function %s", getattr(sampler_cfg_function, "__name__", sampler_cfg_function))

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_post_cfg_function(
            self.model_options,
            post_cfg_function,
            disable_cfg1_optimization,
        )

    def set_model_sampler_pre_cfg_function(self, pre_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_pre_cfg_function(
            self.model_options,
            pre_cfg_function,
            disable_cfg1_optimization,
        )

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        self.model_options["model_function_wrapper"] = unet_wrapper_function
        logger.debug("Registered model_function_wrapper %s", unet_wrapper_function)

    def set_model_vae_encode_wrapper(self, wrapper_function):
        self.model_options["model_vae_encode_wrapper"] = wrapper_function
        logger.debug("Registered model_vae_encode_wrapper %s", wrapper_function)

    def set_model_vae_decode_wrapper(self, wrapper_function):
        self.model_options["model_vae_decode_wrapper"] = wrapper_function
        logger.debug("Registered model_vae_decode_wrapper %s", wrapper_function)

    def set_model_vae_regulation(self, vae_regulation):
        self.model_options["model_vae_regulation"] = vae_regulation
        logger.debug("Registered model_vae_regulation %s", vae_regulation)

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function
        logger.debug("Registered denoise_mask_function %s", denoise_mask_function)

    def set_model_patch(self, patch, name):
        transformer_options = _ensure_transformer_options(self.model_options)
        patches = transformer_options.setdefault("patches", {})
        bucket = patches.setdefault(name, [])
        bucket.append(patch)
        logger.debug("Registered transformer patch %s (size=%d)", name, len(bucket))

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = set_model_options_patch_replace(
            self.model_options,
            patch,
            name,
            block_name,
            number,
            transformer_index=transformer_index,
        )

    def set_model_patch_replace_many(self, name: str, patches: MutableMapping[tuple[Any, ...], Any]) -> None:
        transformer_options = _ensure_transformer_options(self._model_options)
        patches_replace = transformer_options.setdefault("patches_replace", {})
        if not isinstance(patches_replace, MutableMapping):
            raise TypeError("transformer_options['patches_replace'] must be a mapping")
        target = patches_replace.get(name)
        if target is None:
            target = {}
            patches_replace[name] = target
        elif not isinstance(target, MutableMapping):
            raise TypeError(f"transformer_options['patches_replace']['{name}'] must be a mapping")
        for raw_block, patch in patches.items():
            block, block_name, number, transformer_index = _coerce_patch_replace_block(raw_block)
            target[block] = patch
            logger.debug(
                "Registered transformer replace patch name=%s block=%s number=%s transformer_index=%s",
                name,
                block_name,
                number,
                transformer_index,
            )

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_replace_many(self, patches: MutableMapping[tuple[Any, ...], Any]) -> None:
        self.set_model_patch_replace_many("attn1", patches)

    def set_model_attn2_replace_many(self, patches: MutableMapping[tuple[Any, ...], Any]) -> None:
        self.set_model_patch_replace_many("attn2", patches)

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        self.set_model_patch(patch, "output_block_patch")

    def add_object_patch(self, name, obj):
        self._object_registry.register(name, obj)

    def get_model_object(self, name):
        return self._object_registry.get(name, self.model)

    def model_patches_to(self, device):
        transformer_options = self.model_options.get("transformer_options", {})
        patches = transformer_options.get("patches", {})
        for name, patch_list in patches.items():
            for idx, patch in enumerate(list(patch_list)):
                if hasattr(patch, "to"):
                    _trace.event("patch_to", name=name, idx=idx, device=str(device))
                    patch_list[idx] = patch.to(device)
                    logger.debug("Moved patch %s[%d] to %s", name, idx, device)
        patches_replace = transformer_options.get("patches_replace", {})
        for name, patch_map in patches_replace.items():
            for key, patch in list(patch_map.items()):
                if hasattr(patch, "to"):
                    _trace.event("patch_replace_to", name=name, key=str(key), device=str(device))
                    patch_map[key] = patch.to(device)
                    logger.debug("Moved replace patch %s[%s] to %s", name, key, device)
        wrapper = self.model_options.get("model_function_wrapper")
        if hasattr(wrapper, "to"):
            _trace.event("wrapper_to", device=str(device))
            self.model_options["model_function_wrapper"] = wrapper.to(device)
            logger.debug("Moved model_function_wrapper to %s", device)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def get_key_patches(self, filter_prefix=None):
        memory_management.manager.unload_model_clones(self)
        model_sd = self.model_state_dict()
        patches_dict = self.patches
        p = {}
        for key, value in model_sd.items():
            if filter_prefix is not None and not key.startswith(filter_prefix):
                continue
            if key in patches_dict:
                p[key] = [value] + patches_dict[key]
            else:
                p[key] = (value,)
        return p

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def codex_patch_model(self, target_device=None):
        self._object_registry.apply_to_model(self.model)

        if target_device is not None:
            if self._streamed_core_runtime is not None:
                storage_device = torch.device(self.offload_device)
                compute_device = torch.device(target_device)
                snapshot = self._streamed_core_runtime.prepare_streamed_residency(
                    storage_device=storage_device,
                    compute_device=compute_device,
                )
                if snapshot.phase is not StreamedResidencyPhase.READY:
                    raise RuntimeError(
                        "Streamed core prepare must finish in READY phase, "
                        f"got {snapshot.phase.value}."
                    )
                if snapshot.storage_device != storage_device:
                    raise RuntimeError(
                        "Streamed core prepare reported the wrong storage device: "
                        f"expected={storage_device}, observed={snapshot.storage_device}."
                    )
                if snapshot.compute_device != compute_device:
                    raise RuntimeError(
                        "Streamed core prepare reported the wrong compute device: "
                        f"expected={compute_device}, observed={snapshot.compute_device}."
                    )
                self.current_device = snapshot.storage_device
                logger.debug(
                    "Prepared streamed model residency storage=%s compute=%s current_device_bytes=%d",
                    snapshot.storage_device,
                    snapshot.compute_device,
                    snapshot.current_device_bytes,
                )
                return self.model
            try:
                # Prefer non_blocking=True to leverage pinned host buffers
                self.model.to(target_device, non_blocking=True)
            except TypeError:
                self.model.to(target_device)
            self.current_device = target_device
            logger.debug("Moved model to device %s during patch", target_device)

        return self.model

    def codex_unpatch_model(self, target_device=None):
        if self._streamed_core_runtime is not None:
            self._streamed_core_runtime.offload_all()
            snapshot = self._streamed_core_runtime.verify_residency()
            if snapshot.phase is not StreamedResidencyPhase.OFFLOADED:
                raise RuntimeError(
                    "Streamed core unpatch must finish in OFFLOADED phase, "
                    f"got {snapshot.phase.value}."
                )
            if target_device is not None and snapshot.storage_device != torch.device(target_device):
                raise RuntimeError(
                    "Streamed core unpatch reported the wrong storage device: "
                    f"expected={torch.device(target_device)}, observed={snapshot.storage_device}."
                )
            self.current_device = snapshot.storage_device
            self._object_registry.restore(self.model)
            logger.debug(
                "Offloaded streamed model residency storage=%s compute=%s",
                snapshot.storage_device,
                snapshot.compute_device,
            )
            return

        if target_device is not None:
            self.model.to(target_device)
            self.current_device = target_device
            logger.debug("Moved model to device %s during unpatch", target_device)

        # If we're offloading to CPU under smart-offload, pin host memory buffers
        should_pin = False
        if target_device is not None and getattr(target_device, "type", "") == "cpu" and smart_offload_enabled():
            cfg = memory_management.manager.config
            should_pin = bool(cfg.swap.pin_shared_memory)

        if should_pin:
            pinned_params = 0
            for p in self.model.parameters(recurse=True):
                try:
                    if not p.is_cuda and not p.data.is_pinned():
                        p.data = p.data.pin_memory()
                        pinned_params += 1
                except Exception:
                    continue
            pinned_bufs = 0
            for b in self.model.buffers(recurse=True):
                try:
                    if not b.is_cuda and not b.data.is_pinned():
                        b.data = b.data.pin_memory()
                        pinned_bufs += 1
                except Exception:
                    continue
            if pinned_params or pinned_bufs:
                logger.info(
                    "Pinned host memory for offloaded model params=%d buffers=%d (dtype preserved)",
                    pinned_params,
                    pinned_bufs,
                )
                log_smart_offload_action(
                    SmartOffloadAction.PIN_HOST_MEMORY,
                    source="patchers.base.codex_unpatch_model",
                    component=type(self.model).__name__,
                    to_device=str(target_device),
                    pinned_params=pinned_params,
                    pinned_buffers=pinned_bufs,
                )

        self._object_registry.restore(self.model)
        return

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Exact tensor-slot inventory and physical ownership for Qwen Image Edit-2511 streamed residency.
Owns deterministic direct parameter/buffer traversal, complete static/block coverage, packed/logical byte accounting,
physical-overlap rejection, exact host storage identity checks, and bounded staged tensor construction.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageDirectSlot` (dataclass): One exact direct parameter or buffer location with immutable host identity.
- `QwenImageResidencyOwner` (dataclass): One static or transformer-block slot/byte ownership set.
- `QwenImageSlotInventory` (class): Complete 60-block transformer inventory, alias validation, and staging primitive owner.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

from apps.backend.quantization.tensor import CodexParameter


_EXPECTED_BLOCK_COUNT = 60
_STATIC_OWNER_NAMES = (
    "pos_embed",
    "time_text_embed",
    "txt_norm",
    "img_in",
    "txt_in",
    "norm_out",
    "proj_out",
)


class _SlotKind(str, Enum):
    PARAMETER = "parameter"
    BUFFER = "buffer"


@dataclass(frozen=True, slots=True)
class _TensorInterval:
    device: torch.device
    storage_base_pointer: int
    storage_offset_bytes: int
    interval_end_pointer: int


@dataclass(frozen=True, slots=True)
class QwenImageDirectSlot:
    module: nn.Module
    kind: _SlotKind
    name: str
    path: str
    host_tensor: torch.Tensor
    packed_bytes: int
    interval: _TensorInterval

    @property
    def location(self) -> tuple[int, _SlotKind, str]:
        return (id(self.module), self.kind, self.name)

    def current_tensor(self) -> torch.Tensor:
        slots = self.module._parameters if self.kind is _SlotKind.PARAMETER else self.module._buffers
        current = slots.get(self.name)
        if not isinstance(current, torch.Tensor):
            raise RuntimeError(f"Qwen Image streamed slot {self.path} is missing a live tensor.")
        return current

    def install(self, tensor: torch.Tensor) -> None:
        if self.kind is _SlotKind.PARAMETER and not isinstance(tensor, nn.Parameter):
            raise TypeError(
                f"Qwen Image streamed parameter slot {self.path} requires nn.Parameter; "
                f"got {type(tensor).__name__}."
            )
        slots = self.module._parameters if self.kind is _SlotKind.PARAMETER else self.module._buffers
        slots[self.name] = tensor

    def verify_host_identity(self, *, storage_device: torch.device) -> None:
        if self.host_tensor.device != storage_device:
            raise RuntimeError(
                f"Qwen Image host slot {self.path} left storage device {storage_device}: "
                f"got {self.host_tensor.device}."
            )
        current_interval = _tensor_interval(self.host_tensor, path=self.path)
        if current_interval != self.interval:
            raise RuntimeError(
                f"Qwen Image host slot {self.path} changed physical storage identity: "
                f"expected={self.interval} got={current_interval}."
            )


@dataclass(frozen=True, slots=True)
class QwenImageResidencyOwner:
    label: str
    segment_index: int | None
    slots: tuple[QwenImageDirectSlot, ...]
    unique_sources: tuple[torch.Tensor, ...]
    packed_bytes: int
    transient_device_bytes: int


def _packed_bytes(tensor: torch.Tensor) -> int:
    raw_tensor = tensor.data if isinstance(tensor, nn.Parameter) else tensor
    return int(raw_tensor.numel()) * int(raw_tensor.element_size())


def _tensor_interval(tensor: torch.Tensor, *, path: str) -> _TensorInterval:
    if not tensor.is_contiguous():
        raise RuntimeError(
            f"Qwen Image streamed slot {path} must be contiguous; "
            f"got shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}."
        )
    try:
        storage_base_pointer = int(tensor.untyped_storage().data_ptr())
    except Exception as exc:  # noqa: BLE001 - fail-loud physical ownership boundary
        raise RuntimeError(
            f"Qwen Image streamed slot {path} does not expose untyped storage identity."
        ) from exc
    storage_offset_bytes = int(tensor.storage_offset()) * int(tensor.element_size())
    return _TensorInterval(
        device=torch.device(tensor.device),
        storage_base_pointer=storage_base_pointer,
        storage_offset_bytes=storage_offset_bytes,
        interval_end_pointer=storage_base_pointer + storage_offset_bytes + _packed_bytes(tensor),
    )


def _walk_direct_slots(module: nn.Module, *, root_path: str) -> tuple[QwenImageDirectSlot, ...]:
    seen_modules: set[int] = set()
    slots: list[QwenImageDirectSlot] = []

    def _visit(current: nn.Module, path: str) -> None:
        module_identity = id(current)
        if module_identity in seen_modules:
            return
        seen_modules.add(module_identity)
        for name, parameter in current._parameters.items():
            if parameter is not None:
                slot_path = f"{path}.{name}"
                slots.append(
                    QwenImageDirectSlot(
                        module=current,
                        kind=_SlotKind.PARAMETER,
                        name=name,
                        path=slot_path,
                        host_tensor=parameter,
                        packed_bytes=_packed_bytes(parameter),
                        interval=_tensor_interval(parameter, path=slot_path),
                    )
                )
        for name, buffer in current._buffers.items():
            if buffer is not None:
                slot_path = f"{path}.{name}"
                slots.append(
                    QwenImageDirectSlot(
                        module=current,
                        kind=_SlotKind.BUFFER,
                        name=name,
                        path=slot_path,
                        host_tensor=buffer,
                        packed_bytes=_packed_bytes(buffer),
                        interval=_tensor_interval(buffer, path=slot_path),
                    )
                )
        for child_name, child in current._modules.items():
            if child is not None:
                _visit(child, f"{path}.{child_name}")

    _visit(module, root_path)
    return tuple(slots)


def _logical_dequant_bytes(tensor: torch.Tensor, *, path: str) -> int:
    if not isinstance(tensor, CodexParameter) or tensor.qtype is None:
        return 0
    real_shape = tuple(int(dimension) for dimension in tensor.real_shape)
    if not real_shape or any(dimension <= 0 for dimension in real_shape):
        raise RuntimeError(
            f"Qwen Image GGUF slot {path} requires a positive logical real_shape; got {real_shape!r}."
        )
    computation_dtype = tensor.computation_dtype
    if not isinstance(computation_dtype, torch.dtype):
        raise TypeError(
            f"Qwen Image GGUF slot {path} requires torch.dtype computation_dtype; "
            f"got {type(computation_dtype).__name__}."
        )
    return int(math.prod(real_shape)) * int(torch.empty((), dtype=computation_dtype).element_size())


def _build_owner(
    *,
    label: str,
    segment_index: int | None,
    owner_roots: Sequence[tuple[str, nn.Module]],
) -> QwenImageResidencyOwner:
    location_slots: dict[tuple[int, _SlotKind, str], QwenImageDirectSlot] = {}
    for path, module in owner_roots:
        for slot in _walk_direct_slots(module, root_path=path):
            location_slots.setdefault(slot.location, slot)
    slots = tuple(location_slots.values())
    if segment_index is not None and not slots:
        raise RuntimeError(f"Qwen Image streamed {label} owns no parameter or buffer slots.")

    unique_sources: dict[int, torch.Tensor] = {}
    transient_by_module: dict[int, dict[int, int]] = {}
    for slot in slots:
        source_identity = id(slot.host_tensor)
        prior = unique_sources.get(source_identity)
        if prior is not None and prior is not slot.host_tensor:
            raise RuntimeError(
                f"Qwen Image streamed {label} encountered a Python identity collision at {slot.path}."
            )
        unique_sources[source_identity] = slot.host_tensor
        transient_by_module.setdefault(id(slot.module), {}).setdefault(
            source_identity,
            _logical_dequant_bytes(slot.host_tensor, path=slot.path),
        )

    return QwenImageResidencyOwner(
        label=label,
        segment_index=segment_index,
        slots=slots,
        unique_sources=tuple(unique_sources.values()),
        packed_bytes=sum(_packed_bytes(source) for source in unique_sources.values()),
        transient_device_bytes=max(
            (sum(values.values()) for values in transient_by_module.values()),
            default=0,
        ),
    )


def _validate_physical_ownership(owners: Sequence[QwenImageResidencyOwner]) -> None:
    object_owners: dict[int, tuple[str, torch.Tensor]] = {}
    interval_groups: dict[torch.device, list[tuple[_TensorInterval, str, torch.Tensor]]] = {}
    for owner in owners:
        first_slot_by_source = {id(slot.host_tensor): slot for slot in owner.slots}
        for source in owner.unique_sources:
            source_identity = id(source)
            prior = object_owners.get(source_identity)
            if prior is not None:
                prior_owner, prior_source = prior
                if prior_source is source and prior_owner != owner.label:
                    raise RuntimeError(
                        "Qwen Image streamed tensor object is owned by multiple residency owners: "
                        f"owners=({prior_owner!r}, {owner.label!r})."
                    )
            else:
                object_owners[source_identity] = (owner.label, source)
            slot = first_slot_by_source[source_identity]
            interval_groups.setdefault(slot.interval.device, []).append(
                (slot.interval, f"{owner.label}:{slot.path}", source)
            )

    for intervals in interval_groups.values():
        intervals.sort(
            key=lambda item: (
                item[0].storage_base_pointer + item[0].storage_offset_bytes,
                item[0].interval_end_pointer,
                item[1],
            )
        )
        previous: tuple[_TensorInterval, str, torch.Tensor] | None = None
        for current in intervals:
            if previous is None:
                previous = current
                continue
            previous_interval, previous_label, previous_tensor = previous
            current_interval, current_label, current_tensor = current
            current_start_pointer = current_interval.storage_base_pointer + current_interval.storage_offset_bytes
            if current_start_pointer < previous_interval.interval_end_pointer and previous_tensor is not current_tensor:
                raise RuntimeError(
                    "Qwen Image streamed tensors have overlapping physical storage intervals: "
                    f"left={previous_label} right={current_label} "
                    f"left_base={previous_interval.storage_base_pointer} "
                    f"left_offset={previous_interval.storage_offset_bytes} "
                    f"left_end={previous_interval.interval_end_pointer} "
                    f"right_base={current_interval.storage_base_pointer} "
                    f"right_offset={current_interval.storage_offset_bytes}."
                )
            if current_interval.interval_end_pointer > previous_interval.interval_end_pointer:
                previous = current


class QwenImageSlotInventory:
    """Complete static/block slot ownership and staging primitive owner."""

    def __init__(self, transformer: nn.Module, *, storage_device: torch.device) -> None:
        if not isinstance(transformer, nn.Module):
            raise TypeError(
                "Qwen Image slot inventory requires an nn.Module transformer; "
                f"got {type(transformer).__name__}."
            )
        self.transformer = transformer
        self.storage_device = torch.device(storage_device)
        if self.storage_device.type != "cpu":
            raise RuntimeError(
                f"Qwen Image slot inventory requires CPU storage; got {self.storage_device}."
            )
        transformer_blocks = getattr(transformer, "transformer_blocks", None)
        if not isinstance(transformer_blocks, nn.ModuleList):
            raise TypeError(
                "Qwen Image slot inventory requires transformer.transformer_blocks as nn.ModuleList."
            )
        if len(transformer_blocks) != _EXPECTED_BLOCK_COUNT:
            raise RuntimeError(
                "Qwen Image slot inventory requires exactly 60 transformer blocks; "
                f"got {len(transformer_blocks)}."
            )

        static_roots: list[tuple[str, nn.Module]] = []
        for owner_name in _STATIC_OWNER_NAMES:
            owner_module = getattr(transformer, owner_name, None)
            if not isinstance(owner_module, nn.Module):
                raise TypeError(
                    f"Qwen Image streamed static owner transformer.{owner_name} must be nn.Module; "
                    f"got {type(owner_module).__name__}."
                )
            static_roots.append((f"transformer.{owner_name}", owner_module))
        self.static_owner = _build_owner(
            label="static",
            segment_index=None,
            owner_roots=tuple(static_roots),
        )
        self.block_owners = tuple(
            _build_owner(
                label=f"block[{block_index}]",
                segment_index=block_index,
                owner_roots=((f"transformer.transformer_blocks.{block_index}", block),),
            )
            for block_index, block in enumerate(transformer_blocks)
        )
        self.all_owners = (self.static_owner, *self.block_owners)
        self.validate_current_topology()
        _validate_physical_ownership(self.all_owners)

    def validate_current_topology(self) -> None:
        root_slots = {
            slot.location: slot
            for slot in _walk_direct_slots(self.transformer, root_path="transformer")
        }
        owned_locations: dict[tuple[int, _SlotKind, str], str] = {}
        for owner in self.all_owners:
            for slot in owner.slots:
                prior_owner = owned_locations.get(slot.location)
                if prior_owner is not None and prior_owner != owner.label:
                    raise RuntimeError(
                        "Qwen Image streamed direct slot has multiple owners: "
                        f"slot={slot.path} owners=({prior_owner!r}, {owner.label!r})."
                    )
                owned_locations[slot.location] = owner.label
                slot.verify_host_identity(storage_device=self.storage_device)
        root_locations = set(root_slots)
        owner_locations = set(owned_locations)
        if root_locations != owner_locations:
            missing = [root_slots[key].path for key in root_locations - owner_locations]
            extra = [str(key) for key in owner_locations - root_locations]
            raise RuntimeError(
                "Qwen Image streamed owner coverage must equal the complete transformer direct-slot traversal: "
                f"unowned={missing[:20]} extra={extra[:20]} "
                f"root_slots={len(root_locations)} owned_slots={len(owner_locations)}."
            )

    @staticmethod
    def stage_tensor(tensor: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        if isinstance(tensor, CodexParameter):
            return tensor.to(device=device, non_blocking=False, copy=tensor.device == device)
        if isinstance(tensor, nn.Parameter):
            moved = tensor.detach().to(
                device=device,
                non_blocking=False,
                copy=tensor.device == device,
            )
            return nn.Parameter(moved, requires_grad=tensor.requires_grad)
        return tensor.to(
            device=device,
            non_blocking=False,
            copy=tensor.device == device,
        )


__all__ = [
    "QwenImageDirectSlot",
    "QwenImageResidencyOwner",
    "QwenImageSlotInventory",
]

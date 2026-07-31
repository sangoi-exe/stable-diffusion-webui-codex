"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Explicit memmap-preserving streamed residency lifecycle for the Qwen Image Edit-2511 transformer core.
Owns strict activation guards, synchronous static/one-block residency transitions, exact host-object restoration,
and telemetry; streaming_slots.py owns slot inventory, physical alias validation, and staged tensor construction.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageStreamedCoreRuntime` (class): Exact 60-block streamed residency owner for the native Qwen Image transformer.
- `require_qwen_image_streaming_activation` (function): Fail-loud normalized option and blocked-swap admission guard.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch
from torch import nn

from apps.backend.runtime.logging import get_backend_logger
from apps.backend.runtime.memory.config import SwapMethod
from apps.backend.runtime.memory.streamed_residency import (
    StreamedFootprint,
    StreamedResidencyPhase,
    StreamedResidencySnapshot,
)
from apps.backend.runtime.families.qwen_image.streaming_slots import (
    QwenImageResidencyOwner,
    QwenImageSlotInventory,
)


logger = get_backend_logger("backend.runtime.families.qwen_image.streaming")

_CORE_STREAMING_OPTION = "core_streaming_enabled"


def require_qwen_image_streaming_activation(
    options: Mapping[str, object],
    *,
    swap_method: SwapMethod,
) -> None:
    if not isinstance(options, Mapping):
        raise TypeError(
            "Qwen Image streaming activation options must be a mapping; "
            f"got {type(options).__name__}."
        )
    if _CORE_STREAMING_OPTION not in options:
        raise RuntimeError(
            "Qwen Image Edit-2511 requires the existing Core streaming setting: "
            "missing normalized engine option 'core_streaming_enabled'."
        )
    enabled = options[_CORE_STREAMING_OPTION]
    if not isinstance(enabled, bool):
        raise TypeError(
            "Qwen Image normalized engine option 'core_streaming_enabled' must be a boolean; "
            f"got {type(enabled).__name__}."
        )
    if enabled is not True:
        raise RuntimeError(
            "Qwen Image Edit-2511 requires Core streaming to be enabled "
            "(engine option 'core_streaming_enabled'=True)."
        )
    if not isinstance(swap_method, SwapMethod):
        raise TypeError(
            "Qwen Image streaming activation requires a typed SwapMethod; "
            f"got {type(swap_method).__name__}."
        )
    if swap_method is not SwapMethod.BLOCKED:
        raise RuntimeError(
            "Qwen Image Edit-2511 synchronous block streaming requires swap_method='blocked'; "
            f"got {swap_method.value!r}."
        )


class QwenImageStreamedCoreRuntime:
    """Synchronous one-block residency owner for the native Edit-2511 core."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        storage_device: torch.device,
        compute_device: torch.device,
    ) -> None:
        if not isinstance(transformer, nn.Module):
            raise TypeError(
                "Qwen Image streamed runtime requires an nn.Module transformer; "
                f"got {type(transformer).__name__}."
            )
        self._transformer = transformer
        self._storage_device = torch.device(storage_device)
        self._compute_device = torch.device(compute_device)
        if self._storage_device.type != "cpu":
            raise RuntimeError(
                "Qwen Image streamed runtime requires CPU storage; "
                f"got {self._storage_device}."
            )

        self._inventory = QwenImageSlotInventory(
            transformer,
            storage_device=self._storage_device,
        )
        self._static_owner = self._inventory.static_owner
        self._block_owners = self._inventory.block_owners
        self._all_owners = self._inventory.all_owners

        total_host_bytes = sum(owner.packed_bytes for owner in self._all_owners)
        max_segment_bytes = max(owner.packed_bytes for owner in self._block_owners)
        transient_device_bytes = max(
            owner.transient_device_bytes for owner in self._all_owners
        )
        self._footprint = StreamedFootprint(
            total_host_bytes=total_host_bytes,
            static_device_bytes=self._static_owner.packed_bytes,
            max_segment_bytes=max_segment_bytes,
            max_resident_segments=1,
            transient_device_bytes=transient_device_bytes,
        )

        self._phase = StreamedResidencyPhase.OFFLOADED
        self._static_staged: dict[int, torch.Tensor] | None = None
        self._active_block_index: int | None = None
        self._active_block_staged: dict[int, torch.Tensor] | None = None
        self._state_lock = threading.RLock()
        self._execution_lock = threading.RLock()
        self._cpu_to_device_bytes = 0
        self._restored_host_slot_bytes = 0
        self._max_allocated_bytes = 0
        self._max_reserved_bytes = 0
        self._min_free_bytes: int | None = None

    @property
    def footprint(self) -> StreamedFootprint:
        return self._footprint

    def _synchronize_compute(self) -> None:
        if self._compute_device.type == "cuda":
            torch.cuda.synchronize(self._compute_device)

    def _record_device_counters(self) -> None:
        if self._compute_device.type != "cuda":
            return
        allocated = int(torch.cuda.max_memory_allocated(self._compute_device))
        reserved = int(torch.cuda.max_memory_reserved(self._compute_device))
        free_bytes, _total_bytes = torch.cuda.mem_get_info(self._compute_device)
        self._max_allocated_bytes = max(self._max_allocated_bytes, allocated)
        self._max_reserved_bytes = max(self._max_reserved_bytes, reserved)
        free_value = int(free_bytes)
        if self._min_free_bytes is None or free_value < self._min_free_bytes:
            self._min_free_bytes = free_value

    def _stage_owner(self, owner: QwenImageResidencyOwner) -> dict[int, torch.Tensor]:
        staged: dict[int, torch.Tensor] = {}
        installed = False
        try:
            for source in owner.unique_sources:
                staged[id(source)] = self._inventory.stage_tensor(
                    source,
                    device=self._compute_device,
                )
            self._synchronize_compute()
            for slot in owner.slots:
                if slot.current_tensor() is not slot.host_tensor:
                    raise RuntimeError(
                        f"Qwen Image streamed slot {slot.path} changed before {owner.label} staging."
                    )
            installed = True
            for slot in owner.slots:
                slot.install(staged[id(slot.host_tensor)])
            self._cpu_to_device_bytes += owner.packed_bytes
            self._record_device_counters()
            return staged
        except BaseException:
            if installed:
                for slot in owner.slots:
                    slot.install(slot.host_tensor)
            raise

    def _restore_owner(
        self,
        owner: QwenImageResidencyOwner,
        staged: Mapping[int, torch.Tensor],
    ) -> None:
        mismatches: list[str] = []
        for slot in owner.slots:
            expected = staged.get(id(slot.host_tensor))
            if expected is None or slot.current_tensor() is not expected:
                mismatches.append(slot.path)
        for slot in owner.slots:
            slot.install(slot.host_tensor)
        self._restored_host_slot_bytes += owner.packed_bytes
        if mismatches:
            raise RuntimeError(
                f"Qwen Image streamed {owner.label} slot identity changed before restoration: "
                f"slots={mismatches[:20]}."
            )

    def _verify_owner(
        self,
        owner: QwenImageResidencyOwner,
        staged: Mapping[int, torch.Tensor] | None,
    ) -> None:
        for slot in owner.slots:
            slot.verify_host_identity(storage_device=self._storage_device)
            expected = slot.host_tensor if staged is None else staged.get(id(slot.host_tensor))
            if expected is None:
                raise RuntimeError(
                    f"Qwen Image streamed {owner.label} is missing staged identity for {slot.path}."
                )
            current = slot.current_tensor()
            if current is not expected:
                raise RuntimeError(
                    f"Qwen Image streamed slot identity mismatch at {slot.path}: "
                    f"expected_id={id(expected)} current_id={id(current)}."
                )
            expected_device = self._storage_device if staged is None else self._compute_device
            if current.device != expected_device:
                raise RuntimeError(
                    f"Qwen Image streamed slot {slot.path} is on the wrong device: "
                    f"expected={expected_device} got={current.device}."
                )

    def _snapshot_unlocked(self) -> StreamedResidencySnapshot:
        if self._phase is StreamedResidencyPhase.OFFLOADED:
            if self._static_staged is not None or self._active_block_staged is not None:
                raise RuntimeError("Qwen Image OFFLOADED phase retains staged tensor references.")
            if self._active_block_index is not None:
                raise RuntimeError("Qwen Image OFFLOADED phase retains an active block index.")
            self._verify_owner(self._static_owner, None)
            for owner in self._block_owners:
                self._verify_owner(owner, None)
            static_bytes = 0
            resident_indices: tuple[int, ...] = ()
            resident_segment_bytes = 0
        elif self._phase is StreamedResidencyPhase.READY:
            if self._static_staged is None:
                raise RuntimeError("Qwen Image READY phase is missing static staged tensors.")
            if self._active_block_index is not None or self._active_block_staged is not None:
                raise RuntimeError("Qwen Image READY phase cannot retain an active block.")
            self._verify_owner(self._static_owner, self._static_staged)
            for owner in self._block_owners:
                self._verify_owner(owner, None)
            static_bytes = self._static_owner.packed_bytes
            resident_indices = ()
            resident_segment_bytes = 0
        else:
            if self._static_staged is None:
                raise RuntimeError("Qwen Image ACTIVE phase is missing static staged tensors.")
            if self._active_block_index is None or self._active_block_staged is None:
                raise RuntimeError("Qwen Image ACTIVE phase is missing its resident block.")
            self._verify_owner(self._static_owner, self._static_staged)
            for block_index, owner in enumerate(self._block_owners):
                self._verify_owner(
                    owner,
                    self._active_block_staged if block_index == self._active_block_index else None,
                )
            active_owner = self._block_owners[self._active_block_index]
            static_bytes = self._static_owner.packed_bytes
            resident_indices = (self._active_block_index,)
            resident_segment_bytes = active_owner.packed_bytes

        return StreamedResidencySnapshot(
            phase=self._phase,
            storage_device=self._storage_device,
            compute_device=self._compute_device,
            static_resident_bytes=static_bytes,
            resident_segment_indices=resident_indices,
            resident_segment_bytes=resident_segment_bytes,
            current_device_bytes=static_bytes + resident_segment_bytes,
        )

    def prepare_streamed_residency(
        self,
        *,
        storage_device: torch.device,
        compute_device: torch.device,
    ) -> StreamedResidencySnapshot:
        requested_storage = torch.device(storage_device)
        requested_compute = torch.device(compute_device)
        if requested_storage != self._storage_device:
            raise RuntimeError(
                "Qwen Image streamed prepare storage device mismatch: "
                f"configured={self._storage_device} requested={requested_storage}."
            )
        if requested_compute != self._compute_device:
            raise RuntimeError(
                "Qwen Image streamed prepare compute device mismatch: "
                f"configured={self._compute_device} requested={requested_compute}."
            )
        with self._execution_lock, self._state_lock:
            if self._phase is not StreamedResidencyPhase.OFFLOADED:
                raise RuntimeError(
                    "Qwen Image streamed prepare requires OFFLOADED phase; "
                    f"got {self._phase.value}."
                )
            self._inventory.validate_current_topology()
            self._snapshot_unlocked()
            self._cpu_to_device_bytes = 0
            self._restored_host_slot_bytes = 0
            self._max_allocated_bytes = 0
            self._max_reserved_bytes = 0
            self._min_free_bytes = None
            try:
                transfer_started = time.perf_counter()
                staged = self._stage_owner(self._static_owner)
                transfer_ms = (time.perf_counter() - transfer_started) * 1000.0
                self._static_staged = staged
                self._phase = StreamedResidencyPhase.READY
                snapshot = self._snapshot_unlocked()
            except BaseException:
                if self._static_staged is not None:
                    try:
                        self._restore_owner(self._static_owner, self._static_staged)
                    finally:
                        self._static_staged = None
                self._phase = StreamedResidencyPhase.OFFLOADED
                raise
            logger.info(
                "Qwen Image streamed residency prepared: mode=streamed storage=%s compute=%s "
                "host_bytes=%d static_bytes=%d max_segment_bytes=%d transient_bytes=%d "
                "peak_bytes=%d max_resident_segments=1 static_transfer_ms=%.3f",
                self._storage_device,
                self._compute_device,
                self._footprint.total_host_bytes,
                self._footprint.static_device_bytes,
                self._footprint.max_segment_bytes,
                self._footprint.transient_device_bytes,
                self._footprint.peak_device_bytes,
                transfer_ms,
            )
            return snapshot

    def verify_residency(self) -> StreamedResidencySnapshot:
        with self._state_lock:
            return self._snapshot_unlocked()

    @contextmanager
    def activate_block(self, block_index: int) -> Iterator[None]:
        if type(block_index) is not int:
            raise TypeError(
                f"Qwen Image streamed block index must be an exact int; got {type(block_index).__name__}."
            )
        if block_index < 0 or block_index >= len(self._block_owners):
            raise IndexError(
                f"Qwen Image streamed block index out of range: {block_index}."
            )
        with self._execution_lock:
            owner = self._block_owners[block_index]
            with self._state_lock:
                if self._phase is not StreamedResidencyPhase.READY:
                    raise RuntimeError(
                        "Qwen Image streamed block activation requires READY phase; "
                        f"got {self._phase.value}."
                    )
                self._snapshot_unlocked()
                staged: dict[int, torch.Tensor] | None = None
                try:
                    transfer_started = time.perf_counter()
                    staged = self._stage_owner(owner)
                    transfer_ms = (time.perf_counter() - transfer_started) * 1000.0
                    self._active_block_index = block_index
                    self._active_block_staged = staged
                    self._phase = StreamedResidencyPhase.ACTIVE
                    active_snapshot = self._snapshot_unlocked()
                except BaseException as activation_error:
                    cleanup_errors: list[BaseException] = []
                    if staged is not None:
                        try:
                            self._restore_owner(owner, staged)
                        except BaseException as cleanup_error:  # noqa: BLE001 - preserve activation failure
                            cleanup_errors.append(cleanup_error)
                    self._active_block_index = None
                    self._active_block_staged = None
                    self._phase = StreamedResidencyPhase.READY
                    try:
                        self._snapshot_unlocked()
                    except BaseException as cleanup_error:  # noqa: BLE001 - preserve activation failure
                        cleanup_errors.append(cleanup_error)
                    for cleanup_error in cleanup_errors:
                        activation_error.add_note(
                            "Qwen Image streamed activation cleanup failure: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    raise
            primary_error: BaseException | None = None
            cleanup_errors: list[BaseException] = []
            compute_started: float | None = None
            try:
                logger.debug(
                    "Qwen Image streamed block active: block=%d segment_bytes=%d "
                    "resident_blocks=%s resident_count=%d transfer_ms=%.3f "
                    "max_memory_allocated=%d max_memory_reserved=%d min_free_bytes=%s",
                    block_index,
                    owner.packed_bytes,
                    active_snapshot.resident_segment_indices,
                    len(active_snapshot.resident_segment_indices),
                    transfer_ms,
                    self._max_allocated_bytes,
                    self._max_reserved_bytes,
                    self._min_free_bytes,
                )
                compute_started = time.perf_counter()
                yield
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                try:
                    self._synchronize_compute()
                except BaseException as exc:  # noqa: BLE001 - preserve primary block failure
                    cleanup_errors.append(exc)
                compute_ms = (
                    (time.perf_counter() - compute_started) * 1000.0
                    if compute_started is not None
                    else 0.0
                )
                with self._state_lock:
                    try:
                        if self._active_block_staged is not None:
                            self._restore_owner(owner, self._active_block_staged)
                    except BaseException as exc:  # noqa: BLE001 - preserve primary block failure
                        cleanup_errors.append(exc)
                    finally:
                        self._active_block_staged = None
                        self._active_block_index = None
                        self._phase = StreamedResidencyPhase.READY
                    try:
                        self._record_device_counters()
                        self._snapshot_unlocked()
                    except BaseException as exc:  # noqa: BLE001 - preserve primary block failure
                        cleanup_errors.append(exc)
                logger.debug(
                    "Qwen Image streamed block complete: block=%d segment_bytes=%d "
                    "transfer_ms=%.3f compute_ms=%.3f resident_count=0 "
                    "max_memory_allocated=%d max_memory_reserved=%d min_free_bytes=%s",
                    block_index,
                    owner.packed_bytes,
                    transfer_ms,
                    compute_ms,
                    self._max_allocated_bytes,
                    self._max_reserved_bytes,
                    self._min_free_bytes,
                )
                if cleanup_errors:
                    if primary_error is not None:
                        for cleanup_error in cleanup_errors:
                            primary_error.add_note(
                                "Qwen Image streamed block cleanup failure: "
                                f"{type(cleanup_error).__name__}: {cleanup_error}"
                            )
                    else:
                        first_error = cleanup_errors[0]
                        raise RuntimeError(
                            "Qwen Image streamed block cleanup failed: "
                            f"{type(first_error).__name__}: {first_error}"
                        ) from first_error

    def offload_all(self) -> StreamedResidencySnapshot:
        with self._execution_lock, self._state_lock:
            cleanup_errors: list[BaseException] = []
            try:
                self._synchronize_compute()
            except BaseException as exc:  # noqa: BLE001 - continue exact host restoration
                cleanup_errors.append(exc)
            if self._active_block_index is not None and self._active_block_staged is not None:
                try:
                    self._restore_owner(
                        self._block_owners[self._active_block_index],
                        self._active_block_staged,
                    )
                except BaseException as exc:  # noqa: BLE001 - restore remaining owners
                    cleanup_errors.append(exc)
            self._active_block_index = None
            self._active_block_staged = None
            if self._static_staged is not None:
                try:
                    self._restore_owner(self._static_owner, self._static_staged)
                except BaseException as exc:  # noqa: BLE001 - finish state reset
                    cleanup_errors.append(exc)
            self._static_staged = None
            self._phase = StreamedResidencyPhase.OFFLOADED
            try:
                snapshot = self._snapshot_unlocked()
            except BaseException as exc:  # noqa: BLE001 - report complete cleanup failure set
                cleanup_errors.append(exc)
                snapshot = StreamedResidencySnapshot(
                    phase=StreamedResidencyPhase.OFFLOADED,
                    storage_device=self._storage_device,
                    compute_device=self._compute_device,
                    static_resident_bytes=0,
                    resident_segment_indices=(),
                    resident_segment_bytes=0,
                    current_device_bytes=0,
                )
            logger.info(
                "Qwen Image streamed residency offloaded: storage=%s compute=%s "
                "cpu_to_device_bytes=%d restored_host_slot_bytes=%d device_to_host_bytes=0 "
                "max_memory_allocated=%d max_memory_reserved=%d min_free_bytes=%s",
                self._storage_device,
                self._compute_device,
                self._cpu_to_device_bytes,
                self._restored_host_slot_bytes,
                self._max_allocated_bytes,
                self._max_reserved_bytes,
                self._min_free_bytes,
            )
            if cleanup_errors:
                first_error = cleanup_errors[0]
                raise RuntimeError(
                    "Qwen Image streamed offload completed with cleanup failures: "
                    + "; ".join(
                        f"{type(error).__name__}: {error}" for error in cleanup_errors
                    )
                ) from first_error
            return snapshot


__all__ = [
    "QwenImageStreamedCoreRuntime",
    "require_qwen_image_streaming_activation",
]

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed contracts for model cores whose host payload remains offloaded while bounded segments execute on a compute device.

Symbols (top-level; keep in sync; no ghosts):
- `ResidencyMode` (enum): Distinguishes ordinary full-model residency from explicit streamed residency.
- `StreamedResidencyPhase` (enum): Tracks offloaded, prepared, and actively executing streamed-core states.
- `StreamedFootprint` (dataclass): Exact host/static/segment/transient byte contract and derived peak device requirement.
- `StreamedResidencySnapshot` (dataclass): Verified mixed-residency state reported by a streamed runtime.
- `StreamedCoreRuntime` (protocol): Lifecycle contract implemented by family-specific streamed-core runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch


class ResidencyMode(str, Enum):
    FULL = "full"
    STREAMED = "streamed"


class StreamedResidencyPhase(str, Enum):
    OFFLOADED = "offloaded"
    READY = "ready"
    ACTIVE = "active"


def _require_exact_int(name: str, value: int, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


@dataclass(frozen=True, slots=True)
class StreamedFootprint:
    total_host_bytes: int
    static_device_bytes: int
    max_segment_bytes: int
    max_resident_segments: int
    transient_device_bytes: int

    def __post_init__(self) -> None:
        total_host_bytes = _require_exact_int("total_host_bytes", self.total_host_bytes, minimum=1)
        _require_exact_int("static_device_bytes", self.static_device_bytes, minimum=0)
        max_segment_bytes = _require_exact_int("max_segment_bytes", self.max_segment_bytes, minimum=1)
        max_resident_segments = _require_exact_int(
            "max_resident_segments",
            self.max_resident_segments,
            minimum=1,
        )
        _require_exact_int("transient_device_bytes", self.transient_device_bytes, minimum=0)
        if max_segment_bytes > total_host_bytes:
            raise ValueError(
                "max_segment_bytes must not exceed total_host_bytes: "
                f"segment={max_segment_bytes}, host={total_host_bytes}."
            )
        peak_device_bytes = (
            self.static_device_bytes
            + (max_segment_bytes * max_resident_segments)
            + self.transient_device_bytes
        )
        if type(peak_device_bytes) is not int or peak_device_bytes < 0:
            raise ValueError("peak_device_bytes must be representable by exact non-negative integer arithmetic.")

    @property
    def peak_device_bytes(self) -> int:
        return (
            self.static_device_bytes
            + (self.max_segment_bytes * self.max_resident_segments)
            + self.transient_device_bytes
        )


@dataclass(frozen=True, slots=True)
class StreamedResidencySnapshot:
    phase: StreamedResidencyPhase
    storage_device: torch.device
    compute_device: torch.device
    static_resident_bytes: int
    resident_segment_indices: tuple[int, ...]
    resident_segment_bytes: int
    current_device_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.phase, StreamedResidencyPhase):
            raise TypeError(
                "phase must be a StreamedResidencyPhase, "
                f"got {type(self.phase).__name__}."
            )
        if not isinstance(self.storage_device, torch.device):
            raise TypeError(
                "storage_device must be a torch.device, "
                f"got {type(self.storage_device).__name__}."
            )
        if not isinstance(self.compute_device, torch.device):
            raise TypeError(
                "compute_device must be a torch.device, "
                f"got {type(self.compute_device).__name__}."
            )
        static_resident_bytes = _require_exact_int(
            "static_resident_bytes",
            self.static_resident_bytes,
            minimum=0,
        )
        resident_segment_bytes = _require_exact_int(
            "resident_segment_bytes",
            self.resident_segment_bytes,
            minimum=0,
        )
        current_device_bytes = _require_exact_int(
            "current_device_bytes",
            self.current_device_bytes,
            minimum=0,
        )
        if not isinstance(self.resident_segment_indices, tuple):
            raise TypeError("resident_segment_indices must be a tuple of exact ints.")
        seen_indices: set[int] = set()
        for index in self.resident_segment_indices:
            _require_exact_int("resident_segment_index", index, minimum=0)
            if index in seen_indices:
                raise ValueError(f"resident_segment_indices contains duplicate index {index}.")
            seen_indices.add(index)
        expected_device_bytes = static_resident_bytes + resident_segment_bytes
        if current_device_bytes != expected_device_bytes:
            raise ValueError(
                "current_device_bytes must equal static_resident_bytes + resident_segment_bytes: "
                f"current={current_device_bytes}, expected={expected_device_bytes}."
            )
        if self.phase is StreamedResidencyPhase.OFFLOADED:
            if static_resident_bytes or resident_segment_bytes or self.resident_segment_indices:
                raise ValueError("OFFLOADED snapshots cannot report resident static or segment payloads.")
        elif self.phase is StreamedResidencyPhase.READY:
            if resident_segment_bytes or self.resident_segment_indices:
                raise ValueError("READY snapshots cannot report an active resident segment.")
        elif not self.resident_segment_indices or resident_segment_bytes <= 0:
            raise ValueError("ACTIVE snapshots must report at least one resident segment and positive segment bytes.")


@runtime_checkable
class StreamedCoreRuntime(Protocol):
    @property
    def footprint(self) -> StreamedFootprint:
        ...

    def prepare_streamed_residency(
        self,
        *,
        storage_device: torch.device,
        compute_device: torch.device,
    ) -> StreamedResidencySnapshot:
        ...

    def verify_residency(self) -> StreamedResidencySnapshot:
        ...

    def offload_all(self) -> StreamedResidencySnapshot:
        ...


__all__ = [
    "ResidencyMode",
    "StreamedCoreRuntime",
    "StreamedFootprint",
    "StreamedResidencyPhase",
    "StreamedResidencySnapshot",
]

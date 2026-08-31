"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Fail-loud SeedVR2 child-process runner and resource planner for dedicated video upscaling.
Prepares the deterministic SeedVR2 checkout and model directory, models the exact upstream batch schedule plus host/CUDA/MPS phase peaks including
MPS Phase 4 source reconstruction and PNG-conversion memory, measures available accelerator memory, selects the approved direct or bounded
streaming CLI invocation, terminates the complete child process group for task-scoped cancellation, drains diagnostics to EOF, and returns
numerically validated PNG paths.

Symbols (top-level; keep in sync; no ghosts):
- `SeedVR2OutOfMemoryError` (class): Typed local execution failure that maps to the public out-of-memory task result.
- `SeedVR2UpscaleResult` (dataclass): Validated child-output frame paths, dimensions, and runtime evidence for one SeedVR2 run.
- `SeedVR2BatchSchedule` (dataclass): Exact retained upstream batch schedule including uniform and mandatory 4n+1 construction peaks.
- `SeedVR2HostMemoryEstimate` (dataclass): Auditable host-memory estimate including prepend/uniform padding construction for one chunk shape.
- `SeedVR2HostMemoryAdmission` (dataclass): Router-measured host-memory admission shared with runtime plan selection.
- `calculate_seedvr2_target_dimensions` (function): Computes the exact target and padded geometry used by runtime and public admission.
- `calculate_seedvr2_host_memory_admission` (function): Computes direct and host-bounded streaming admission from the upstream schedule.
- `_mps_phase4_live_bytes` (function): Computes the color-correction-aware MPS Phase 4 retained input/output and source-reconstruction peak.
- `run_seedvr2_upscaling` (function): Executes the selected direct or streaming SeedVR2 child route for one source video.
- `__all__` (constant): Explicit export list for this module.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from apps.backend.core.params.video import SeedVR2UpscaleOptions
from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.video.io.ffmpeg import VideoProbe


_SEEDVR2_REPO_ENV = "CODEX_SEEDVR2_REPO_DIR"
_SEEDVR2_REPO_URL_ENV = "CODEX_SEEDVR2_REPO_URL"
_SEEDVR2_REPO_REF_ENV = "CODEX_SEEDVR2_REPO_REF"
_SEEDVR2_MODEL_DIR_ENV = "CODEX_SEEDVR2_MODEL_DIR"
_SEEDVR2_CUDA_DEVICE_ENV = "CODEX_SEEDVR2_CUDA_DEVICE"
_DEFAULT_SEEDVR2_REPO_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
_DEFAULT_SEEDVR2_REPO_REF = "4490bd1f482e026674543386bb2a4d176da245b9"
_DEFAULT_SEEDVR2_RUNTIME_ROOT_RELATIVE = Path(".uv/xdg-data/seedvr2")
_DEFAULT_SEEDVR2_REPO_RELATIVE = _DEFAULT_SEEDVR2_RUNTIME_ROOT_RELATIVE / "repo"
_DEFAULT_SEEDVR2_MODEL_DIR_RELATIVE = Path(".uv/xdg-data/seedvr2")
_SEEDVR2_REPO_LOCK_FILE = ".seedvr2-repo.lock"
_CHILD_OUTPUT_LIMIT = 4000
_CHILD_OUTPUT_READ_SIZE = 1024
_CHILD_POLL_SECONDS = 0.05
_CHILD_TERMINATE_TIMEOUT_SECONDS = 5.0
_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_MINIMUM_OPERATIONAL_RESERVE_BYTES = _GIB
_RUNTIME_RESERVATION_BYTES = 768 * _MIB
_PER_PADDED_OUTPUT_PIXEL_BYTES = 72
_HOST_MEMORY_BASE_BYTES = 64 * _MIB
_HOST_MEMORY_OPERATIONAL_RESERVE_BYTES = _GIB
_TIMING_EVIDENCE_BYTES_PER_FRAME = 4096
_RGB_CHANNELS = 3
_FLOAT32_BYTES = 4
_COMPUTE_DTYPE_BYTES = 2
_UINT8_BYTES = 1
_LATENT_CHANNELS = 16
_LATENT_SPATIAL_DOWNSAMPLE = 8
_LATENT_TEMPORAL_DOWNSAMPLE = 4
_MODEL_RESERVATION_BYTES = {
    "seedvr2_ema_3b_fp16.safetensors": 8 * _GIB,
    "seedvr2_ema_7b_fp16.safetensors": 16 * _GIB,
    "seedvr2_ema_7b_sharp_fp16.safetensors": 16 * _GIB,
}
_OOM_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "cuda oom",
    "not enough memory",
    "alloc_failed",
    "allocation failed",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)
_OOM_MARKER_SUFFIX_SIZE = max(len(marker) for marker in _OOM_MARKERS) - 1
_OUTPUT_FRAME_NAME_RE = re.compile(r".+_(?P<index>[0-9]+)\.png\Z")

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None

try:
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover - non-windows fallback
    msvcrt = None


class SeedVR2OutOfMemoryError(RuntimeError):
    """Fail with a message that the public task error owner classifies as OOM."""


@dataclass(frozen=True, slots=True)
class SeedVR2UpscaleResult:
    frame_paths: tuple[str, ...]
    output_width: int
    output_height: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SeedVR2BatchSchedule:
    processing_frames: int
    batch_count: int
    peak_effective_batch_frames: int
    total_batch_frame_instances: int
    total_latent_frame_instances: int
    uniform_padding_frames: int
    peak_uniform_padding_frames: int
    four_n_one_padding_frames: int
    peak_four_n_one_padding_live_frames: int

    def as_dict(self) -> dict[str, int]:
        return {
            "processing_frames": int(self.processing_frames),
            "batch_count": int(self.batch_count),
            "peak_effective_batch_frames": int(self.peak_effective_batch_frames),
            "total_batch_frame_instances": int(self.total_batch_frame_instances),
            "total_latent_frame_instances": int(self.total_latent_frame_instances),
            "uniform_padding_frames": int(self.uniform_padding_frames),
            "peak_uniform_padding_frames": int(self.peak_uniform_padding_frames),
            "four_n_one_padding_frames": int(self.four_n_one_padding_frames),
            "peak_four_n_one_padding_live_frames": int(self.peak_four_n_one_padding_live_frames),
        }


@dataclass(frozen=True, slots=True)
class SeedVR2HostMemoryEstimate:
    new_frames: int
    retained_context_frames: int
    prepend_frames: int
    required_bytes: int
    timing_evidence_bytes: int
    decoder_peak_bytes: int
    retained_context_bytes: int
    concatenated_float32_bytes: int
    processing_float16_bytes: int
    source_padding_peak_extra_bytes: int
    retained_latent_bytes: int
    source_batch_bytes: int
    target_batch_bytes: int
    returned_float32_bytes: int
    png_multiply_float32_bytes: int
    png_uint8_bytes: int
    schedule: SeedVR2BatchSchedule

    def as_dict(self) -> dict[str, object]:
        return {
            "new_frames": int(self.new_frames),
            "retained_context_frames": int(self.retained_context_frames),
            "prepend_frames": int(self.prepend_frames),
            "required_bytes": int(self.required_bytes),
            "timing_evidence_bytes": int(self.timing_evidence_bytes),
            "decoder_peak_bytes": int(self.decoder_peak_bytes),
            "retained_context_bytes": int(self.retained_context_bytes),
            "concatenated_float32_bytes": int(self.concatenated_float32_bytes),
            "processing_float16_bytes": int(self.processing_float16_bytes),
            "source_padding_peak_extra_bytes": int(self.source_padding_peak_extra_bytes),
            "retained_latent_bytes": int(self.retained_latent_bytes),
            "source_batch_bytes": int(self.source_batch_bytes),
            "target_batch_bytes": int(self.target_batch_bytes),
            "returned_float32_bytes": int(self.returned_float32_bytes),
            "png_multiply_float32_bytes": int(self.png_multiply_float32_bytes),
            "png_uint8_bytes": int(self.png_uint8_bytes),
            "schedule": self.schedule.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class SeedVR2HostMemoryAdmission:
    available_bytes: int
    operational_reserve_bytes: int
    direct: SeedVR2HostMemoryEstimate
    streaming_chunk_size: int
    streaming_first: SeedVR2HostMemoryEstimate
    streaming_steady: SeedVR2HostMemoryEstimate

    @property
    def direct_admitted(self) -> bool:
        return self.direct.required_bytes <= self.available_bytes

    @property
    def streaming_admitted(self) -> bool:
        return self.streaming_chunk_size > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "available_bytes": int(self.available_bytes),
            "operational_reserve_bytes": int(self.operational_reserve_bytes),
            "direct_admitted": bool(self.direct_admitted),
            "streaming_admitted": bool(self.streaming_admitted),
            "streaming_chunk_size": int(self.streaming_chunk_size),
            "direct": self.direct.as_dict(),
            "streaming_first": self.streaming_first.as_dict(),
            "streaming_steady": self.streaming_steady.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class _SeedVR2RepoResolution:
    repo_dir: Path
    uses_default_repo_path: bool
    pinned_ref: str


@dataclass(frozen=True, slots=True)
class _SeedVR2MemoryBudget:
    device_label: str
    metric: str
    available_bytes: int
    operational_reserve_bytes: int
    model_reservation_bytes: int
    runtime_reservation_bytes: int
    source_batch_frame_bytes: int
    target_frame_estimate_bytes: int
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    padded_width: int
    padded_height: int

    @property
    def concurrent_batch_frame_bytes(self) -> int:
        return self.source_batch_frame_bytes + self.target_frame_estimate_bytes

    @property
    def latent_frame_bytes(self) -> int:
        return (
            (self.padded_width // _LATENT_SPATIAL_DOWNSAMPLE)
            * (self.padded_height // _LATENT_SPATIAL_DOWNSAMPLE)
            * _LATENT_CHANNELS
            * _COMPUTE_DTYPE_BYTES
        )

    @property
    def final_output_frame_bytes(self) -> int:
        return self.target_width * self.target_height * _RGB_CHANNELS * _COMPUTE_DTYPE_BYTES


@dataclass(frozen=True, slots=True)
class _SeedVR2ExecutionPlan:
    mode: str
    chunk_size: int
    selection_reason: str
    direct_required_bytes: int
    direct_unified_required_bytes: int | None
    direct_schedule: SeedVR2BatchSchedule
    selected_first_schedule: SeedVR2BatchSchedule
    selected_steady_schedule: SeedVR2BatchSchedule
    budget: _SeedVR2MemoryBudget
    host_memory: SeedVR2HostMemoryAdmission


def _truncate_text(text: str, *, max_chars: int) -> str:
    normalized = str(text or "")
    if len(normalized) <= max_chars:
        return normalized
    keep = max(256, max_chars // 2)
    return normalized[:keep] + "\n...<truncated>...\n" + normalized[-keep:]


def _run_checked_subprocess(
    cmd: Sequence[str],
    *,
    purpose: str,
    cwd: Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess[str]:
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "cwd": str(cwd) if cwd is not None else None,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(list(cmd), **popen_kwargs)
    except FileNotFoundError as exc:
        missing = cmd[0] if cmd else "<unknown>"
        raise RuntimeError(
            f"{purpose} failed because required executable '{missing}' was not found in PATH."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"{purpose} failed to start: {exc}") from exc

    leader_exited_at: float | None = None
    while True:
        try:
            stdout, stderr = process.communicate(timeout=_CHILD_POLL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            if should_cancel is not None and should_cancel():
                _terminate_seedvr2_child(process)
                process.communicate()
                raise RuntimeError("cancelled")
            if process.poll() is None:
                leader_exited_at = None
                continue
            if leader_exited_at is None:
                leader_exited_at = time.monotonic()
                continue
            if time.monotonic() - leader_exited_at < _CHILD_TERMINATE_TIMEOUT_SECONDS:
                continue
            _terminate_seedvr2_child(process)
            stdout, stderr = process.communicate()
            break
    _terminate_seedvr2_child(process)
    proc = subprocess.CompletedProcess(
        args=list(cmd),
        returncode=int(process.returncode or 0),
        stdout=stdout or "",
        stderr=stderr or "",
    )
    if proc.returncode == 0:
        return proc

    stderr_preview = _truncate_text(proc.stderr or proc.stdout or "", max_chars=_CHILD_OUTPUT_LIMIT)
    raise RuntimeError(
        f"{purpose} failed (exit {proc.returncode}; command={list(cmd)!r}).\n"
        f"stderr:\n{stderr_preview}"
    )


def _resolve_seedvr2_repo_url() -> str:
    raw = str(os.environ.get(_SEEDVR2_REPO_URL_ENV) or "").strip()
    return raw or _DEFAULT_SEEDVR2_REPO_URL


def _resolve_seedvr2_repo_ref() -> str:
    raw = str(os.environ.get(_SEEDVR2_REPO_REF_ENV) or "").strip()
    return raw or _DEFAULT_SEEDVR2_REPO_REF


def _resolve_seedvr2_runtime_root_dir() -> Path:
    return (get_repo_root() / _DEFAULT_SEEDVR2_RUNTIME_ROOT_RELATIVE).resolve()


@contextlib.contextmanager
def _exclusive_seedvr2_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            if msvcrt is None:
                raise RuntimeError(
                    "SeedVR2 runtime locking requires the Windows 'msvcrt' module, but it is unavailable."
                )
            os.lseek(lock_fd, 0, os.SEEK_SET)
            os.write(lock_fd, b"0")
            os.lseek(lock_fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(lock_fd, msvcrt.LK_LOCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Failed to acquire SeedVR2 lock at '{lock_path}': {exc}") from exc
            try:
                yield
            finally:
                msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
        else:
            if fcntl is None:
                raise RuntimeError(
                    "SeedVR2 runtime locking requires the POSIX 'fcntl' module, but it is unavailable."
                )
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except OSError as exc:
                raise RuntimeError(f"Failed to acquire SeedVR2 lock at '{lock_path}': {exc}") from exc
            try:
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _ensure_default_repo_is_git_checkout(
    *,
    git_bin: str,
    repo_dir: Path,
    should_cancel: Callable[[], bool] | None,
) -> None:
    try:
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "rev-parse", "--git-dir"],
            purpose="SeedVR2 default repo bootstrap git checkout validation",
            should_cancel=should_cancel,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "SeedVR2 default repo path exists but is not a valid git checkout. "
            f"Expected '{repo_dir}'. Remove it or set {_SEEDVR2_REPO_ENV} to a valid SeedVR2 checkout."
        ) from exc


def _bootstrap_default_seedvr2_repo(
    *,
    repo_dir: Path,
    repo_url: str,
    repo_ref: str,
    should_cancel: Callable[[], bool] | None,
) -> None:
    git_bin = shutil.which("git")
    if not git_bin:
        raise RuntimeError(
            "SeedVR2 default repo bootstrap requires 'git', but it was not found in PATH. "
            f"Install git or set {_SEEDVR2_REPO_ENV} to an existing checkout."
        )

    if repo_dir.exists():
        if not repo_dir.is_dir():
            raise RuntimeError(
                "SeedVR2 repo path exists but is not a directory. "
                f"Expected '{repo_dir}'. Remove it or set {_SEEDVR2_REPO_ENV}."
            )
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_checked_subprocess(
            [git_bin, "clone", "--filter=blob:none", "--no-checkout", repo_url, str(repo_dir)],
            purpose="SeedVR2 default repo bootstrap clone",
            should_cancel=should_cancel,
        )

    _ensure_default_repo_is_git_checkout(
        git_bin=git_bin,
        repo_dir=repo_dir,
        should_cancel=should_cancel,
    )
    try:
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "checkout", "--detach", repo_ref],
            purpose="SeedVR2 default repo bootstrap checkout",
            should_cancel=should_cancel,
        )
    except RuntimeError:
        _raise_if_cancelled(should_cancel)
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "fetch", "--depth", "1", "origin", repo_ref],
            purpose="SeedVR2 default repo bootstrap fetch ref",
            should_cancel=should_cancel,
        )
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "checkout", "--detach", repo_ref],
            purpose="SeedVR2 default repo bootstrap checkout",
            should_cancel=should_cancel,
        )


def _resolve_seedvr2_repo_dir(
    *,
    should_cancel: Callable[[], bool] | None,
) -> _SeedVR2RepoResolution:
    raw = str(os.environ.get(_SEEDVR2_REPO_ENV) or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = get_repo_root() / candidate
        uses_default_repo_path = False
        pinned_ref = ""
    else:
        candidate = get_repo_root() / _DEFAULT_SEEDVR2_REPO_RELATIVE
        uses_default_repo_path = True
        pinned_ref = _resolve_seedvr2_repo_ref()
        with _exclusive_seedvr2_lock(_resolve_seedvr2_runtime_root_dir() / _SEEDVR2_REPO_LOCK_FILE):
            _bootstrap_default_seedvr2_repo(
                repo_dir=candidate.resolve(),
                repo_url=_resolve_seedvr2_repo_url(),
                repo_ref=pinned_ref,
                should_cancel=should_cancel,
            )

    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise RuntimeError(
            "SeedVR2 repo directory is missing. "
            f"Expected '{resolved}'. Set {_SEEDVR2_REPO_ENV} to a valid checkout path."
        )
    entrypoint_path = resolved / "inference_cli.py"
    if not entrypoint_path.is_file():
        raise RuntimeError(
            f"SeedVR2 runtime entrypoint not found at '{entrypoint_path}'. "
            "Verify the repository checkout contains inference_cli.py."
        )
    return _SeedVR2RepoResolution(
        repo_dir=resolved,
        uses_default_repo_path=uses_default_repo_path,
        pinned_ref=pinned_ref,
    )


def _resolve_seedvr2_model_dir() -> Path:
    raw = str(os.environ.get(_SEEDVR2_MODEL_DIR_ENV) or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = get_repo_root() / candidate
    else:
        candidate = get_repo_root() / _DEFAULT_SEEDVR2_MODEL_DIR_RELATIVE
    resolved = candidate.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise RuntimeError(
            "SeedVR2 model directory is invalid. "
            f"Expected a directory at '{resolved}'."
        )
    return resolved


def _normalize_cuda_device_index(component_device: str | None) -> int | None:
    override_raw = str(os.environ.get(_SEEDVR2_CUDA_DEVICE_ENV) or "").strip()
    if override_raw:
        if not re.fullmatch(r"\d+", override_raw):
            raise RuntimeError(
                f"{_SEEDVR2_CUDA_DEVICE_ENV} must be a non-negative CUDA device index, got: {override_raw!r}"
            )
        return int(override_raw)

    raw_device = str(component_device or "").strip().lower()
    requested_component_cuda_index: int | None = None
    if raw_device:
        exact_match = re.fullmatch(r"cuda:(\d+)", raw_device)
        if exact_match:
            requested_component_cuda_index = int(exact_match.group(1))
        elif raw_device not in {"cuda", "cpu", "mps"}:
            return None

    raw_visible = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw_visible:
        return requested_component_cuda_index

    entries = [entry.strip() for entry in raw_visible.split(",") if entry.strip()]
    if not entries:
        return requested_component_cuda_index

    if requested_component_cuda_index is not None:
        if requested_component_cuda_index < len(entries):
            return requested_component_cuda_index

        numeric_entries = [entry for entry in entries if re.fullmatch(r"\d+", entry)]
        if len(numeric_entries) == len(entries):
            if len(set(entries)) != len(entries):
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES contains duplicate numeric entries, so SeedVR2 cannot map devices unambiguously. "
                    f"Got CUDA_VISIBLE_DEVICES={raw_visible!r}. Deduplicate it or set {_SEEDVR2_CUDA_DEVICE_ENV} explicitly."
                )
            requested_token = str(requested_component_cuda_index)
            if requested_token in entries:
                return entries.index(requested_token)

        raise RuntimeError(
            "SeedVR2 CUDA device mapping failed: component device "
            f"{component_device!r} is outside visible index range [0, {len(entries) - 1}] for "
            f"CUDA_VISIBLE_DEVICES={raw_visible!r}, and no physical-id fallback mapping is available. "
            f"Set {_SEEDVR2_CUDA_DEVICE_ENV} to an explicit visible index."
        )

    if raw_device == "cuda" or (not raw_device and len(entries) == 1):
        return 0
    return None


def _sanitize_metadata_path(path: Path) -> str:
    resolved = path.resolve()
    repo_root = get_repo_root().resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        return f"<external>/{resolved.name}"
    return f"CODEX_ROOT/{relative.as_posix()}"


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise RuntimeError("cancelled")


def calculate_seedvr2_target_dimensions(
    *,
    source_width: int,
    source_height: int,
    options: SeedVR2UpscaleOptions,
) -> tuple[int, int, int, int]:
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError(f"SeedVR2 source dimensions must be positive, got {source_width}x{source_height}.")
    resolution = int(options.resolution or 0)
    if resolution <= 0:
        raise RuntimeError(f"SeedVR2 resolution must be positive, got {options.resolution!r}.")
    if source_width <= source_height:
        target_width = resolution
        target_height = int(resolution * source_height / source_width)
    else:
        target_height = resolution
        target_width = int(resolution * source_width / source_height)
    max_resolution = int(options.max_resolution or 0)
    if max_resolution > 0 and max(target_width, target_height) > max_resolution:
        maximum_scale = float(max_resolution) / float(max(target_width, target_height))
        target_width = int(round(target_width * maximum_scale))
        target_height = int(round(target_height * maximum_scale))
    target_width = max(2, target_width)
    target_height = max(2, target_height)
    target_width = max(2, (target_width // 2) * 2)
    target_height = max(2, (target_height // 2) * 2)
    padded_width = int(math.ceil(target_width / 16.0) * 16)
    padded_height = int(math.ceil(target_height / 16.0) * 16)
    return target_width, target_height, padded_width, padded_height


def _calculate_seedvr2_batch_schedule(
    *,
    processing_frames: int,
    options: SeedVR2UpscaleOptions,
) -> SeedVR2BatchSchedule:
    if processing_frames <= 0:
        raise RuntimeError(f"SeedVR2 processing frame count must be positive, got {processing_frames}.")
    batch_size = int(options.batch_size or 0)
    temporal_overlap = int(options.temporal_overlap or 0)
    if batch_size <= 0:
        raise RuntimeError(f"SeedVR2 batch_size must be positive, got {batch_size}.")
    if temporal_overlap < 0 or temporal_overlap >= batch_size:
        raise RuntimeError(
            "SeedVR2 batch scheduling requires temporal_overlap greater than or equal to 0 and lower than batch_size."
        )

    step = batch_size - temporal_overlap if temporal_overlap > 0 else batch_size
    batch_count = 0
    peak_effective_batch_frames = 0
    total_batch_frame_instances = 0
    total_latent_frame_instances = 0
    uniform_padding_frames = 0
    peak_uniform_padding_frames = 0
    four_n_one_padding_frames = 0
    peak_four_n_one_padding_live_frames = 0
    for start in range(0, processing_frames, step):
        raw_batch_frames = min(batch_size, processing_frames - start)
        if start > 0 and raw_batch_frames <= temporal_overlap:
            break
        uniform_batch_frames = raw_batch_frames
        if bool(options.uniform_batch_size) and raw_batch_frames < batch_size:
            uniform_batch_frames = batch_size
        effective_batch_frames = uniform_batch_frames
        if effective_batch_frames % 4 != 1:
            effective_batch_frames = ((effective_batch_frames - 1) // 4 + 1) * 4 + 1

        uniform_padding = uniform_batch_frames - raw_batch_frames
        four_n_one_padding = effective_batch_frames - uniform_batch_frames
        four_n_one_padding_live_frames = (
            2 * effective_batch_frames if four_n_one_padding > 0 else effective_batch_frames
        )

        batch_count += 1
        peak_effective_batch_frames = max(peak_effective_batch_frames, effective_batch_frames)
        total_batch_frame_instances += effective_batch_frames
        total_latent_frame_instances += (
            (effective_batch_frames - 1) // _LATENT_TEMPORAL_DOWNSAMPLE + 1
        )
        uniform_padding_frames += uniform_padding
        peak_uniform_padding_frames = max(peak_uniform_padding_frames, uniform_padding)
        four_n_one_padding_frames += four_n_one_padding
        peak_four_n_one_padding_live_frames = max(
            peak_four_n_one_padding_live_frames,
            four_n_one_padding_live_frames,
        )

    if batch_count <= 0 or peak_effective_batch_frames <= 0:
        raise RuntimeError("SeedVR2 batch scheduling produced no retained upstream batch.")
    return SeedVR2BatchSchedule(
        processing_frames=processing_frames,
        batch_count=batch_count,
        peak_effective_batch_frames=peak_effective_batch_frames,
        total_batch_frame_instances=total_batch_frame_instances,
        total_latent_frame_instances=total_latent_frame_instances,
        uniform_padding_frames=uniform_padding_frames,
        peak_uniform_padding_frames=peak_uniform_padding_frames,
        four_n_one_padding_frames=four_n_one_padding_frames,
        peak_four_n_one_padding_live_frames=peak_four_n_one_padding_live_frames,
    )


def _calculate_host_memory_estimate(
    *,
    new_frames: int,
    retained_context_frames: int,
    prepend_frames: int,
    processing_frames: int,
    source_frame_count: int,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    padded_width: int,
    padded_height: int,
    options: SeedVR2UpscaleOptions,
    operational_reserve_bytes: int,
) -> SeedVR2HostMemoryEstimate:
    schedule = _calculate_seedvr2_batch_schedule(
        processing_frames=processing_frames,
        options=options,
    )
    timing_evidence_bytes = source_frame_count * _TIMING_EVIDENCE_BYTES_PER_FRAME
    source_pixels = source_width * source_height
    target_pixels = target_width * target_height
    padded_pixels = padded_width * padded_height
    source_float32_frame_bytes = source_pixels * _RGB_CHANNELS * _FLOAT32_BYTES
    decoder_peak_bytes = 2 * new_frames * source_float32_frame_bytes
    retained_context_bytes = retained_context_frames * source_float32_frame_bytes
    concatenated_float32_bytes = (
        processing_frames * source_float32_frame_bytes if retained_context_frames > 0 else 0
    )
    processing_float16_bytes = (
        processing_frames * source_pixels * _RGB_CHANNELS * _COMPUTE_DTYPE_BYTES
    )
    prepend_padding_peak_extra_frames = (
        max(0, processing_frames - schedule.peak_effective_batch_frames)
        if prepend_frames > 0
        else 0
    )
    source_padding_peak_extra_frames = max(
        prepend_padding_peak_extra_frames,
        schedule.peak_uniform_padding_frames,
    )
    source_padding_peak_extra_bytes = (
        source_padding_peak_extra_frames
        * source_pixels
        * _RGB_CHANNELS
        * _COMPUTE_DTYPE_BYTES
    )
    latent_width = padded_width // _LATENT_SPATIAL_DOWNSAMPLE
    latent_height = padded_height // _LATENT_SPATIAL_DOWNSAMPLE
    retained_latent_bytes = (
        schedule.total_latent_frame_instances
        * latent_width
        * latent_height
        * _LATENT_CHANNELS
        * _COMPUTE_DTYPE_BYTES
    )
    source_batch_bytes = (
        schedule.peak_effective_batch_frames
        * source_pixels
        * _RGB_CHANNELS
        * _COMPUTE_DTYPE_BYTES
    )
    target_batch_bytes = (
        schedule.peak_effective_batch_frames
        * padded_pixels
        * _RGB_CHANNELS
        * _COMPUTE_DTYPE_BYTES
    )
    returned_float32_bytes = processing_frames * target_pixels * _RGB_CHANNELS * _FLOAT32_BYTES
    png_multiply_float32_bytes = returned_float32_bytes
    png_uint8_bytes = processing_frames * target_pixels * _RGB_CHANNELS * _UINT8_BYTES
    required_bytes = (
        operational_reserve_bytes
        + _HOST_MEMORY_BASE_BYTES
        + timing_evidence_bytes
        + decoder_peak_bytes
        + retained_context_bytes
        + concatenated_float32_bytes
        + processing_float16_bytes
        + source_padding_peak_extra_bytes
        + retained_latent_bytes
        + source_batch_bytes
        + target_batch_bytes
        + returned_float32_bytes
        + png_multiply_float32_bytes
        + png_uint8_bytes
    )
    return SeedVR2HostMemoryEstimate(
        new_frames=new_frames,
        retained_context_frames=retained_context_frames,
        prepend_frames=prepend_frames,
        required_bytes=required_bytes,
        timing_evidence_bytes=timing_evidence_bytes,
        decoder_peak_bytes=decoder_peak_bytes,
        retained_context_bytes=retained_context_bytes,
        concatenated_float32_bytes=concatenated_float32_bytes,
        processing_float16_bytes=processing_float16_bytes,
        source_padding_peak_extra_bytes=source_padding_peak_extra_bytes,
        retained_latent_bytes=retained_latent_bytes,
        source_batch_bytes=source_batch_bytes,
        target_batch_bytes=target_batch_bytes,
        returned_float32_bytes=returned_float32_bytes,
        png_multiply_float32_bytes=png_multiply_float32_bytes,
        png_uint8_bytes=png_uint8_bytes,
        schedule=schedule,
    )


def _calculate_streaming_host_estimates(
    *,
    new_frames: int,
    source_width: int,
    source_height: int,
    source_frame_count: int,
    target_width: int,
    target_height: int,
    padded_width: int,
    padded_height: int,
    options: SeedVR2UpscaleOptions,
    operational_reserve_bytes: int,
) -> tuple[SeedVR2HostMemoryEstimate, SeedVR2HostMemoryEstimate]:
    first_processing_frames = new_frames + int(options.prepend_frames or 0)
    first = _calculate_host_memory_estimate(
        new_frames=new_frames,
        retained_context_frames=0,
        prepend_frames=int(options.prepend_frames or 0),
        processing_frames=first_processing_frames,
        source_frame_count=source_frame_count,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        padded_width=padded_width,
        padded_height=padded_height,
        options=options,
        operational_reserve_bytes=operational_reserve_bytes,
    )
    if new_frames >= source_frame_count:
        return first, first
    retained_context_frames = min(int(options.temporal_overlap or 0), new_frames)
    steady = _calculate_host_memory_estimate(
        new_frames=new_frames,
        retained_context_frames=retained_context_frames,
        prepend_frames=0,
        processing_frames=new_frames + retained_context_frames,
        source_frame_count=source_frame_count,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        padded_width=padded_width,
        padded_height=padded_height,
        options=options,
        operational_reserve_bytes=operational_reserve_bytes,
    )
    return first, steady


def calculate_seedvr2_host_memory_admission(
    *,
    source_width: int,
    source_height: int,
    source_frame_count: int,
    target_width: int,
    target_height: int,
    padded_width: int,
    padded_height: int,
    options: SeedVR2UpscaleOptions,
    available_bytes: int,
) -> SeedVR2HostMemoryAdmission:
    """Calculate direct and largest host-safe streaming shapes from exact upstream scheduling."""

    if source_frame_count <= 0:
        raise RuntimeError(f"SeedVR2 source frame count must be positive, got {source_frame_count}.")
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        raise RuntimeError("SeedVR2 host admission requires positive source and target dimensions.")
    if padded_width < target_width or padded_height < target_height:
        raise RuntimeError("SeedVR2 padded dimensions cannot be smaller than target dimensions.")
    if available_bytes <= 0:
        raise RuntimeError(f"SeedVR2 available host memory must be positive, got {available_bytes}.")

    operational_reserve_bytes = _HOST_MEMORY_OPERATIONAL_RESERVE_BYTES
    direct = _calculate_host_memory_estimate(
        new_frames=source_frame_count,
        retained_context_frames=0,
        prepend_frames=int(options.prepend_frames or 0),
        processing_frames=source_frame_count + int(options.prepend_frames or 0),
        source_frame_count=source_frame_count,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        padded_width=padded_width,
        padded_height=padded_height,
        options=options,
        operational_reserve_bytes=operational_reserve_bytes,
    )

    maximum_streaming_frames = source_frame_count - 1 if source_frame_count > 1 else 1
    minimum_first, minimum_steady = _calculate_streaming_host_estimates(
        new_frames=1,
        source_width=source_width,
        source_height=source_height,
        source_frame_count=source_frame_count,
        target_width=target_width,
        target_height=target_height,
        padded_width=padded_width,
        padded_height=padded_height,
        options=options,
        operational_reserve_bytes=operational_reserve_bytes,
    )
    streaming_chunk_size = 0
    streaming_first = minimum_first
    streaming_steady = minimum_steady
    if max(minimum_first.required_bytes, minimum_steady.required_bytes) <= available_bytes:
        low = 1
        high = maximum_streaming_frames
        while low <= high:
            candidate = (low + high) // 2
            candidate_first, candidate_steady = _calculate_streaming_host_estimates(
                new_frames=candidate,
                source_width=source_width,
                source_height=source_height,
                source_frame_count=source_frame_count,
                target_width=target_width,
                target_height=target_height,
                padded_width=padded_width,
                padded_height=padded_height,
                options=options,
                operational_reserve_bytes=operational_reserve_bytes,
            )
            candidate_required_bytes = max(
                candidate_first.required_bytes,
                candidate_steady.required_bytes,
            )
            if candidate_required_bytes <= available_bytes:
                streaming_chunk_size = candidate
                streaming_first = candidate_first
                streaming_steady = candidate_steady
                low = candidate + 1
            else:
                high = candidate - 1

    return SeedVR2HostMemoryAdmission(
        available_bytes=available_bytes,
        operational_reserve_bytes=operational_reserve_bytes,
        direct=direct,
        streaming_chunk_size=streaming_chunk_size,
        streaming_first=streaming_first,
        streaming_steady=streaming_steady,
    )


def _read_accelerator_memory(
    *,
    component_device: str | None,
    cuda_device_index: int | None,
) -> tuple[str, str, int]:
    raw_device = str(component_device or "").strip().lower()
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"SeedVR2 VRAM admission requires torch: {exc}") from exc

    if raw_device.startswith("cuda"):
        if not bool(torch.cuda.is_available()):
            raise SeedVR2OutOfMemoryError("out of memory: CUDA is unavailable for SeedVR2 execution.")
        device_index = 0 if cuda_device_index is None else int(cuda_device_index)
        device = torch.device("cuda", device_index)
        try:
            from apps.backend.runtime.memory import memory_management

            free_bytes = int(memory_management.manager.get_free_memory(device))
        except Exception as exc:
            raise RuntimeError(f"SeedVR2 failed to measure available CUDA VRAM on {device}: {exc}") from exc
        if free_bytes <= 0:
            raise SeedVR2OutOfMemoryError(f"out of memory: CUDA device {device_index} reports no free VRAM.")
        return f"cuda:{device_index}", "cuda_mem_get_info_free_bytes", free_bytes

    if raw_device == "mps":
        mps_runtime = getattr(torch, "mps", None)
        available = bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())
        if not available or mps_runtime is None:
            raise SeedVR2OutOfMemoryError("out of memory: MPS is unavailable for SeedVR2 execution.")
        recommended_max_memory = getattr(mps_runtime, "recommended_max_memory", None)
        current_allocated_memory = getattr(mps_runtime, "current_allocated_memory", None)
        if not callable(recommended_max_memory) or not callable(current_allocated_memory):
            raise SeedVR2OutOfMemoryError(
                "out of memory: MPS cannot report available accelerator memory for SeedVR2 admission."
            )
        try:
            free_bytes = int(recommended_max_memory()) - int(current_allocated_memory())
        except Exception as exc:
            raise RuntimeError(f"SeedVR2 failed to measure available MPS accelerator memory: {exc}") from exc
        if free_bytes <= 0:
            raise SeedVR2OutOfMemoryError("out of memory: MPS reports no free accelerator memory.")
        return "mps", "mps_recommended_max_minus_current_allocated_bytes", free_bytes

    raise RuntimeError(
        "SeedVR2 VRAM admission requires a CUDA or MPS device, "
        f"got component_device={component_device!r}."
    )


def _operational_reserve_bytes() -> int:
    try:
        from apps.backend.runtime.memory import memory_management

        budgets = memory_management.manager.config.budgets
        hard_reservation_bytes = max(0, int(budgets.hard_reservation_mb)) * _MIB
        safety_margin_bytes = max(0, int(budgets.safety_margin_mb)) * _MIB
        minimum_inference_bytes = max(0, int(budgets.minimum_inference_mb)) * _MIB
    except Exception as exc:
        raise RuntimeError(f"SeedVR2 failed to read the runtime VRAM reserve policy: {exc}") from exc
    return max(_MINIMUM_OPERATIONAL_RESERVE_BYTES, hard_reservation_bytes, minimum_inference_bytes) + safety_margin_bytes


def _model_reservation_bytes(options: SeedVR2UpscaleOptions) -> int:
    model_name = str(options.dit_model or "").strip()
    try:
        return _MODEL_RESERVATION_BYTES[model_name]
    except KeyError as exc:
        raise RuntimeError(f"SeedVR2 VRAM admission has no reservation for model {model_name!r}.") from exc


def _build_memory_budget(
    *,
    source_probe: VideoProbe,
    options: SeedVR2UpscaleOptions,
    component_device: str | None,
    cuda_device_index: int | None,
) -> _SeedVR2MemoryBudget:
    target_width, target_height, padded_width, padded_height = calculate_seedvr2_target_dimensions(
        source_width=int(source_probe.width),
        source_height=int(source_probe.height),
        options=options,
    )
    device_label, metric, available_bytes = _read_accelerator_memory(
        component_device=component_device,
        cuda_device_index=cuda_device_index,
    )
    source_batch_frame_bytes = max(
        1,
        int(source_probe.width) * int(source_probe.height) * _RGB_CHANNELS * _COMPUTE_DTYPE_BYTES,
    )
    target_frame_estimate_bytes = max(1, padded_width * padded_height * _PER_PADDED_OUTPUT_PIXEL_BYTES)
    return _SeedVR2MemoryBudget(
        device_label=device_label,
        metric=metric,
        available_bytes=available_bytes,
        operational_reserve_bytes=_operational_reserve_bytes(),
        model_reservation_bytes=_model_reservation_bytes(options),
        runtime_reservation_bytes=_RUNTIME_RESERVATION_BYTES,
        source_batch_frame_bytes=source_batch_frame_bytes,
        target_frame_estimate_bytes=target_frame_estimate_bytes,
        source_width=int(source_probe.width),
        source_height=int(source_probe.height),
        target_width=target_width,
        target_height=target_height,
        padded_width=padded_width,
        padded_height=padded_height,
    )


def _mps_phase4_live_bytes(
    *,
    budget: _SeedVR2MemoryBudget,
    schedule: SeedVR2BatchSchedule,
    color_correction_enabled: bool,
) -> int:
    complete_input_bytes = schedule.processing_frames * budget.source_batch_frame_bytes
    final_output_bytes = schedule.processing_frames * budget.final_output_frame_bytes
    if not color_correction_enabled:
        return complete_input_bytes + final_output_bytes

    uniform_padding_construction_bytes = 0
    if schedule.peak_uniform_padding_frames > 0:
        uniform_padding_construction_bytes = (
            schedule.peak_uniform_padding_frames + schedule.peak_effective_batch_frames
        ) * budget.source_batch_frame_bytes
    four_n_one_padding_construction_bytes = 0
    if schedule.four_n_one_padding_frames > 0:
        four_n_one_padding_construction_bytes = (
            schedule.peak_four_n_one_padding_live_frames * budget.source_batch_frame_bytes
        )
    transform_and_color_workspace_bytes = (
        schedule.peak_effective_batch_frames * budget.concurrent_batch_frame_bytes
    )
    return complete_input_bytes + final_output_bytes + max(
        uniform_padding_construction_bytes,
        four_n_one_padding_construction_bytes,
        transform_and_color_workspace_bytes,
    )


def _accelerator_schedule_required_bytes(
    *,
    budget: _SeedVR2MemoryBudget,
    schedule: SeedVR2BatchSchedule,
    color_correction_enabled: bool,
) -> int:
    fixed_bytes = (
        budget.operational_reserve_bytes
        + budget.model_reservation_bytes
        + budget.runtime_reservation_bytes
    )
    transform_phase_bytes = (
        schedule.peak_effective_batch_frames * budget.concurrent_batch_frame_bytes
    )
    four_n_one_padding_phase_bytes = (
        schedule.peak_four_n_one_padding_live_frames * budget.source_batch_frame_bytes
    )
    if budget.device_label != "mps":
        return fixed_bytes + max(transform_phase_bytes, four_n_one_padding_phase_bytes)

    complete_input_bytes = schedule.processing_frames * budget.source_batch_frame_bytes
    retained_latent_bytes = schedule.total_latent_frame_instances * budget.latent_frame_bytes
    final_output_bytes = schedule.processing_frames * budget.final_output_frame_bytes
    decode_workspace_bytes = (
        schedule.peak_effective_batch_frames * budget.target_frame_estimate_bytes
    )
    batch_phase_bytes = complete_input_bytes + max(
        transform_phase_bytes,
        four_n_one_padding_phase_bytes,
    )
    decode_phase_bytes = (
        complete_input_bytes
        + retained_latent_bytes
        + final_output_bytes
        + decode_workspace_bytes
    )
    phase4_live_bytes = _mps_phase4_live_bytes(
        budget=budget,
        schedule=schedule,
        color_correction_enabled=color_correction_enabled,
    )
    output_phase_bytes = final_output_bytes
    return fixed_bytes + max(
        batch_phase_bytes,
        decode_phase_bytes,
        phase4_live_bytes,
        output_phase_bytes,
    )


def _mps_unified_required_bytes(
    *,
    budget: _SeedVR2MemoryBudget,
    schedule: SeedVR2BatchSchedule,
    host_estimate: SeedVR2HostMemoryEstimate,
    host_operational_reserve_bytes: int,
    color_correction_enabled: bool,
) -> int:
    if budget.device_label != "mps":
        raise RuntimeError("MPS unified-memory admission requires an MPS memory budget.")

    shared_reserve_bytes = max(
        int(host_operational_reserve_bytes),
        int(budget.operational_reserve_bytes),
    )
    host_only_nonreserve_bytes = max(
        0,
        int(host_estimate.required_bytes) - int(host_operational_reserve_bytes),
    )
    host_processing_nonreserve_bytes = (
        _HOST_MEMORY_BASE_BYTES
        + host_estimate.timing_evidence_bytes
        + host_estimate.decoder_peak_bytes
        + host_estimate.retained_context_bytes
        + host_estimate.concatenated_float32_bytes
        + host_estimate.processing_float16_bytes
        + host_estimate.source_padding_peak_extra_bytes
        + host_estimate.source_batch_bytes
        + host_estimate.target_batch_bytes
    )
    host_output_nonreserve_bytes = (
        _HOST_MEMORY_BASE_BYTES
        + host_estimate.timing_evidence_bytes
        + host_estimate.decoder_peak_bytes
        + host_estimate.retained_context_bytes
        + host_estimate.concatenated_float32_bytes
        + host_estimate.returned_float32_bytes
        + host_estimate.png_multiply_float32_bytes
        + host_estimate.png_uint8_bytes
    )

    complete_input_bytes = schedule.processing_frames * budget.source_batch_frame_bytes
    retained_latent_bytes = schedule.total_latent_frame_instances * budget.latent_frame_bytes
    final_output_bytes = schedule.processing_frames * budget.final_output_frame_bytes
    transform_phase_bytes = (
        schedule.peak_effective_batch_frames * budget.concurrent_batch_frame_bytes
    )
    four_n_one_padding_phase_bytes = (
        schedule.peak_four_n_one_padding_live_frames * budget.source_batch_frame_bytes
    )
    decode_workspace_bytes = (
        schedule.peak_effective_batch_frames * budget.target_frame_estimate_bytes
    )
    fixed_accelerator_nonreserve_bytes = (
        budget.model_reservation_bytes + budget.runtime_reservation_bytes
    )
    accelerator_batch_nonreserve_bytes = (
        fixed_accelerator_nonreserve_bytes
        + complete_input_bytes
        + max(transform_phase_bytes, four_n_one_padding_phase_bytes)
    )
    accelerator_decode_nonreserve_bytes = (
        fixed_accelerator_nonreserve_bytes
        + complete_input_bytes
        + retained_latent_bytes
        + final_output_bytes
        + decode_workspace_bytes
    )
    accelerator_phase4_nonreserve_bytes = (
        fixed_accelerator_nonreserve_bytes
        + _mps_phase4_live_bytes(
            budget=budget,
            schedule=schedule,
            color_correction_enabled=color_correction_enabled,
        )
    )
    accelerator_output_nonreserve_bytes = (
        fixed_accelerator_nonreserve_bytes + final_output_bytes
    )
    return shared_reserve_bytes + max(
        host_only_nonreserve_bytes,
        host_processing_nonreserve_bytes + accelerator_batch_nonreserve_bytes,
        host_processing_nonreserve_bytes + accelerator_decode_nonreserve_bytes,
        host_processing_nonreserve_bytes + accelerator_phase4_nonreserve_bytes,
        host_output_nonreserve_bytes + accelerator_output_nonreserve_bytes,
    )


def _streaming_chunk_size(
    *,
    budget: _SeedVR2MemoryBudget,
    source_frame_count: int,
    options: SeedVR2UpscaleOptions,
    host_memory: SeedVR2HostMemoryAdmission,
) -> int:
    if source_frame_count <= 0:
        raise RuntimeError(f"SeedVR2 source frame count must be positive, got {source_frame_count}.")
    if not host_memory.streaming_admitted:
        required_bytes = max(
            host_memory.streaming_first.required_bytes,
            host_memory.streaming_steady.required_bytes,
        )
        raise SeedVR2OutOfMemoryError(
            "out of memory: SeedVR2 streaming cannot admit one new frame within the measured host-memory reserve; "
            f"required_bytes={required_bytes}, available_bytes={host_memory.available_bytes}, "
            f"reserve_bytes={host_memory.operational_reserve_bytes}."
        )
    maximum_new_frames = min(host_memory.streaming_chunk_size, source_frame_count)
    color_correction_enabled = (
        str(options.color_correction or "lab").strip().lower() != "none"
    )
    low = 1
    high = maximum_new_frames
    selected = 0
    while low <= high:
        candidate = (low + high) // 2
        first_host, steady_host = _calculate_streaming_host_estimates(
            new_frames=candidate,
            source_width=budget.source_width,
            source_height=budget.source_height,
            source_frame_count=source_frame_count,
            target_width=budget.target_width,
            target_height=budget.target_height,
            padded_width=budget.padded_width,
            padded_height=budget.padded_height,
            options=options,
            operational_reserve_bytes=host_memory.operational_reserve_bytes,
        )
        accelerator_required_bytes = max(
            _accelerator_schedule_required_bytes(
                budget=budget,
                schedule=first_host.schedule,
                color_correction_enabled=color_correction_enabled,
            ),
            _accelerator_schedule_required_bytes(
                budget=budget,
                schedule=steady_host.schedule,
                color_correction_enabled=color_correction_enabled,
            ),
        )
        accelerator_admitted = accelerator_required_bytes <= budget.available_bytes
        unified_admitted = True
        if budget.device_label == "mps":
            unified_required_bytes = max(
                _mps_unified_required_bytes(
                    budget=budget,
                    schedule=first_host.schedule,
                    host_estimate=first_host,
                    host_operational_reserve_bytes=host_memory.operational_reserve_bytes,
                    color_correction_enabled=color_correction_enabled,
                ),
                _mps_unified_required_bytes(
                    budget=budget,
                    schedule=steady_host.schedule,
                    host_estimate=steady_host,
                    host_operational_reserve_bytes=host_memory.operational_reserve_bytes,
                    color_correction_enabled=color_correction_enabled,
                ),
            )
            unified_admitted = unified_required_bytes <= host_memory.available_bytes
        if accelerator_admitted and unified_admitted:
            selected = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    if selected <= 0:
        minimum_first, minimum_steady = _calculate_streaming_host_estimates(
            new_frames=1,
            source_width=budget.source_width,
            source_height=budget.source_height,
            source_frame_count=source_frame_count,
            target_width=budget.target_width,
            target_height=budget.target_height,
            padded_width=budget.padded_width,
            padded_height=budget.padded_height,
            options=options,
            operational_reserve_bytes=host_memory.operational_reserve_bytes,
        )
        minimum_accelerator_required_bytes = max(
            _accelerator_schedule_required_bytes(
                budget=budget,
                schedule=minimum_first.schedule,
                color_correction_enabled=color_correction_enabled,
            ),
            _accelerator_schedule_required_bytes(
                budget=budget,
                schedule=minimum_steady.schedule,
                color_correction_enabled=color_correction_enabled,
            ),
        )
        minimum_unified_required_bytes = None
        if budget.device_label == "mps":
            minimum_unified_required_bytes = max(
                _mps_unified_required_bytes(
                    budget=budget,
                    schedule=minimum_first.schedule,
                    host_estimate=minimum_first,
                    host_operational_reserve_bytes=host_memory.operational_reserve_bytes,
                    color_correction_enabled=color_correction_enabled,
                ),
                _mps_unified_required_bytes(
                    budget=budget,
                    schedule=minimum_steady.schedule,
                    host_estimate=minimum_steady,
                    host_operational_reserve_bytes=host_memory.operational_reserve_bytes,
                    color_correction_enabled=color_correction_enabled,
                ),
            )
        if (
            minimum_unified_required_bytes is not None
            and minimum_accelerator_required_bytes <= budget.available_bytes
            and minimum_unified_required_bytes > host_memory.available_bytes
        ):
            raise SeedVR2OutOfMemoryError(
                "out of memory: SeedVR2 streaming cannot admit even one new frame within shared MPS physical memory; "
                f"shared_required_bytes={minimum_unified_required_bytes}, "
                f"shared_available_bytes={host_memory.available_bytes}, "
                f"accelerator_required_bytes={minimum_accelerator_required_bytes}, "
                f"accelerator_available_bytes={budget.available_bytes}."
            )
        raise SeedVR2OutOfMemoryError(
            "out of memory: SeedVR2 streaming cannot admit even one new frame within measured accelerator memory; "
            f"accelerator_available_bytes={budget.available_bytes}, "
            f"accelerator_reserve_bytes={budget.operational_reserve_bytes}, "
            f"model_bytes={budget.model_reservation_bytes}, runtime_bytes={budget.runtime_reservation_bytes}, "
            f"source_batch_frame_bytes={budget.source_batch_frame_bytes}, "
            f"target_frame_estimate_bytes={budget.target_frame_estimate_bytes}, "
            f"accelerator_required_bytes={minimum_accelerator_required_bytes}, "
            f"shared_required_bytes={minimum_unified_required_bytes}, "
            f"shared_available_bytes={host_memory.available_bytes}."
        )
    return selected


def _calculate_streaming_batch_schedules(
    *,
    new_frames: int,
    source_frame_count: int,
    options: SeedVR2UpscaleOptions,
) -> tuple[SeedVR2BatchSchedule, SeedVR2BatchSchedule]:
    if new_frames <= 0:
        raise RuntimeError(f"SeedVR2 streaming new-frame count must be positive, got {new_frames}.")
    first_schedule = _calculate_seedvr2_batch_schedule(
        processing_frames=new_frames + int(options.prepend_frames or 0),
        options=options,
    )
    if new_frames >= source_frame_count:
        return first_schedule, first_schedule
    retained_context_frames = min(int(options.temporal_overlap or 0), new_frames)
    steady_schedule = _calculate_seedvr2_batch_schedule(
        processing_frames=new_frames + retained_context_frames,
        options=options,
    )
    return first_schedule, steady_schedule


def _select_execution_plan(
    *,
    source_probe: VideoProbe,
    source_frame_count: int,
    options: SeedVR2UpscaleOptions,
    component_device: str | None,
    cuda_device_index: int | None,
    host_memory: SeedVR2HostMemoryAdmission,
) -> _SeedVR2ExecutionPlan:
    if bool(options.streaming) and bool(options.smart_fallback):
        raise RuntimeError("SeedVR2 streaming and smart_fallback cannot both be enabled.")
    temporal_overlap = int(options.temporal_overlap or 0)
    batch_size = int(options.batch_size or 0)
    if temporal_overlap < 0 or temporal_overlap >= batch_size:
        raise RuntimeError(
            "SeedVR2 temporal_overlap must be greater than or equal to 0 and lower than batch_size; "
            f"got temporal_overlap={temporal_overlap}, batch_size={batch_size}."
        )
    budget = _build_memory_budget(
        source_probe=source_probe,
        options=options,
        component_device=component_device,
        cuda_device_index=cuda_device_index,
    )
    processing_frames = int(source_frame_count) + int(options.prepend_frames or 0)
    direct_schedule = _calculate_seedvr2_batch_schedule(
        processing_frames=processing_frames,
        options=options,
    )
    color_correction_enabled = (
        str(options.color_correction or "lab").strip().lower() != "none"
    )
    direct_required_bytes = _accelerator_schedule_required_bytes(
        budget=budget,
        schedule=direct_schedule,
        color_correction_enabled=color_correction_enabled,
    )
    direct_unified_required_bytes = None
    if budget.device_label == "mps":
        direct_unified_required_bytes = _mps_unified_required_bytes(
            budget=budget,
            schedule=direct_schedule,
            host_estimate=host_memory.direct,
            host_operational_reserve_bytes=host_memory.operational_reserve_bytes,
            color_correction_enabled=color_correction_enabled,
        )
    direct_accelerator_admitted = direct_required_bytes <= budget.available_bytes
    direct_unified_admitted = (
        direct_unified_required_bytes is None
        or direct_unified_required_bytes <= host_memory.available_bytes
    )
    if bool(options.streaming):
        chunk_size = _streaming_chunk_size(
            budget=budget,
            source_frame_count=source_frame_count,
            options=options,
            host_memory=host_memory,
        )
        first_schedule, steady_schedule = _calculate_streaming_batch_schedules(
            new_frames=chunk_size,
            source_frame_count=source_frame_count,
            options=options,
        )
        return _SeedVR2ExecutionPlan(
            mode="streaming",
            chunk_size=chunk_size,
            selection_reason="streaming_requested",
            direct_required_bytes=direct_required_bytes,
            direct_unified_required_bytes=direct_unified_required_bytes,
            direct_schedule=direct_schedule,
            selected_first_schedule=first_schedule,
            selected_steady_schedule=steady_schedule,
            budget=budget,
            host_memory=host_memory,
        )
    if host_memory.direct_admitted and direct_accelerator_admitted and direct_unified_admitted:
        return _SeedVR2ExecutionPlan(
            mode="direct",
            chunk_size=0,
            selection_reason=(
                "direct_host_accelerator_and_unified_admitted"
                if budget.device_label == "mps"
                else "direct_host_and_accelerator_admitted"
            ),
            direct_required_bytes=direct_required_bytes,
            direct_unified_required_bytes=direct_unified_required_bytes,
            direct_schedule=direct_schedule,
            selected_first_schedule=direct_schedule,
            selected_steady_schedule=direct_schedule,
            budget=budget,
            host_memory=host_memory,
        )
    if bool(options.smart_fallback):
        chunk_size = _streaming_chunk_size(
            budget=budget,
            source_frame_count=source_frame_count,
            options=options,
            host_memory=host_memory,
        )
        first_schedule, steady_schedule = _calculate_streaming_batch_schedules(
            new_frames=chunk_size,
            source_frame_count=source_frame_count,
            options=options,
        )
        if not host_memory.direct_admitted:
            selection_reason = "smart_fallback_after_direct_host_capacity_rejection"
        elif not direct_accelerator_admitted:
            selection_reason = "smart_fallback_after_direct_accelerator_capacity_rejection"
        else:
            selection_reason = "smart_fallback_after_direct_unified_memory_capacity_rejection"
        return _SeedVR2ExecutionPlan(
            mode="streaming",
            chunk_size=chunk_size,
            selection_reason=selection_reason,
            direct_required_bytes=direct_required_bytes,
            direct_unified_required_bytes=direct_unified_required_bytes,
            direct_schedule=direct_schedule,
            selected_first_schedule=first_schedule,
            selected_steady_schedule=steady_schedule,
            budget=budget,
            host_memory=host_memory,
        )
    if not host_memory.direct_admitted:
        raise SeedVR2OutOfMemoryError(
            "out of memory: SeedVR2 direct execution exceeds measured available host memory and smart fallback is disabled; "
            f"required_bytes={host_memory.direct.required_bytes}, available_bytes={host_memory.available_bytes}, "
            f"reserve_bytes={host_memory.operational_reserve_bytes}."
        )
    if not direct_accelerator_admitted:
        raise SeedVR2OutOfMemoryError(
            "out of memory: SeedVR2 direct execution exceeds measured available accelerator memory and smart fallback is disabled; "
            f"accelerator_required_bytes={direct_required_bytes}, "
            f"accelerator_available_bytes={budget.available_bytes}, "
            f"accelerator_reserve_bytes={budget.operational_reserve_bytes}."
        )
    assert direct_unified_required_bytes is not None
    raise SeedVR2OutOfMemoryError(
        "out of memory: SeedVR2 direct execution exceeds shared MPS physical memory and smart fallback is disabled; "
        f"shared_required_bytes={direct_unified_required_bytes}, "
        f"shared_available_bytes={host_memory.available_bytes}, "
        f"accelerator_required_bytes={direct_required_bytes}, "
        f"accelerator_available_bytes={budget.available_bytes}."
    )


def _seedvr2_child_command(
    *,
    repo_dir: Path,
    source_path: Path,
    output_root: Path,
    model_dir: Path,
    options: SeedVR2UpscaleOptions,
    cuda_device_index: int | None,
    plan: _SeedVR2ExecutionPlan,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_dir / "inference_cli.py"),
        str(source_path),
        "--output",
        str(output_root),
        "--output_format",
        "png",
        "--model_dir",
        str(model_dir),
        "--dit_model",
        str(options.dit_model),
        "--resolution",
        str(int(options.resolution or 0)),
        "--max_resolution",
        str(int(options.max_resolution or 0)),
        "--batch_size",
        str(int(options.batch_size or 0)),
        "--skip_first_frames",
        "0",
        "--load_cap",
        "0",
        "--prepend_frames",
        str(int(options.prepend_frames or 0)),
        "--temporal_overlap",
        str(int(options.temporal_overlap or 0)),
        "--color_correction",
        str(options.color_correction or "lab"),
        "--input_noise_scale",
        format(float(options.input_noise_scale or 0.0), ".12g"),
        "--latent_noise_scale",
        format(float(options.latent_noise_scale or 0.0), ".12g"),
    ]
    if bool(options.uniform_batch_size):
        command.append("--uniform_batch_size")
    if plan.mode == "streaming":
        if plan.chunk_size <= 0:
            raise RuntimeError("SeedVR2 streaming plan must have a positive chunk size.")
        command += ["--chunk_size", str(plan.chunk_size)]
    if cuda_device_index is not None:
        command += ["--cuda_device", str(cuda_device_index)]
    return command


def _posix_process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_posix_process_group_exit(
    process: subprocess.Popen[str],
    process_group_id: int,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        process.poll()
        if not _posix_process_group_exists(process_group_id):
            return True
        time.sleep(_CHILD_POLL_SECONDS)
    return not _posix_process_group_exists(process_group_id)


def _terminate_seedvr2_child(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
                timeout=_CHILD_TERMINATE_TIMEOUT_SECONDS,
            )
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        try:
            process.wait(timeout=_CHILD_TERMINATE_TIMEOUT_SECONDS)
        except Exception:
            pass
        return

    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass
    if not _wait_for_posix_process_group_exit(
        process,
        process_group_id,
        timeout_seconds=_CHILD_TERMINATE_TIMEOUT_SECONDS,
    ):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass
        _wait_for_posix_process_group_exit(
            process,
            process_group_id,
            timeout_seconds=_CHILD_TERMINATE_TIMEOUT_SECONDS,
        )
    try:
        process.kill()
    except Exception:
        pass
    try:
        process.wait(timeout=_CHILD_TERMINATE_TIMEOUT_SECONDS)
    except Exception:
        pass


def _run_seedvr2_child(
    command: Sequence[str],
    *,
    repo_dir: Path,
    should_cancel: Callable[[], bool] | None,
) -> str:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    popen_kwargs: dict[str, Any] = {
        "cwd": str(repo_dir),
        "env": environment,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(list(command), **popen_kwargs)
    except FileNotFoundError as exc:
        raise RuntimeError(f"SeedVR2 child executable was not found: {command[0]!r}.") from exc
    except Exception as exc:
        raise RuntimeError(f"SeedVR2 child execution failed to start: {exc}") from exc

    output_parts: deque[str] = deque()
    output_size = 0
    oom_marker_seen = False
    oom_marker_suffix = ""

    def drain_output() -> None:
        nonlocal oom_marker_seen, oom_marker_suffix, output_size
        if process.stdout is None:
            return
        while chunk := process.stdout.read(_CHILD_OUTPUT_READ_SIZE):
            marker_text = (oom_marker_suffix + chunk).lower()
            if any(marker in marker_text for marker in _OOM_MARKERS):
                oom_marker_seen = True
            oom_marker_suffix = marker_text[-_OOM_MARKER_SUFFIX_SIZE:]
            if len(chunk) >= _CHILD_OUTPUT_LIMIT:
                output_parts.clear()
                output_parts.append(chunk[-_CHILD_OUTPUT_LIMIT:])
                output_size = _CHILD_OUTPUT_LIMIT
                continue
            output_parts.append(chunk)
            output_size += len(chunk)
            while output_size > _CHILD_OUTPUT_LIMIT:
                discarded = output_parts.popleft()
                overflow = output_size - _CHILD_OUTPUT_LIMIT
                if len(discarded) > overflow:
                    output_parts.appendleft(discarded[overflow:])
                    output_size -= overflow
                else:
                    output_size -= len(discarded)

    reader = threading.Thread(target=drain_output, name="seedvr2-child-output", daemon=True)
    reader.start()
    cancellation_observed = False
    while True:
        if process.poll() is not None:
            break
        if should_cancel is not None and should_cancel():
            cancellation_observed = True
            _terminate_seedvr2_child(process)
            break
        time.sleep(_CHILD_POLL_SECONDS)

    reader.join(timeout=_CHILD_TERMINATE_TIMEOUT_SECONDS)
    if reader.is_alive():
        _terminate_seedvr2_child(process)
        reader.join()
    else:
        _terminate_seedvr2_child(process)
    if cancellation_observed:
        raise RuntimeError("cancelled")

    output = "".join(output_parts).strip()
    if process.returncode == 0:
        return output
    if oom_marker_seen:
        raise SeedVR2OutOfMemoryError(
            "out of memory: SeedVR2 child GPU execution failed. "
            f"Child output: {_truncate_text(output, max_chars=_CHILD_OUTPUT_LIMIT)}"
        )
    raise RuntimeError(
        "SeedVR2 child execution failed "
        f"(exit {process.returncode}; command={list(command)!r}).\n"
        f"output:\n{_truncate_text(output, max_chars=_CHILD_OUTPUT_LIMIT)}"
    )


def _collect_output_frame_paths(
    *,
    output_root: Path,
    expected_frame_count: int,
    prepend_frames: int,
    should_cancel: Callable[[], bool] | None,
) -> tuple[tuple[str, ...], int, int, int]:
    if expected_frame_count <= 0:
        raise RuntimeError(f"SeedVR2 expected frame count must be positive, got {expected_frame_count}.")
    discovered_paths_by_index: dict[int, Path] = {}
    for output_path in output_root.rglob("*.png"):
        _raise_if_cancelled(should_cancel)
        if not output_path.is_file():
            continue
        match = _OUTPUT_FRAME_NAME_RE.fullmatch(output_path.name)
        if match is None:
            raise RuntimeError(
                "SeedVR2 child produced a PNG with no trailing numeric frame identity: "
                f"'{output_path}'."
            )
        frame_index = int(match.group("index"))
        duplicate_path = discovered_paths_by_index.get(frame_index)
        if duplicate_path is not None:
            raise RuntimeError(
                "SeedVR2 child produced duplicate numeric frame identities: "
                f"index={frame_index}, first='{duplicate_path}', duplicate='{output_path}'."
            )
        discovered_paths_by_index[frame_index] = output_path
    ordered_indices = sorted(discovered_paths_by_index)
    png_paths = tuple(discovered_paths_by_index[index] for index in ordered_indices)
    if not png_paths:
        raise RuntimeError(f"SeedVR2 child produced no PNG frames under '{output_root}'.")
    raw_frame_count = len(png_paths)
    for expected_index, actual_index in enumerate(ordered_indices):
        if actual_index != expected_index:
            raise RuntimeError(
                "SeedVR2 child output frame identities are not contiguous from zero: "
                f"expected_index={expected_index}, actual_index={actual_index}, "
                f"raw_output={raw_frame_count}."
            )
    if raw_frame_count == expected_frame_count:
        normalized_paths = png_paths
        removed_priming_frames = 0
    elif prepend_frames > 0 and raw_frame_count == expected_frame_count + prepend_frames:
        normalized_paths = png_paths[prepend_frames:]
        removed_priming_frames = prepend_frames
    else:
        raise RuntimeError(
            "SeedVR2 child output frame count does not match the source/timing contract: "
            f"expected={expected_frame_count}, raw_output={raw_frame_count}, prepend_frames={prepend_frames}."
        )
    if len(normalized_paths) != expected_frame_count:
        raise RuntimeError(
            "SeedVR2 priming-frame normalization failed to restore the source/timing frame cardinality: "
            f"expected={expected_frame_count}, normalized={len(normalized_paths)}."
        )

    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Pillow is required to inspect SeedVR2 output frames: {exc}") from exc

    output_width: int | None = None
    output_height: int | None = None
    for index, output_path in enumerate(normalized_paths):
        _raise_if_cancelled(should_cancel)
        try:
            with Image.open(output_path) as image:
                width, height = image.size
        except Exception as exc:
            raise RuntimeError(f"SeedVR2 child output frame {index} is unreadable: '{output_path}': {exc}") from exc
        if width <= 0 or height <= 0:
            raise RuntimeError(f"SeedVR2 child output frame {index} has invalid dimensions {width}x{height}.")
        if output_width is None or output_height is None:
            output_width, output_height = int(width), int(height)
        elif (width, height) != (output_width, output_height):
            raise RuntimeError(
                "SeedVR2 child produced inconsistent output dimensions: "
                f"frame[0]={output_width}x{output_height}, frame[{index}]={width}x{height}."
            )
    assert output_width is not None and output_height is not None
    return tuple(str(path) for path in normalized_paths), output_width, output_height, removed_priming_frames


def _run_plan(
    *,
    source_path: Path,
    output_root: Path,
    source_frame_count: int,
    repo_dir: Path,
    model_dir: Path,
    options: SeedVR2UpscaleOptions,
    cuda_device_index: int | None,
    plan: _SeedVR2ExecutionPlan,
    should_cancel: Callable[[], bool] | None,
) -> tuple[tuple[str, ...], int, int, int, str]:
    output_root.mkdir(parents=True, exist_ok=False)
    command = _seedvr2_child_command(
        repo_dir=repo_dir,
        source_path=source_path,
        output_root=output_root,
        model_dir=model_dir,
        options=options,
        cuda_device_index=cuda_device_index,
        plan=plan,
    )
    child_output = _run_seedvr2_child(command, repo_dir=repo_dir, should_cancel=should_cancel)
    frame_paths, output_width, output_height, removed_priming_frames = _collect_output_frame_paths(
        output_root=output_root,
        expected_frame_count=source_frame_count,
        prepend_frames=int(options.prepend_frames or 0),
        should_cancel=should_cancel,
    )
    return frame_paths, output_width, output_height, removed_priming_frames, child_output


def _plan_metadata(plan: _SeedVR2ExecutionPlan) -> dict[str, object]:
    budget = plan.budget
    return {
        "mode": plan.mode,
        "selection_reason": plan.selection_reason,
        "chunk_size": int(plan.chunk_size),
        "direct_required_bytes": int(plan.direct_required_bytes),
        "direct_unified_required_bytes": (
            int(plan.direct_unified_required_bytes)
            if plan.direct_unified_required_bytes is not None
            else None
        ),
        "direct_schedule": plan.direct_schedule.as_dict(),
        "selected_schedule": {
            "first_chunk": plan.selected_first_schedule.as_dict(),
            "steady_chunk": plan.selected_steady_schedule.as_dict(),
        },
        "host_memory": plan.host_memory.as_dict(),
        "device": budget.device_label,
        "memory_metric": budget.metric,
        "memory_scope": "unified_host_and_mps" if budget.device_label == "mps" else "accelerator",
        "available_bytes": int(budget.available_bytes),
        "operational_reserve_bytes": int(budget.operational_reserve_bytes),
        "model_reservation_bytes": int(budget.model_reservation_bytes),
        "runtime_reservation_bytes": int(budget.runtime_reservation_bytes),
        "source_batch_frame_bytes": int(budget.source_batch_frame_bytes),
        "target_frame_estimate_bytes": int(budget.target_frame_estimate_bytes),
        "concurrent_batch_frame_bytes": int(budget.concurrent_batch_frame_bytes),
        "latent_frame_bytes": int(budget.latent_frame_bytes),
        "final_output_frame_bytes": int(budget.final_output_frame_bytes),
        "source_size_estimate": {"width": int(budget.source_width), "height": int(budget.source_height)},
        "target_size_estimate": {"width": int(budget.target_width), "height": int(budget.target_height)},
        "padded_size_estimate": {"width": int(budget.padded_width), "height": int(budget.padded_height)},
    }


def run_seedvr2_upscaling(
    source_path: str,
    *,
    source_probe: VideoProbe,
    source_frame_count: int,
    output_dir: str | Path,
    options: SeedVR2UpscaleOptions,
    component_device: str | None,
    host_memory: SeedVR2HostMemoryAdmission,
    should_cancel: Callable[[], bool] | None = None,
    logger_: logging.Logger | None = None,
) -> SeedVR2UpscaleResult:
    """Run one bounded SeedVR2 child execution and return source-cardinality PNG output paths."""

    resolved_source_path = Path(source_path).expanduser().resolve()
    if not resolved_source_path.is_file():
        raise RuntimeError("SeedVR2 source video path does not point to a readable file.")
    if source_frame_count <= 0:
        raise RuntimeError(f"SeedVR2 source timing returned invalid frame count {source_frame_count}.")
    _raise_if_cancelled(should_cancel)
    repo_resolution = _resolve_seedvr2_repo_dir(should_cancel=should_cancel)
    model_dir = _resolve_seedvr2_model_dir()
    cuda_device_index = _normalize_cuda_device_index(component_device)
    output_base_dir = Path(output_dir).expanduser().resolve()
    if output_base_dir.exists():
        raise RuntimeError(f"SeedVR2 output directory must not already exist: '{output_base_dir}'.")
    output_base_dir.mkdir(parents=True, exist_ok=False)

    plan = _select_execution_plan(
        source_probe=source_probe,
        source_frame_count=source_frame_count,
        options=options,
        component_device=component_device,
        cuda_device_index=cuda_device_index,
        host_memory=host_memory,
    )
    selected_plan = plan
    fallback_attempted = (
        plan.mode == "streaming"
        and plan.selection_reason.startswith("smart_fallback_after_direct_")
    )
    try:
        frame_paths, output_width, output_height, removed_priming_frames, child_output = _run_plan(
            source_path=resolved_source_path,
            output_root=output_base_dir / plan.mode,
            source_frame_count=source_frame_count,
            repo_dir=repo_resolution.repo_dir,
            model_dir=model_dir,
            options=options,
            cuda_device_index=cuda_device_index,
            plan=plan,
            should_cancel=should_cancel,
        )
    except SeedVR2OutOfMemoryError:
        if plan.mode != "direct" or not bool(options.smart_fallback):
            raise
        if should_cancel is not None and should_cancel():
            raise
        fallback_attempted = True
        selected_plan = _select_execution_plan(
            source_probe=source_probe,
            source_frame_count=source_frame_count,
            options=SeedVR2UpscaleOptions(
                **{**options.as_dict(), "streaming": True, "smart_fallback": False}
            ),
            component_device=component_device,
            cuda_device_index=cuda_device_index,
            host_memory=host_memory,
        )
        if selected_plan.mode != "streaming":
            raise RuntimeError("SeedVR2 smart-fallback retry did not produce a streaming plan.")
        selected_plan = replace(
            selected_plan,
            selection_reason="smart_fallback_after_direct_oom",
        )
        frame_paths, output_width, output_height, removed_priming_frames, child_output = _run_plan(
            source_path=resolved_source_path,
            output_root=output_base_dir / "streaming_after_direct_oom",
            source_frame_count=source_frame_count,
            repo_dir=repo_resolution.repo_dir,
            model_dir=model_dir,
            options=options,
            cuda_device_index=cuda_device_index,
            plan=selected_plan,
            should_cancel=should_cancel,
        )
    _raise_if_cancelled(should_cancel)

    metadata: dict[str, Any] = {
        "applied": True,
        "runner": "seedvr2",
        "execution_mode": selected_plan.mode,
        "smart_fallback_attempted": fallback_attempted,
        "input_frames": int(source_frame_count),
        "raw_child_output_frames": int(source_frame_count + removed_priming_frames),
        "output_frames": len(frame_paths),
        "priming_frames_removed": int(removed_priming_frames),
        "input_size": {"width": int(source_probe.width), "height": int(source_probe.height)},
        "output_size": {"width": int(output_width), "height": int(output_height)},
        "execution": _plan_metadata(selected_plan),
        "repo_dir": _sanitize_metadata_path(repo_resolution.repo_dir),
        "model_dir": _sanitize_metadata_path(model_dir),
    }
    if repo_resolution.uses_default_repo_path:
        metadata["repo_ref"] = repo_resolution.pinned_ref
    if cuda_device_index is not None:
        metadata["cuda_device"] = int(cuda_device_index)
    if child_output:
        metadata["child_output_tail"] = _truncate_text(child_output, max_chars=1000)

    if logger_ is not None:
        logger_.info(
            "video upscaling (SeedVR2 child %s): %d frame(s) %dx%d -> %dx%d, chunk_size=%d, available_vram_bytes=%d",
            selected_plan.mode,
            source_frame_count,
            int(source_probe.width),
            int(source_probe.height),
            output_width,
            output_height,
            selected_plan.chunk_size,
            selected_plan.budget.available_bytes,
        )

    return SeedVR2UpscaleResult(
        frame_paths=frame_paths,
        output_width=output_width,
        output_height=output_height,
        metadata=metadata,
    )


__all__ = [
    "SeedVR2BatchSchedule",
    "SeedVR2HostMemoryAdmission",
    "SeedVR2HostMemoryEstimate",
    "SeedVR2OutOfMemoryError",
    "SeedVR2UpscaleResult",
    "calculate_seedvr2_host_memory_admission",
    "calculate_seedvr2_target_dimensions",
    "run_seedvr2_upscaling",
]

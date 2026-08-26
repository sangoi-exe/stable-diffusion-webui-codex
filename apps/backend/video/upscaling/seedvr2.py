"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Fail-loud SeedVR2 in-process runner for dedicated video upscaling.
Validates frame inputs, provisions deterministic repo-local SeedVR2 runtime assets (repo checkout),
loads SeedVR2 Python modules directly from the checkout, runs upscaling in-process on tensors, and
returns validated PIL output frames.

Symbols (top-level; keep in sync; no ghosts):
- `run_seedvr2_upscaling` (function): Executes SeedVR2 upscaling from in-memory frames and returns `(frames_out, metadata)`.
- `__all__` (constant): Explicit export list for this module.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from apps.backend.core.params.video import SeedVR2UpscaleOptions
from apps.backend.infra.config.repo_root import get_repo_root, repo_scratch_path

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
_STDERR_PREVIEW_LIMIT = 4000

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None

try:
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover - non-windows fallback
    msvcrt = None


@dataclass(frozen=True)
class _SeedVR2RepoResolution:
    repo_dir: Path
    uses_default_repo_path: bool
    pinned_ref: str


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
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            list(cmd),
            check=False,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
        )
    except FileNotFoundError as exc:
        missing = cmd[0] if cmd else "<unknown>"
        raise RuntimeError(
            f"{purpose} failed because required executable '{missing}' was not found in PATH."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"{purpose} failed to start: {exc}") from exc

    if proc.returncode == 0:
        return proc

    stderr_preview = _truncate_text(proc.stderr or proc.stdout or "", max_chars=_STDERR_PREVIEW_LIMIT)
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


def _ensure_default_repo_is_git_checkout(*, git_bin: str, repo_dir: Path) -> None:
    try:
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "rev-parse", "--git-dir"],
            purpose="SeedVR2 default repo bootstrap git checkout validation",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "SeedVR2 default repo path exists but is not a valid git checkout. "
            f"Expected '{repo_dir}'. Remove it or set {_SEEDVR2_REPO_ENV} to a valid SeedVR2 checkout."
        ) from exc


def _bootstrap_default_seedvr2_repo(*, repo_dir: Path, repo_url: str, repo_ref: str) -> None:
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
        )

    _ensure_default_repo_is_git_checkout(git_bin=git_bin, repo_dir=repo_dir)

    try:
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "checkout", "--detach", repo_ref],
            purpose="SeedVR2 default repo bootstrap checkout",
        )
    except RuntimeError:
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "fetch", "--depth", "1", "origin", repo_ref],
            purpose="SeedVR2 default repo bootstrap fetch ref",
        )
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "checkout", "--detach", repo_ref],
            purpose="SeedVR2 default repo bootstrap checkout",
        )


def _resolve_seedvr2_repo_dir() -> _SeedVR2RepoResolution:
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
        elif raw_device not in {"cuda", "cpu"}:
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


def _validate_input_frames(frames: Sequence[Any]) -> tuple[list[Any], tuple[int, int]]:
    from PIL import Image  # type: ignore

    frames_list = frames if isinstance(frames, list) else list(frames)
    if not frames_list:
        raise RuntimeError("SeedVR2 upscaling requires a non-empty frame sequence.")

    first = frames_list[0]
    if not isinstance(first, Image.Image):
        raise RuntimeError(
            "SeedVR2 upscaling requires PIL image frames; "
            f"frame[0] is {type(first).__name__}."
        )
    size = first.size
    if size[0] <= 0 or size[1] <= 0:
        raise RuntimeError(f"SeedVR2 upscaling received invalid frame size: {size!r}.")

    for index, frame in enumerate(frames_list, start=1):
        if not isinstance(frame, Image.Image):
            raise RuntimeError(
                "SeedVR2 upscaling requires PIL image frames; "
                f"frame[{index - 1}] is {type(frame).__name__}."
            )
        if frame.size != size:
            raise RuntimeError(
                "SeedVR2 upscaling requires same-size frames; "
                f"frame[0]={size!r} frame[{index - 1}]={frame.size!r}."
            )
    return frames_list, size


def _load_seedvr2_inference_module(repo_dir: Path) -> Any:
    module_path = repo_dir / "inference_cli.py"
    if not module_path.is_file():
        raise RuntimeError(
            f"SeedVR2 runtime entrypoint not found at '{module_path}'. "
            "Verify the repository checkout contains inference_cli.py."
        )

    module_name = f"codex_seedvr2_runtime_{os.getpid()}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to build import spec for SeedVR2 module at '{module_path}'.")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing_name = str(exc.name or "<unknown>")
        requirements_path = repo_dir / "requirements.txt"
        raise RuntimeError(
            "SeedVR2 in-process runtime import failed due to missing dependency "
            f"'{missing_name}'. Install runtime dependencies from '{requirements_path}' into this backend environment."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to import SeedVR2 runtime module from '{module_path}': {exc}") from exc
    return module


def _build_seedvr2_runtime_args(
    *,
    module: Any,
    options: SeedVR2UpscaleOptions,
    model_dir: Path,
    cuda_device_index: int | None,
) -> Any:
    if not hasattr(module, "parse_arguments"):
        raise RuntimeError("SeedVR2 runtime module is missing parse_arguments().")

    original_argv = list(sys.argv)
    try:
        sys.argv = ["inference_cli.py", "__codex_in_memory__.png", "--output_format", "png"]
        try:
            args = module.parse_arguments()
        except SystemExit as exc:
            raise RuntimeError(
                "Failed to build SeedVR2 runtime argument namespace for in-process execution. "
                f"parse_arguments exited with code {exc.code!r}."
            ) from exc
    finally:
        sys.argv = original_argv

    args.input = "__codex_in_memory__.png"
    args.output = None
    args.output_format = "png"
    args.model_dir = str(model_dir)
    args.debug = False
    args.chunk_size = 0
    args.skip_first_frames = 0
    args.load_cap = 0
    args.cache_dit = False
    args.cache_vae = False

    if options.dit_model:
        args.dit_model = str(options.dit_model)
    if options.resolution is not None:
        args.resolution = int(options.resolution)
    if options.max_resolution is not None:
        args.max_resolution = int(options.max_resolution)
    if options.batch_size is not None:
        args.batch_size = int(options.batch_size)
    if options.uniform_batch_size is not None:
        args.uniform_batch_size = bool(options.uniform_batch_size)
    if options.temporal_overlap is not None:
        args.temporal_overlap = int(options.temporal_overlap)
    if options.prepend_frames is not None:
        args.prepend_frames = int(options.prepend_frames)
    if options.color_correction:
        args.color_correction = str(options.color_correction)
    if options.input_noise_scale is not None:
        args.input_noise_scale = float(options.input_noise_scale)
    if options.latent_noise_scale is not None:
        args.latent_noise_scale = float(options.latent_noise_scale)

    args.cuda_device = str(cuda_device_index) if cuda_device_index is not None else None
    return args


def _resolve_seedvr2_runtime_device_id(*, module: Any, cuda_device_index: int | None) -> str:
    if not hasattr(module, "get_gpu_backend"):
        raise RuntimeError("SeedVR2 runtime module is missing get_gpu_backend().")
    if not hasattr(module, "torch"):
        raise RuntimeError("SeedVR2 runtime module did not initialize torch runtime.")

    backend = str(module.get_gpu_backend() or "").strip().lower()
    if backend == "cuda":
        torch_mod = module.torch
        if not bool(torch_mod.cuda.is_available()):
            raise RuntimeError(
                "SeedVR2 in-process runtime selected CUDA backend but torch reports CUDA unavailable."
            )
        visible_count = int(torch_mod.cuda.device_count())
        if visible_count <= 0:
            raise RuntimeError(
                "SeedVR2 in-process runtime selected CUDA backend but no visible CUDA devices were found."
            )
        requested_index = 0 if cuda_device_index is None else int(cuda_device_index)
        if requested_index < 0 or requested_index >= visible_count:
            raise RuntimeError(
                "SeedVR2 in-process CUDA device index is out of range for visible devices: "
                f"requested={requested_index}, visible_count={visible_count}. "
                f"Set {_SEEDVR2_CUDA_DEVICE_ENV} to a valid visible index."
            )
        return str(requested_index)

    if backend == "mps":
        return "0"

    raise RuntimeError(
        "SeedVR2 in-process runtime requires CUDA or MPS backend, "
        f"but detected backend={backend!r}."
    )


def _frames_to_seedvr2_tensor(*, module: Any, frames_list: list[Any]) -> Any:
    np_mod = getattr(module, "np", None)
    torch_mod = getattr(module, "torch", None)
    if np_mod is None:
        raise RuntimeError("SeedVR2 runtime module did not provide numpy (np) namespace.")
    if torch_mod is None:
        raise RuntimeError("SeedVR2 runtime module did not provide torch namespace.")

    frame_arrays: list[Any] = []
    for index, frame in enumerate(frames_list, start=1):
        rgb_frame = frame.convert("RGB")
        frame_array = np_mod.asarray(rgb_frame, dtype=np_mod.float32)
        if frame_array.ndim != 3 or int(frame_array.shape[2]) != 3:
            raise RuntimeError(
                "SeedVR2 input frame conversion produced invalid array shape for frame "
                f"{index - 1}: {tuple(frame_array.shape)!r}."
            )
        frame_arrays.append(frame_array / 255.0)

    stacked = np_mod.stack(frame_arrays, axis=0)
    return torch_mod.from_numpy(stacked).to(torch_mod.float16)


def _collect_in_process_output_frames(*, module: Any, result_tensor: Any, expected_count: int) -> list[Any]:
    from PIL import Image  # type: ignore

    torch_mod = getattr(module, "torch", None)
    if torch_mod is None:
        raise RuntimeError("SeedVR2 runtime module did not provide torch namespace.")
    if not bool(torch_mod.is_tensor(result_tensor)):
        raise RuntimeError(
            "SeedVR2 in-process runtime returned a non-tensor result "
            f"({type(result_tensor).__name__})."
        )

    output_tensor = result_tensor.detach()
    if output_tensor.ndim != 4:
        raise RuntimeError(
            "SeedVR2 in-process runtime returned tensor with invalid rank; "
            f"expected 4D [T,H,W,C], got shape={tuple(output_tensor.shape)!r}."
        )

    frame_count = int(output_tensor.shape[0])
    height = int(output_tensor.shape[1])
    width = int(output_tensor.shape[2])
    channels = int(output_tensor.shape[3])

    if frame_count != int(expected_count):
        raise RuntimeError(
            "SeedVR2 in-process output frame count mismatch: "
            f"expected {expected_count}, got {frame_count}."
        )
    if height <= 0 or width <= 0:
        raise RuntimeError(
            "SeedVR2 in-process output contains non-positive frame dimensions: "
            f"shape={tuple(output_tensor.shape)!r}."
        )
    if channels != 3:
        raise RuntimeError(
            "SeedVR2 in-process output channel mismatch: expected RGB (3 channels), "
            f"got {channels}."
        )

    output_cpu = output_tensor.to(device="cpu", dtype=torch_mod.float32).clamp_(0.0, 1.0)
    output_np = (output_cpu.numpy() * 255.0).round().astype("uint8")

    out_frames: list[Any] = []
    for frame_index in range(frame_count):
        out_frames.append(Image.fromarray(output_np[frame_index], mode="RGB"))

    first_size = out_frames[0].size
    for index, frame in enumerate(out_frames, start=1):
        if frame.size != first_size:
            raise RuntimeError(
                "SeedVR2 in-process produced inconsistent output frame sizes: "
                f"frame[0]={first_size!r} frame[{index - 1}]={frame.size!r}."
            )

    return out_frames


def _run_seedvr2_in_process(
    *,
    frames_list: list[Any],
    options: SeedVR2UpscaleOptions,
    repo_dir: Path,
    model_dir: Path,
    cuda_device_index: int | None,
) -> tuple[list[Any], dict[str, Any]]:
    module = _load_seedvr2_inference_module(repo_dir)

    required_symbols = (
        "Debug",
        "DEFAULT_VAE",
        "download_weight",
        "_single_gpu_direct_processing",
    )
    missing = [name for name in required_symbols if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "SeedVR2 runtime module is missing required symbols for in-process execution: "
            f"{', '.join(sorted(missing))}."
        )

    runtime_args = _build_seedvr2_runtime_args(
        module=module,
        options=options,
        model_dir=model_dir,
        cuda_device_index=cuda_device_index,
    )
    attention_mode = str(getattr(runtime_args, "attention_mode", "sdpa") or "sdpa").strip().lower()
    sdpa_flash_runtime: bool | None = None
    if attention_mode == "sdpa":
        torch_mod = getattr(module, "torch", None)
        backends = getattr(torch_mod, "backends", None) if torch_mod is not None else None
        cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
        if cuda_backend is not None and hasattr(cuda_backend, "flash_sdp_enabled"):
            try:
                sdpa_flash_runtime = bool(cuda_backend.flash_sdp_enabled())
            except Exception:
                sdpa_flash_runtime = None

    try:
        module.debug = module.Debug(enabled=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize SeedVR2 runtime debug context: {exc}") from exc

    try:
        download_ok = bool(
            module.download_weight(
                dit_model=runtime_args.dit_model,
                vae_model=module.DEFAULT_VAE,
                model_dir=str(model_dir),
                debug=module.debug,
            )
        )
    except Exception as exc:
        raise RuntimeError(f"SeedVR2 model preparation failed: {exc}") from exc
    if not download_ok:
        raise RuntimeError(
            "SeedVR2 model preparation failed: download_weight reported failure. "
            f"DiT model={runtime_args.dit_model!r}, VAE model={module.DEFAULT_VAE!r}, model_dir='{model_dir}'."
        )

    device_id = _resolve_seedvr2_runtime_device_id(module=module, cuda_device_index=cuda_device_index)
    input_tensor = _frames_to_seedvr2_tensor(module=module, frames_list=frames_list)

    try:
        output_tensor = module._single_gpu_direct_processing(
            frames_tensor=input_tensor,
            args=runtime_args,
            device_id=device_id,
            runner_cache=None,
        )
    except Exception as exc:
        raise RuntimeError(f"SeedVR2 in-process runtime execution failed: {exc}") from exc

    out_frames = _collect_in_process_output_frames(
        module=module,
        result_tensor=output_tensor,
        expected_count=len(frames_list),
    )
    runtime_meta: dict[str, Any] = {"attention_mode": attention_mode}
    if sdpa_flash_runtime is not None:
        runtime_meta["sdpa_flash_runtime"] = bool(sdpa_flash_runtime)
    return out_frames, runtime_meta


def run_seedvr2_upscaling(
    frames: Sequence[Any],
    *,
    options: SeedVR2UpscaleOptions,
    component_device: str | None,
    logger_: logging.Logger | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    frames_list, input_size = _validate_input_frames(frames)
    repo_resolution = _resolve_seedvr2_repo_dir()
    repo_dir = repo_resolution.repo_dir
    model_dir = _resolve_seedvr2_model_dir()
    cuda_device_index = _normalize_cuda_device_index(component_device)

    out_frames, runtime_meta = _run_seedvr2_in_process(
        frames_list=frames_list,
        options=options,
        repo_dir=repo_dir,
        model_dir=model_dir,
        cuda_device_index=cuda_device_index,
    )

    output_size = out_frames[0].size
    run_id = uuid4().hex
    work_dir = repo_scratch_path("seedvr2", run_id)
    meta: dict[str, Any] = {
        "applied": True,
        "runner": "seedvr2",
        "execution_mode": "in_process",
        "input_frames": len(frames_list),
        "output_frames": len(out_frames),
        "input_size": {"width": int(input_size[0]), "height": int(input_size[1])},
        "output_size": {"width": int(output_size[0]), "height": int(output_size[1])},
        "work_dir": _sanitize_metadata_path(work_dir),
        "repo_dir": _sanitize_metadata_path(repo_dir),
        "model_dir": _sanitize_metadata_path(model_dir),
    }
    if repo_resolution.uses_default_repo_path:
        meta["repo_ref"] = repo_resolution.pinned_ref
    if cuda_device_index is not None:
        meta["cuda_device"] = int(cuda_device_index)
    if runtime_meta:
        meta["attention"] = dict(runtime_meta)

    if logger_ is not None:
        logger_.info(
            "video upscaling (SeedVR2 in-process): %d frame(s) %dx%d -> %dx%d",
            len(frames_list),
            int(input_size[0]),
            int(input_size[1]),
            int(output_size[0]),
            int(output_size[1]),
        )
        if str(runtime_meta.get("attention_mode") or "").strip().lower() == "sdpa":
            sdpa_flash_runtime = runtime_meta.get("sdpa_flash_runtime")
            logger_.info(
                "seedvr2 attention runtime: mode=sdpa sdpa_flash_runtime=%s "
                "(SeedVR2 'Flash Attention' optimization check reports flash-attn package availability, not PyTorch SDPA flash kernels).",
                sdpa_flash_runtime if sdpa_flash_runtime is not None else "unknown",
            )

    return out_frames, meta


__all__ = ["run_seedvr2_upscaling"]

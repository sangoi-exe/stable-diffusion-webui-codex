"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Checkpoint/VAE discovery with sha256 and layout-metadata caching.
Scans configured model roots (via `apps/paths.json` accessors) for checkpoint and VAE weight files, including file-level checkpoint entries,
computes sha256 hashes, and maintains a persistent cache in `models/.hashes.json` (schema v2) for fast UI inventory, backend SHA-based
resolution, CLIP layout metadata reuse, and checkpoint-scoped metadata forwarding such as current LTX2 execution-profile/default hints,
Netflix VOID overlay pairing readiness, exact Qwen Image Edit-2511 GGUF identity, and Z-Image L2P GGUF admission metadata.
Paths config resolution is fail-loud (no silent fallback to defaults on invalid config payloads).
Family hints and root selection cover SD/Flux/Qwen Image/Anima/WAN/ZImage/ZImage L2P/LTX2/Netflix VOID keyspaces while generic VAE inventory excludes audio-bundle files.

Symbols (top-level; keep in sync; no ghosts):
- `_default_models_root` (function): Returns the default `models/` directory under `CODEX_ROOT`.
- `_default_hf_root` (function): Returns the default Hugging Face vendor cache root under `CODEX_ROOT` (when used).
- `_sha256` (function): Computes sha256 digest for a file path.
- `detect_safetensors_primary_dtype` (function): Best-effort safetensors dtype hint reader (header-only parse; used for defaults/telemetry).
- `_detect_sdxl_core_only_checkpoint` (function): Header-only SDXL checkpoint classifier for UNet-only `.safetensors` assets discovered under SDXL roots.
- `_inspect_qwen_image_edit_core_checkpoint` (function): Header-only exact Qwen Image Edit-2511 transformer GGUF inspector.
- `_inspect_zimage_l2p_core_checkpoint` (function): Header-only Z-Image L2P checkpoint inspector for no-VAE pixel DiT SafeTensors/GGUF assets.
- `_detect_zimage_l2p_core_checkpoint` (function): Boolean wrapper around the L2P checkpoint inspector.
- `_HashCacheEntry` (dataclass): Cache entry for one file (sha + mtime + size) used to avoid re-hashing unchanged files.
- `LayoutMetadata` (dataclass): Typed CLIP layout metadata entry (`qkv_layout`, `projection_orientation`, optional `source_style`).
- `_load_hash_cache` (function): Loads `.hashes.json` cache from disk (v1/v2 migration aware).
- `_save_hash_cache` (function): Writes `.hashes.json` cache to disk atomically (v2 schema) and fails loud on persistence errors.
- `ModelRegistry` (class): Registry service; scans paths (fail-loud paths.json resolution + directory/file checkpoint roots), maintains caches, and produces `CheckpointRecord`/`VAERecord` lists for UI/API (also provides public hash-cache helpers).
- `get_registry` (function): Returns the singleton `ModelRegistry` instance.
- `list_checkpoints` (function): Returns checkpoint records (optional refresh).
- `list_vaes` (function): Returns VAE records (optional refresh).
- `refresh` (function): Forces a rescan + cache update for checkpoints/VAEs.
- `invalidate` (function): Clears in-memory checkpoint/VAE scan snapshots (next read lazily rescans).
"""

from __future__ import annotations
from apps.backend.runtime.logging import get_backend_logger

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from apps.backend.infra.config.paths import get_paths_for
from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.runtime import trace as _trace
from apps.backend.runtime.checkpoint.safetensors_header import (
    detect_safetensors_primary_dtype,
    read_safetensors_header,
)
from apps.backend.runtime.model_registry.ltx2_execution import build_ltx2_checkpoint_metadata
from apps.backend.runtime.model_registry.netflix_void_execution import build_netflix_void_checkpoint_metadata

from .types import (
    CheckpointFormat,
    CheckpointRecord,
    VAERecord,
)

_LOGGER = get_backend_logger("backend.registry")

_ALLOWED_CHECKPOINT_EXTS = {".ckpt", ".safetensor", ".safetensors", ".pt", ".pth", ".bin", ".gguf"}
_CHECKPOINT_BLACKLIST_SUFFIXES = {".vae.ckpt", ".vae.safetensor", ".vae.safetensors", ".vae.pt", ".vae.pth", ".vae.bin"}
_VAE_EXTS = {".safetensor", ".safetensors", ".ckpt", ".pt"}
_HASH_CACHE_SCHEMA_VERSION = 2
_LAYOUT_QKV_VALUES = frozenset({"split", "fused"})
_LAYOUT_PROJECTION_VALUES = frozenset({"none", "linear", "matmul"})
_LAYOUT_STYLE_VALUES = frozenset({"codex", "hf", "openclip"})
_CHECKPOINT_ROOT_FAMILY_HINTS: dict[str, str] = {
    "sd15_ckpt": "sd15",
    "sdxl_ckpt": "sdxl",
    "flux1_ckpt": "flux1",
    "flux2_ckpt": "flux2",
    "qwen_image_ckpt": "qwen_image",
    "ltx2_ckpt": "ltx2",
    "netflix_void_ckpt": "netflix_void",
    "anima_ckpt": "anima",
    "wan22_ckpt": "wan22",
    "zimage_ckpt": "zimage",
    "zimage_l2p_ckpt": "zimage_l2p",
}
_DEFAULT_CHECKPOINT_ROOTS: tuple[tuple[str, str], ...] = (
    ("sd15", "sd15"),
    ("sdxl", "sdxl"),
    ("flux", "flux1"),
    ("flux2", "flux2"),
    ("qwen-image", "qwen_image"),
    ("qwen_image", "qwen_image"),
    ("ltx2", "ltx2"),
    ("netflix-void", "netflix_void"),
    ("anima", "anima"),
    ("wan22", "wan22"),
    ("zimage", "zimage"),
    ("zimage-l2p", "zimage_l2p"),
)

def _default_models_root() -> Path:
    return get_repo_root() / "models"


def _default_hf_root() -> Path:
    return get_repo_root() / "apps" / "backend" / "huggingface"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_sdxl_core_only_checkpoint(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in {".safetensor", ".safetensors"}:
        return False
    try:
        header = read_safetensors_header(path)
    except Exception:
        return False
    keys = [key for key in header.keys() if isinstance(key, str) and key != "__metadata__"]
    has_unet = any(key.startswith("model.diffusion_model.") for key in keys)
    has_vae = any(key.startswith("first_stage_model.") or key.startswith("vae.") for key in keys)
    has_text_encoders = any(key.startswith("conditioner.embedders.") for key in keys)
    return has_unet and not has_vae and not has_text_encoders


@dataclass(frozen=True, slots=True)
class _TensorShapeDescriptor:
    shape: tuple[int, ...]


def _inspect_qwen_image_edit_core_checkpoint(path: Path) -> dict[str, object] | None:
    if path.suffix.lower() != ".gguf":
        return None
    try:
        from apps.backend.quantization.gguf import GGUFReader
        from apps.backend.quantization.gguf_loader import get_gguf_metadata
        from apps.backend.runtime.model_registry.detectors.qwen_image import (
            validate_qwen_image_edit_gguf_metadata,
        )
        from apps.backend.runtime.state_dict.keymap_qwen_image_transformer import (
            resolve_qwen_image_edit_transformer_keyspace,
        )

        metadata = dict(get_gguf_metadata(str(path)))
        validate_qwen_image_edit_gguf_metadata(metadata)
        reader = GGUFReader(str(path))
        descriptors = {
            tensor.name: _TensorShapeDescriptor(
                shape=tuple(int(dim) for dim in reversed(tensor.shape.tolist())),
            )
            for tensor in reader.tensors
        }
        resolve_qwen_image_edit_transformer_keyspace(descriptors)
    except Exception:
        return None
    return {
        "requires_vae": True,
        "text_encoder_slots": ["qwen2_5_vl_7b"],
        "qwen_image_variant": "edit_2511",
        "zero_cond_t": True,
        "signature_source": "qwen_image_edit_2511_gguf_profile",
        "codex.quant_policy": str(metadata.get("codex.quant_policy") or ""),
        "codex.quant_policy_preset": str(metadata.get("codex.quant_policy_preset") or ""),
        "model.architecture": str(metadata.get("model.architecture") or ""),
        "model.name": str(metadata.get("model.name") or ""),
    }


def _inspect_zimage_l2p_core_checkpoint(path: Path) -> dict[str, object] | None:
    suffix = path.suffix.lower()
    if suffix not in {".safetensor", ".safetensors", ".gguf"}:
        return None
    metadata: dict[str, object] = {}
    if suffix == ".gguf":
        try:
            from apps.backend.quantization.gguf_loader import get_gguf_metadata
            from apps.backend.quantization.gguf import GGUFReader
            from apps.backend.runtime.model_registry.detectors.zimage_l2p import (
                validate_zimage_l2p_gguf_component_metadata,
            )

            metadata = dict(get_gguf_metadata(str(path)))
            validate_zimage_l2p_gguf_component_metadata(metadata, component="denoiser")
            reader = GGUFReader(str(path))
            keys = [str(tensor.name) for tensor in reader.tensors]
        except Exception:
            return None
    else:
        try:
            header = read_safetensors_header(path)
        except Exception:
            return None
        keys = [key for key in header.keys() if isinstance(key, str) and key != "__metadata__"]
    required = (
        "all_x_embedder.16-1.weight",
        "local_decoder.out_conv.weight",
        "layers.0.adaLN_modulation.0.weight",
        "noise_refiner.0.adaLN_modulation.0.weight",
        "cap_embedder.1.weight",
    )
    has_core = all(key in keys for key in required)
    has_vae = any(key.startswith("vae.") or key.startswith("first_stage_model.") for key in keys)
    has_text_encoder = any(key.startswith("text_encoder.") or key.startswith("model.embed_tokens.") for key in keys)
    has_latent_zimage_final = "final_layer.linear.weight" in keys
    if not has_core or has_vae or has_text_encoder or has_latent_zimage_final:
        return None
    return {
        "requires_vae": False,
        "text_encoder_slots": ["qwen3_4b"],
        "signature_source": "zimage_l2p_gguf_profile" if suffix == ".gguf" else "zimage_l2p_safetensors_header",
        **(
            {
                "codex.zimage_l2p.profile_id": str(metadata.get("codex.zimage_l2p.profile_id") or ""),
                "codex.zimage_l2p.component": str(metadata.get("codex.zimage_l2p.component") or ""),
                "codex.zimage_l2p.family": str(metadata.get("codex.zimage_l2p.family") or ""),
                "codex.zimage_l2p.pixel_space": bool(metadata.get("codex.zimage_l2p.pixel_space") is True),
                "codex.zimage_l2p.requires_vae": bool(metadata.get("codex.zimage_l2p.requires_vae") is True),
                "model.architecture": str(metadata.get("model.architecture") or ""),
                "codex.zimage_l2p.local_decoder": bool(
                    metadata.get("codex.zimage_l2p.local_decoder") is True
                ),
                "codex.zimage_l2p.context_dim": metadata.get("codex.zimage_l2p.context_dim"),
            }
            if suffix == ".gguf"
            else {}
        ),
    }


def _detect_zimage_l2p_core_checkpoint(path: Path) -> bool:
    return _inspect_zimage_l2p_core_checkpoint(path) is not None


@dataclass
class _HashCacheEntry:
    mtime: float
    size: int  # file size for extra validation
    sha256: str
    short_hash: str
    dtype: str | None = None


@dataclass(frozen=True, slots=True)
class LayoutMetadata:
    qkv_layout: str
    projection_orientation: str
    source_style: str | None = None


# Persistent hash cache file location (under models/)
_HASH_CACHE_FILE = _default_models_root() / ".hashes.json"


def _parse_file_cache_entry(*, path: str, raw: Mapping[str, Any]) -> _HashCacheEntry:
    return _HashCacheEntry(
        mtime=float(raw.get("mtime", 0)),
        size=int(raw.get("size", 0)),
        sha256=str(raw.get("sha256", "")),
        short_hash=str(raw.get("short_hash", "")),
        dtype=(str(raw.get("dtype", "")).strip() or None),
    )


def _parse_layout_metadata(*, sha256: str, layout_key: str, raw: Mapping[str, Any]) -> LayoutMetadata:
    qkv_layout = str(raw.get("qkv_layout", "")).strip().lower()
    projection_orientation = str(raw.get("projection_orientation", "")).strip().lower()
    source_style_raw = raw.get("source_style")
    source_style = None if source_style_raw is None else str(source_style_raw).strip().lower()
    if qkv_layout not in _LAYOUT_QKV_VALUES:
        allowed = ", ".join(sorted(_LAYOUT_QKV_VALUES))
        raise RuntimeError(
            f"Invalid layout metadata qkv_layout for sha={sha256} key={layout_key!r}: "
            f"{qkv_layout!r} (allowed: {allowed})"
        )
    if projection_orientation not in _LAYOUT_PROJECTION_VALUES:
        allowed = ", ".join(sorted(_LAYOUT_PROJECTION_VALUES))
        raise RuntimeError(
            f"Invalid layout metadata projection_orientation for sha={sha256} key={layout_key!r}: "
            f"{projection_orientation!r} (allowed: {allowed})"
        )
    if source_style and source_style not in _LAYOUT_STYLE_VALUES:
        allowed = ", ".join(sorted(_LAYOUT_STYLE_VALUES))
        raise RuntimeError(
            f"Invalid layout metadata source_style for sha={sha256} key={layout_key!r}: "
            f"{source_style!r} (allowed: {allowed})"
        )
    return LayoutMetadata(
        qkv_layout=qkv_layout,
        projection_orientation=projection_orientation,
        source_style=source_style or None,
    )


def _serialize_layout_metadata(metadata: LayoutMetadata) -> dict[str, str]:
    payload = {
        "qkv_layout": metadata.qkv_layout,
        "projection_orientation": metadata.projection_orientation,
    }
    if metadata.source_style:
        payload["source_style"] = metadata.source_style
    return payload


def _looks_like_v1_file_cache(raw: Mapping[str, Any]) -> bool:
    if not raw:
        return True
    for value in raw.values():
        if not isinstance(value, Mapping):
            return False
        required = {"mtime", "size", "sha256", "short_hash"}
        if not required.issubset(value.keys()):
            return False
    return True


def _load_hash_cache() -> tuple[Dict[str, _HashCacheEntry], Dict[str, Dict[str, LayoutMetadata]]]:
    """Load persistent hash/layout cache from disk."""
    file_cache: Dict[str, _HashCacheEntry] = {}
    layout_cache: Dict[str, Dict[str, LayoutMetadata]] = {}
    _LOGGER.info("loading hash cache from %s", _HASH_CACHE_FILE)
    try:
        if not _HASH_CACHE_FILE.is_file():
            _LOGGER.info("hash cache not found, will compute hashes on first scan")
            return file_cache, layout_cache
        with _HASH_CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            raise RuntimeError("hash cache root must be a JSON object")

        if "schema_version" not in data:
            if not _looks_like_v1_file_cache(data):
                raise RuntimeError(
                    "legacy hash cache payload is malformed (expected path->entry mapping with mtime/size/sha256/short_hash)"
                )
            for path, entry in data.items():
                if not isinstance(path, str) or not isinstance(entry, Mapping):
                    continue
                file_cache[path] = _parse_file_cache_entry(path=path, raw=entry)
            _LOGGER.info("hash cache loaded (v1): %d file entries", len(file_cache))
            return file_cache, layout_cache

        schema_version = int(data.get("schema_version"))
        if schema_version != _HASH_CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported cache schema version: {schema_version} (expected {_HASH_CACHE_SCHEMA_VERSION})"
            )
        files_raw = data.get("files", {})
        if not isinstance(files_raw, Mapping):
            raise RuntimeError("hash cache schema v2 requires object field 'files'")
        for path, entry in files_raw.items():
            if not isinstance(path, str) or not isinstance(entry, Mapping):
                continue
            file_cache[path] = _parse_file_cache_entry(path=path, raw=entry)

        layouts_raw = data.get("layout_by_sha", {})
        if not isinstance(layouts_raw, Mapping):
            raise RuntimeError("hash cache schema v2 requires object field 'layout_by_sha'")
        for sha256, by_key in layouts_raw.items():
            if not isinstance(sha256, str):
                continue
            sha_norm = sha256.strip().lower()
            if not sha_norm:
                continue
            if not isinstance(by_key, Mapping):
                raise RuntimeError(f"layout_by_sha[{sha_norm!r}] must be an object")
            parsed: Dict[str, LayoutMetadata] = {}
            for layout_key, raw_metadata in by_key.items():
                if not isinstance(layout_key, str):
                    continue
                if not isinstance(raw_metadata, Mapping):
                    raise RuntimeError(
                        f"layout metadata for sha={sha_norm} key={layout_key!r} must be an object"
                    )
                parsed[layout_key] = _parse_layout_metadata(
                    sha256=sha_norm,
                    layout_key=layout_key,
                    raw=raw_metadata,
                )
            if parsed:
                layout_cache[sha_norm] = parsed
        _LOGGER.info(
            "hash cache loaded (v2): file_entries=%d layout_entries=%d",
            len(file_cache),
            sum(len(v) for v in layout_cache.values()),
        )
    except Exception as exc:
        raise RuntimeError(f"hash cache load failed: {exc}") from exc
    return file_cache, layout_cache


def _save_hash_cache(
    file_cache: Dict[str, _HashCacheEntry],
    layout_cache: Dict[str, Dict[str, LayoutMetadata]],
) -> None:
    """Persist hash/layout cache to disk (schema v2)."""
    payload = {
        "schema_version": _HASH_CACHE_SCHEMA_VERSION,
        "files": {
            path: {
                "mtime": entry.mtime,
                "size": entry.size,
                "sha256": entry.sha256,
                "short_hash": entry.short_hash,
                "dtype": entry.dtype,
            }
            for path, entry in file_cache.items()
        },
        "layout_by_sha": {
            sha256: {
                layout_key: _serialize_layout_metadata(metadata)
                for layout_key, metadata in by_key.items()
            }
            for sha256, by_key in layout_cache.items()
        },
    }
    target = _HASH_CACHE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f"{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_handle:
            temp_path = temp_handle.name
            json.dump(payload, temp_handle, indent=2)
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
        os.replace(temp_path, target)
        temp_path = None
        try:
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except Exception as exc:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise RuntimeError(f"hash cache save failed: {exc}") from exc


class ModelRegistry:
    """Discover checkpoint/VAE assets and expose cached views."""

    def __init__(self, *, models_root: Path | None = None, hf_root: Path | None = None) -> None:
        self._models_root = Path(models_root or _default_models_root()).resolve()
        self._hf_root = Path(hf_root or _default_hf_root()).resolve()
        self._lock = threading.Lock()
        self._checkpoints: Dict[str, CheckpointRecord] = {}
        self._vaes: Dict[str, VAERecord] = {}
        self._hash_cache: Dict[str, _HashCacheEntry] | None = None  # Lazy load
        self._layout_cache: Dict[str, Dict[str, LayoutMetadata]] | None = None  # Lazy load
        self._hash_cache_dirty = False  # Track if we need to save
        self._last_scan: float | None = None

    def _ensure_hash_cache(self) -> Dict[str, _HashCacheEntry]:
        """Lazy load hash cache on first access."""
        if self._hash_cache is None:
            self._hash_cache, self._layout_cache = _load_hash_cache()
        return self._hash_cache

    def _ensure_layout_cache(self) -> Dict[str, Dict[str, LayoutMetadata]]:
        self._ensure_hash_cache()
        if self._layout_cache is None:
            self._layout_cache = {}
        return self._layout_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def hash_for(self, path: Path) -> Tuple[str | None, str | None]:
        """Return (sha256, short_hash) for a file path, using the persistent hash cache.

        This is safe to call outside registry scans (it is lock-protected) and is the
        supported way for other subsystems (e.g. inventory) to request hashes without
        reaching into private internals.
        """

        with self._lock:
            return self._hash_for(path)

    def flush_hash_cache(self) -> None:
        """Persist the hash cache to disk if any new hashes were computed."""

        with self._lock:
            if self._hash_cache_dirty:
                _save_hash_cache(self._ensure_hash_cache(), self._ensure_layout_cache())
                self._hash_cache_dirty = False

    def get_layout_metadata(self, *, sha256: str, layout_key: str) -> LayoutMetadata | None:
        sha = str(sha256).strip().lower()
        key = str(layout_key).strip()
        if not sha:
            raise ValueError("sha256 must be a non-empty string")
        if not key:
            raise ValueError("layout_key must be a non-empty string")
        with self._lock:
            by_sha = self._ensure_layout_cache().get(sha)
            if by_sha is None:
                return None
            return by_sha.get(key)

    def set_layout_metadata(
        self,
        *,
        sha256: str,
        layout_key: str,
        metadata: LayoutMetadata,
    ) -> None:
        sha = str(sha256).strip().lower()
        key = str(layout_key).strip()
        if not sha:
            raise ValueError("sha256 must be a non-empty string")
        if not key:
            raise ValueError("layout_key must be a non-empty string")
        if metadata.qkv_layout not in _LAYOUT_QKV_VALUES:
            raise ValueError(f"invalid qkv_layout: {metadata.qkv_layout!r}")
        if metadata.projection_orientation not in _LAYOUT_PROJECTION_VALUES:
            raise ValueError(f"invalid projection_orientation: {metadata.projection_orientation!r}")
        if metadata.source_style is not None and metadata.source_style not in _LAYOUT_STYLE_VALUES:
            raise ValueError(f"invalid source_style: {metadata.source_style!r}")
        with self._lock:
            cache = self._ensure_layout_cache()
            by_sha = cache.setdefault(sha, {})
            previous = by_sha.get(key)
            if previous is not None and previous != metadata:
                raise RuntimeError(
                    "layout metadata conflict for sha=%s key=%s: existing=%s incoming=%s"
                    % (sha, key, previous, metadata)
                )
            if previous != metadata:
                by_sha[key] = metadata
                self._hash_cache_dirty = True

    def list_checkpoints(self, *, refresh: bool = False) -> List[CheckpointRecord]:
        if refresh:
            self.refresh()
        with self._lock:
            if not self._checkpoints:
                self._scan_locked()
            return list(self._checkpoints.values())

    def list_vaes(self, *, refresh: bool = False) -> List[VAERecord]:
        if refresh:
            self.refresh()
        with self._lock:
            if not self._vaes:
                self._scan_locked()
            return list(self._vaes.values())

    def refresh(self) -> None:
        with self._lock:
            self._scan_locked()

    def invalidate(self) -> None:
        with self._lock:
            self._checkpoints = {}
            self._vaes = {}
            self._last_scan = 0.0

    def get_checkpoint(self, name: str) -> CheckpointRecord | None:
        with self._lock:
            if not self._checkpoints:
                self._scan_locked()
            return self._checkpoints.get(name)

    def get_vae(self, name: str) -> VAERecord | None:
        with self._lock:
            if not self._vaes:
                self._scan_locked()
            return self._vaes.get(name)

    # ------------------------------------------------------------------
    # Internal scanning helpers
    # ------------------------------------------------------------------
    def _scan_locked(self) -> None:
        start = time.perf_counter()
        checkpoints = {rec.name: rec for rec in self._scan_checkpoints()}
        vaes = {rec.name: rec for rec in self._scan_vaes()}
        duration_ms = (time.perf_counter() - start) * 1000.0
        self._checkpoints = checkpoints
        self._vaes = vaes
        self._last_scan = time.time()
        # Persist hash cache if we computed any new hashes
        if self._hash_cache_dirty:
            _save_hash_cache(self._ensure_hash_cache(), self._ensure_layout_cache())
            self._hash_cache_dirty = False
        _LOGGER.info(
            "model_registry: scan complete checkpoints=%d vaes=%d ms=%.1f",
            len(checkpoints),
            len(vaes),
            duration_ms,
        )
        _trace.event("model_registry_scan", checkpoints=len(checkpoints), vaes=len(vaes), ms=f"{duration_ms:.2f}")

    def _scan_checkpoints(self) -> Iterable[CheckpointRecord]:
        seen: set[str] = set()
        for file, source_family_hint in self._iter_checkpoint_files():
            path_str = str(file.resolve())
            if path_str in seen:
                continue
            seen.add(path_str)
            suffix = file.suffix.lower()
            gguf_by_suffix = suffix == ".gguf"
            gguf_by_magic = False
            if not gguf_by_suffix:
                try:
                    with file.open("rb") as handle:
                        gguf_by_magic = handle.read(4) == b"GGUF"
                except Exception:
                    gguf_by_magic = False
            is_gguf = gguf_by_suffix or gguf_by_magic

            fmt = CheckpointFormat.GGUF if is_gguf else CheckpointFormat.CHECKPOINT
            core_only = is_gguf
            core_only_reason = "gguf_suffix" if gguf_by_suffix else ("gguf_magic" if gguf_by_magic else None)
            family_hint = source_family_hint
            if family_hint is None:
                try:
                    rel = file.resolve().relative_to(self._models_root)
                except Exception:
                    rel = None
                if rel and rel.parts:
                    top = str(rel.parts[0]).lower()
                    family_hint = {
                        "sd15": "sd15",
                        "sd1": "sd15",
                        "sdxl": "sdxl",
                        "flux": "flux1",
                        "flux2": "flux2",
                        "ltx2": "ltx2",
                        "qwen-image": "qwen_image",
                        "qwen_image": "qwen_image",
                        "netflix-void": "netflix_void",
                        "netflix_void": "netflix_void",
                        "zimage": "zimage",
                        "zimage-l2p": "zimage_l2p",
                        "zimage_l2p": "zimage_l2p",
                        "anima": "anima",
                        "wan22": "wan22",
                    }.get(top)
            if not core_only and family_hint == "sdxl" and _detect_sdxl_core_only_checkpoint(file):
                core_only = True
                core_only_reason = "sdxl_header_unet_only"
            qwen_image_metadata: dict[str, object] = {}
            if family_hint == "qwen_image":
                inspected_qwen_metadata = _inspect_qwen_image_edit_core_checkpoint(file)
                if inspected_qwen_metadata is None:
                    continue
                qwen_image_metadata = inspected_qwen_metadata
                core_only = True
                core_only_reason = "qwen_image_edit_2511_gguf_profile_core_only"
            zimage_l2p_metadata: dict[str, object] = {}
            if family_hint == "zimage_l2p":
                inspected_l2p_metadata = _inspect_zimage_l2p_core_checkpoint(file)
                if inspected_l2p_metadata is None:
                    continue
                zimage_l2p_metadata = inspected_l2p_metadata
                core_only = True
                core_only_reason = f"{str(zimage_l2p_metadata['signature_source'])}_core_only"
            sha256, short_hash = self._hash_for(file)
            stat = file.stat()
            metadata: dict[str, object] = {}
            if family_hint == "ltx2":
                metadata.update(
                    build_ltx2_checkpoint_metadata(
                        CheckpointRecord(
                            name=file.stem,
                            title=file.name,
                            filename=str(file),
                            path=str(file.parent),
                            model_name=file.stem,
                            format=fmt,
                            family_hint=family_hint,
                        )
                    )
                )
            if family_hint == "netflix_void":
                metadata.update(
                    build_netflix_void_checkpoint_metadata(
                        CheckpointRecord(
                            name=file.stem,
                            title=file.name,
                            filename=str(file),
                            path=str(file.parent),
                            model_name=file.stem,
                            format=fmt,
                            family_hint=family_hint,
                        )
                    )
                )
            record = CheckpointRecord(
                name=file.stem,
                title=file.name,
                filename=str(file),
                path=str(file.parent),
                model_name=file.stem,
                format=fmt,
                core_only=core_only,
                core_only_reason=core_only_reason,
                family_hint=family_hint,
                sha256=sha256,
                short_hash=short_hash,
                file_size=stat.st_size,
                metadata={
                    **metadata,
                    **qwen_image_metadata,
                    **zimage_l2p_metadata,
                },
                updated_at=stat.st_mtime,
            )
            yield record

    def _scan_vaes(self) -> Iterable[VAERecord]:
        candidates: List[Path] = []
        for key in (
            "sd15_vae",
            "sdxl_vae",
            "flux1_vae",
            "flux2_vae",
            "qwen_image_vae",
            "ltx2_vae",
            "anima_vae",
            "wan22_vae",
            "zimage_vae",
        ):
            for raw in get_paths_for(key):
                p = Path(raw)
                if p not in candidates:
                    candidates.append(p)

        files: List[Path] = []
        for root in candidates:
            if root.is_file() and root.suffix.lower() in _VAE_EXTS:
                if "audio_vae" not in root.name.lower():
                    files.append(root)
            elif root.is_dir():
                try:
                    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
                        if (
                            path.is_file()
                            and path.suffix.lower() in _VAE_EXTS
                            and "audio_vae" not in path.name.lower()
                        ):
                            files.append(path)
                except Exception:
                    continue

        seen: set[str] = set()
        for path in files:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            sha256, short_hash = self._hash_for(path)
            stat = path.stat()
            yield VAERecord(
                name=path.name,
                filename=str(path),
                source=str(path.parent),
                sha256=sha256,
                short_hash=short_hash,
                updated_at=stat.st_mtime,
            )

    def _iter_checkpoint_files(self) -> Iterable[tuple[Path, str | None]]:
        """Iterate over checkpoint files using paths.json overrides + curated defaults.

        Resolution order:
        1) Explicit roots from apps/paths.json per engine (sd15_ckpt, sdxl_ckpt, flux1_ckpt, flux2_ckpt, qwen_image_ckpt, ltx2_ckpt, wan22_ckpt).
           Entries may be directories (recursive scan) or individual files.
        2) Built-in defaults under models/: per-engine folders only (sd15, sdxl, flux, flux2, qwen-image, ltx2, anima, wan22, zimage).

        This replaces the legacy scatter of ad-hoc checkpoint folders ('stable-diffusion', 'sd', 'checkpoints').
        """
        candidates: List[tuple[Path, str | None]] = []

        # 1) User overrides from apps/paths.json per engine.
        # Fail loud when paths config cannot be resolved (no silent fallback).
        for key in _CHECKPOINT_ROOT_FAMILY_HINTS:
            for raw in get_paths_for(key):
                p = Path(raw)
                candidate = (p, _CHECKPOINT_ROOT_FAMILY_HINTS[key])
                if candidate not in candidates:
                    candidates.append(candidate)

        # 2) Curated built-in defaults quando não há overrides configurados.
        # Keep these per-engine only (never scan models/ root directly).
        if not candidates:
            defaults = [
                (self._models_root / directory_name, family_hint)
                for directory_name, family_hint in _DEFAULT_CHECKPOINT_ROOTS
            ]
            for candidate in defaults:
                if candidate not in candidates:
                    candidates.append(candidate)

        for directory, family_hint in candidates:
            if directory.is_file():
                suffix = directory.suffix.lower()
                if suffix not in _ALLOWED_CHECKPOINT_EXTS:
                    continue
                lower = directory.name.lower()
                if any(lower.endswith(suf) for suf in _CHECKPOINT_BLACKLIST_SUFFIXES):
                    continue
                yield directory, family_hint
                continue
            if not directory.is_dir():
                continue
            for entry in directory.rglob("*"):
                if not entry.is_file():
                    continue
                suffix = entry.suffix.lower()
                if suffix not in _ALLOWED_CHECKPOINT_EXTS:
                    continue
                lower = entry.name.lower()
                if any(lower.endswith(suf) for suf in _CHECKPOINT_BLACKLIST_SUFFIXES):
                    continue
                yield entry, family_hint

    def _hash_for(self, path: Path) -> Tuple[str | None, str | None]:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None, None
        key = str(path)
        cache = self._ensure_hash_cache()
        entry = cache.get(key)
        # Cache hit: validate by mtime AND size (both must match)
        if entry and entry.mtime == stat.st_mtime and entry.size == stat.st_size:
            if entry.dtype is None:
                dtype = detect_safetensors_primary_dtype(path)
                if dtype:
                    entry.dtype = dtype
                    self._hash_cache_dirty = True
            sha256 = entry.sha256
            short_hash = entry.short_hash or None
            return sha256, short_hash
        # Cache miss: compute hash (slow path, but only happens once per file)
        try:
            _LOGGER.debug("computing sha256 for %s (%.1f MB)", path.name, stat.st_size / 1e6)
            sha256 = _sha256(path)
            short_hash = sha256[:10]
        except Exception:
            sha256 = None
            short_hash = None
        if sha256:
            dtype = detect_safetensors_primary_dtype(path)
            cache[key] = _HashCacheEntry(stat.st_mtime, stat.st_size, sha256, short_hash or "", dtype=dtype)
            self._hash_cache_dirty = True  # Mark for persistence
        return sha256, short_hash


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _DEFAULT_REGISTRY


def list_checkpoints(*, refresh: bool = False) -> List[CheckpointRecord]:
    return _DEFAULT_REGISTRY.list_checkpoints(refresh=refresh)


def list_vaes(*, refresh: bool = False) -> List[VAERecord]:
    return _DEFAULT_REGISTRY.list_vaes(refresh=refresh)


def refresh() -> None:
    _DEFAULT_REGISTRY.refresh()


def invalidate() -> None:
    _DEFAULT_REGISTRY.invalidate()


__all__ = [
    "LayoutMetadata",
    "ModelRegistry",
    "get_registry",
    "invalidate",
    "list_checkpoints",
    "list_vaes",
    "refresh",
]

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend-owned engine dependency check contract for WebUI readiness surfaces.
Builds deterministic per-engine check rows from backend inventory/model-registry state so the frontend can render a strict
"Dependency Check" panel and disable generation when required assets are missing. Semantic-engine asset checks resolve through the
canonical contract owner seam (`contract_owner_for_semantic_engine`) to prevent drift between API surfaces, including Qwen Image
Edit-2511 vendored processor/config readiness, Z-Image L2P scoped split-asset roots, the
vendored LTX2 metadata/config readiness required by explicit execution profiles and the explicit Netflix VOID base-bundle +
literal overlay-pair readiness contract. Mode-scoped rows can now report exact masked-runtime readiness (for example SDXL `fooocus_inpaint`)
without making the whole semantic engine globally unready.

Symbols (top-level; keep in sync; no ghosts):
- `DependencyCheckRow` (dataclass): One backend dependency row (`id/label/ok/message`, with optional `inpaint_modes` scope) rendered by the frontend.
- `EngineDependencyStatus` (dataclass): Aggregated dependency status for one semantic engine (`ready + checks`).
- `_qwen_image_vendored_metadata_check` (function): Validate the complete offline Edit-2511 processor/tokenizer/component metadata set.
- `build_engine_dependency_checks` (function): Build per-engine dependency status map for `/api/engines/capabilities`.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from apps.backend.core.contracts.asset_requirements import (
    contract_for_core_only,
    contract_for_engine,
    contract_owner_for_semantic_engine,
)
from apps.backend.infra.config.paths import get_paths_for
from apps.backend.inventory import cache as inventory_cache
from apps.backend.runtime.families.sd.brushnet import resolve_brushnet_assets
from apps.backend.runtime.families.sd.fooocus_inpaint import resolve_fooocus_inpaint_assets
from apps.backend.runtime.families.ltx2.config import LTX2_VENDOR_REPO_ID, resolve_ltx2_vendor_paths
from apps.backend.runtime.families.qwen_image.config import (
    QWEN_IMAGE_EDIT_PIPELINE_CLASS,
    QWEN_IMAGE_EDIT_VARIANT,
)
from apps.backend.runtime.families.qwen_image.scheduler import qwen_image_scheduler_config_from_mapping
from apps.backend.runtime.families.qwen_image.text_encoder import qwen_image_text_encoder_config_from_mapping
from apps.backend.runtime.families.qwen_image.transformer import qwen_image_transformer_config_from_mapping
from apps.backend.runtime.families.qwen_image.vae import qwen_image_vae_config_from_mapping
from apps.backend.runtime.families.netflix_void.loader import resolve_netflix_void_base_dirs
from apps.backend.runtime.model_registry.netflix_void_execution import (
    NETFLIX_VOID_KIND_PASS2,
    NETFLIX_VOID_KIND_UNKNOWN,
    netflix_void_checkpoint_kind,
    netflix_void_record_is_publicly_runnable,
    netflix_void_record_is_publicly_selectable,
)


@dataclass(frozen=True, slots=True)
class DependencyCheckRow:
    """One backend dependency check row shown in the frontend panel."""

    id: str
    label: str
    ok: bool
    message: str
    inpaint_modes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "label": str(self.label),
            "ok": bool(self.ok),
            "message": str(self.message),
            "inpaint_modes": [str(mode) for mode in self.inpaint_modes],
        }


@dataclass(frozen=True, slots=True)
class EngineDependencyStatus:
    """Aggregated dependency checks for one semantic engine surface."""

    ready: bool
    checks: tuple[DependencyCheckRow, ...]

    @classmethod
    def from_checks(cls, checks: Iterable[DependencyCheckRow]) -> "EngineDependencyStatus":
        rows = tuple(checks)
        return cls(
            ready=all(bool(row.ok) for row in rows if len(row.inpaint_modes) == 0),
            checks=rows,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": bool(self.ready),
            "checks": [row.as_dict() for row in self.checks],
        }


_CHECKPOINT_REQUIRED_ENGINES: frozenset[str] = frozenset(
    {
        "sd15",
        "sdxl",
        "flux1",
        "flux2",
        "qwen_image",
        "chroma",
        "zimage",
        "zimage_l2p",
        "anima",
        "ltx2",
        "netflix_void",
        "svd",
        "hunyuan_video",
    }
)
_BACKEND_ROOT = Path(__file__).resolve().parents[2]

_WAN_METADATA_PREFIX = "wan-ai/wan2.2-"

_CHECKPOINT_ROOT_KEYS_BY_ENGINE: dict[str, tuple[str, ...]] = {
    "sd15": ("sd15_ckpt",),
    "sdxl": ("sdxl_ckpt",),
    "flux1": ("flux1_ckpt",),
    "flux2": ("flux2_ckpt",),
    "qwen_image": ("qwen_image_ckpt",),
    "chroma": ("flux1_ckpt",),
    "zimage": ("zimage_ckpt",),
    "zimage_l2p": ("zimage_l2p_ckpt",),
    "anima": ("anima_ckpt",),
    "wan22": ("wan22_ckpt",),
    "ltx2": ("ltx2_ckpt",),
    "netflix_void": ("netflix_void_ckpt",),
}

_CHECKPOINT_FAMILY_HINTS_BY_ENGINE: dict[str, tuple[str, ...]] = {
    "sd15": ("sd15",),
    "sdxl": ("sdxl",),
    "flux1": ("flux1",),
    "flux2": ("flux2",),
    "qwen_image": ("qwen_image",),
    "chroma": ("chroma", "flux1"),
    "zimage": ("zimage",),
    "zimage_l2p": ("zimage_l2p",),
    "anima": ("anima",),
    "wan22": ("wan22",),
    "ltx2": ("ltx2",),
    "netflix_void": ("netflix_void",),
}

_VAE_ROOT_KEYS_BY_CONTRACT_OWNER: dict[str, tuple[str, ...]] = {
    "flux1": ("flux1_vae",),
    "flux2": ("flux2_vae",),
    "qwen_image": ("qwen_image_vae",),
    "zimage": ("zimage_vae", "flux1_vae"),
    "anima": ("anima_vae",),
    "wan22_5b": ("wan22_vae",),
    "wan22_14b": ("wan22_vae",),
    "wan22_14b_animate": ("wan22_vae",),
    "ltx2": ("ltx2_vae",),
}

_TEXT_ENCODER_ROOT_KEYS_BY_CONTRACT_OWNER: dict[str, tuple[str, ...]] = {
    "flux1": ("flux1_tenc",),
    "flux2": ("flux2_tenc",),
    "qwen_image": ("qwen_image_tenc",),
    "zimage": ("zimage_tenc",),
    "zimage_l2p": ("zimage_tenc",),
    "anima": ("anima_tenc",),
    "wan22_5b": ("wan22_tenc",),
    "wan22_14b": ("wan22_tenc",),
    "wan22_14b_animate": ("wan22_tenc",),
    "ltx2": ("ltx2_tenc",),
}

def _count_list_entries(value: object) -> int:
    if not isinstance(value, list):
        return 0
    return sum(1 for item in value if isinstance(item, dict))


def _count_wan_metadata_repos(value: object) -> int:
    if not isinstance(value, list):
        return 0
    count = 0
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().lower()
        if name.startswith(_WAN_METADATA_PREFIX):
            count += 1
    return count


def _ltx2_vendored_metadata_check() -> DependencyCheckRow:
    try:
        vendor_paths = resolve_ltx2_vendor_paths(
            backend_root=_BACKEND_ROOT,
            repo_id=LTX2_VENDOR_REPO_ID,
        )
    except Exception as exc:
        return DependencyCheckRow(
            id="vendored_metadata",
            label="Vendored Metadata",
            ok=False,
            message=str(exc),
        )

    return DependencyCheckRow(
        id="vendored_metadata",
        label="Vendored Metadata",
        ok=True,
        message=(
            "LTX2 vendored runtime metadata ready: "
            f"model_index={vendor_paths.model_index_path}, "
            f"tokenizer={vendor_paths.tokenizer_dir}, "
            f"connectors_config={vendor_paths.connectors_config_path}, "
            "component_configs=text_encoder|scheduler|connectors|transformer|vae|audio_vae|vocoder. "
            "The explicit two_stage lane additionally requires latent_upsampler/config.json."
        ),
    )


def _qwen_image_vendored_metadata_check() -> DependencyCheckRow:
    metadata_root = _BACKEND_ROOT / "huggingface" / "Qwen" / "Qwen-Image-Edit-2511"
    required_files = (
        "model_index.json",
        "processor/added_tokens.json",
        "processor/chat_template.jinja",
        "processor/merges.txt",
        "processor/preprocessor_config.json",
        "processor/special_tokens_map.json",
        "processor/tokenizer.json",
        "processor/tokenizer_config.json",
        "processor/video_preprocessor_config.json",
        "processor/vocab.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/added_tokens.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
    )
    missing = [relative for relative in required_files if not (metadata_root / relative).is_file()]
    if missing:
        return DependencyCheckRow(
            id="vendored_metadata",
            label="Vendored Metadata",
            ok=False,
            message=(
                "Qwen Image Edit-2511 vendored metadata is incomplete under "
                f"{metadata_root}: missing={missing}."
            ),
        )

    def _read_object(relative: str) -> Mapping[str, object]:
        path = metadata_root / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"{path} must contain a JSON object.")
        return payload

    try:
        model_index = _read_object("model_index.json")
        if str(model_index.get("_class_name") or "").strip() != QWEN_IMAGE_EDIT_PIPELINE_CLASS:
            raise RuntimeError(
                "model_index.json must declare "
                f"_class_name={QWEN_IMAGE_EDIT_PIPELINE_CLASS!r}."
            )
        expected_components = {
            "processor": ("transformers", "Qwen2VLProcessor"),
            "scheduler": ("diffusers", "FlowMatchEulerDiscreteScheduler"),
            "text_encoder": ("transformers", "Qwen2_5_VLForConditionalGeneration"),
            "tokenizer": ("transformers", "Qwen2Tokenizer"),
            "transformer": ("diffusers", "QwenImageTransformer2DModel"),
            "vae": ("diffusers", "AutoencoderKLQwenImage"),
        }
        for component, expected in expected_components.items():
            raw = model_index.get(component)
            actual = tuple(raw) if isinstance(raw, list) else ()
            if actual != expected:
                raise RuntimeError(
                    f"model_index.json component {component!r} mismatch: got={actual!r} expected={expected!r}."
                )

        qwen_image_transformer_config_from_mapping(
            _read_object("transformer/config.json"),
            variant=QWEN_IMAGE_EDIT_VARIANT,
            context=str(metadata_root / "transformer/config.json"),
        )
        qwen_image_text_encoder_config_from_mapping(
            _read_object("text_encoder/config.json"),
            context=str(metadata_root / "text_encoder/config.json"),
        )
        qwen_image_vae_config_from_mapping(
            _read_object("vae/config.json"),
            context=str(metadata_root / "vae/config.json"),
        )
        qwen_image_scheduler_config_from_mapping(
            _read_object("scheduler/scheduler_config.json"),
            context=str(metadata_root / "scheduler/scheduler_config.json"),
        )

        processor_config = _read_object("processor/preprocessor_config.json")
        expected_processor_fields = {
            "image_processor_type": "Qwen2VLImageProcessorFast",
            "processor_class": "Qwen2VLProcessor",
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "min_pixels": 3136,
            "max_pixels": 12845056,
        }
        for field, expected in expected_processor_fields.items():
            actual = processor_config.get(field)
            if actual != expected:
                raise RuntimeError(
                    f"processor/preprocessor_config.json field {field!r} mismatch: "
                    f"got={actual!r} expected={expected!r}."
                )
        tokenizer_config = _read_object("processor/tokenizer_config.json")
        expected_tokenizer_fields = {
            "processor_class": "Qwen2VLProcessor",
            "tokenizer_class": "Qwen2Tokenizer",
            "model_max_length": 131072,
        }
        for field, expected in expected_tokenizer_fields.items():
            actual = tokenizer_config.get(field)
            if actual != expected:
                raise RuntimeError(
                    f"processor/tokenizer_config.json field {field!r} mismatch: "
                    f"got={actual!r} expected={expected!r}."
                )
    except Exception as exc:
        return DependencyCheckRow(
            id="vendored_metadata",
            label="Vendored Metadata",
            ok=False,
            message=f"Qwen Image Edit-2511 vendored metadata validation failed: {exc}",
        )

    return DependencyCheckRow(
        id="vendored_metadata",
        label="Vendored Metadata",
        ok=True,
        message=(
            "Qwen Image Edit-2511 offline metadata ready: exact model_index, processor/tokenizer assets, "
            "Qwen2.5-VL config, 60-block transformer config, VAE config, and FlowMatch scheduler config."
        ),
    )


def _netflix_void_warped_noise_runtime_check() -> DependencyCheckRow:
    missing_modules: list[str] = []
    for module_name in ("torch", "torchvision"):
        try:
            importlib.import_module(module_name)
        except Exception:
            missing_modules.append(module_name)
    if missing_modules:
        return DependencyCheckRow(
            id="warped_noise_runtime",
            label="Warped-Noise Runtime",
            ok=False,
            message=(
                "Netflix VOID warped-noise runtime is unavailable because required Python modules failed to import: "
                f"{', '.join(missing_modules)}."
            ),
        )
    return DependencyCheckRow(
        id="warped_noise_runtime",
        label="Warped-Noise Runtime",
        ok=True,
        message="Netflix VOID warped-noise runtime dependencies are importable (torch + torchvision).",
    )


def _fooocus_inpaint_assets_check() -> DependencyCheckRow:
    try:
        assets = resolve_fooocus_inpaint_assets()
    except Exception as exc:
        return DependencyCheckRow(
            id="fooocus_inpaint_assets",
            label="Fooocus Inpaint Assets",
            ok=False,
            message=str(exc),
            inpaint_modes=("fooocus_inpaint",),
        )

    return DependencyCheckRow(
        id="fooocus_inpaint_assets",
        label="Fooocus Inpaint Assets",
        ok=True,
        message=(
            "Fooocus Inpaint assets resolved successfully: "
            f"head={assets.head_path}; patch={assets.patch_path}."
        ),
        inpaint_modes=("fooocus_inpaint",),
    )


def _brushnet_assets_check() -> DependencyCheckRow:
    try:
        assets = resolve_brushnet_assets()
    except Exception as exc:
        return DependencyCheckRow(
            id="brushnet_assets",
            label="BrushNet Assets",
            ok=False,
            message=str(exc),
            inpaint_modes=("brushnet",),
        )

    return DependencyCheckRow(
        id="brushnet_assets",
        label="BrushNet Assets",
        ok=True,
        message=(
            "BrushNet assets resolved successfully: "
            f"variant={assets.variant}; config={assets.config_path}; weights={assets.weights_path}."
        ),
        inpaint_modes=("brushnet",),
    )


def _normalized_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if len(text) <= 1:
        return text
    return text.rstrip("/")


def _path_in_roots(path: str, roots: list[str]) -> bool:
    norm_path = _normalized_path(path)
    if not norm_path:
        return False

    for raw_root in roots:
        root = _normalized_path(raw_root)
        if not root:
            continue
        if norm_path == root or norm_path.startswith(root + "/"):
            return True
        if root.startswith("/"):
            rel = root.lstrip("/")
            if norm_path.endswith("/" + rel) or ("/" + rel + "/") in norm_path:
                return True
    return False


def _count_assets_in_roots(value: object, roots: list[str]) -> int:
    if not isinstance(value, list):
        return 0
    count = 0
    for item in value:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if _path_in_roots(path, roots):
            count += 1
    return count


def _normalized_root_path(raw: object) -> str:
    text = os.path.expanduser(str(raw or "").strip())
    if not text:
        return ""
    try:
        return str(Path(text).resolve(strict=False))
    except Exception:
        return text


def _roots_for_keys(keys: tuple[str, ...]) -> list[str]:
    roots: list[str] = []
    for key in keys:
        for raw in get_paths_for(key):
            resolved = _normalized_root_path(raw)
            if resolved and resolved not in roots:
                roots.append(resolved)
    return roots


def _count_checkpoints_for_engine(model_api: Any, semantic_engine: str) -> int:
    records = model_api.list_checkpoints(refresh=False)
    if not isinstance(records, list):
        return 0

    family_hints = tuple(str(value).strip().lower() for value in _CHECKPOINT_FAMILY_HINTS_BY_ENGINE.get(semantic_engine, ()))
    if family_hints:
        scoped = 0
        for record in records:
            hint = str(getattr(record, "family_hint", "") or "").strip().lower()
            if hint in family_hints:
                scoped += 1
        if scoped > 0:
            return scoped

    root_keys = _CHECKPOINT_ROOT_KEYS_BY_ENGINE.get(semantic_engine, ())
    roots = _roots_for_keys(root_keys)
    if root_keys and not roots:
        return 0
    if not roots:
        return len(records)

    scoped = 0
    for record in records:
        path = str(getattr(record, "filename", "") or "").strip()
        if _path_in_roots(path, roots):
            scoped += 1
    return scoped


def _text_encoder_slots(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    slots: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        slot = str(item.get("slot") or "").strip()
        if slot:
            slots.add(slot)
    return slots


def _checkpoint_count(model_api: Any) -> int:
    records = model_api.list_checkpoints(refresh=False)
    if isinstance(records, list):
        return len(records)
    return 0


def _records_for_semantic_engine(model_api: Any, semantic_engine: str) -> list[Any]:
    records = model_api.list_checkpoints(refresh=False)
    if not isinstance(records, list):
        return []

    family_hints = tuple(str(value).strip().lower() for value in _CHECKPOINT_FAMILY_HINTS_BY_ENGINE.get(semantic_engine, ()))
    if family_hints:
        scoped_by_family = []
        for record in records:
            hint = str(getattr(record, "family_hint", "") or "").strip().lower()
            if hint in family_hints:
                scoped_by_family.append(record)
        if scoped_by_family:
            return scoped_by_family

    root_keys = _CHECKPOINT_ROOT_KEYS_BY_ENGINE.get(semantic_engine, ())
    roots = _roots_for_keys(root_keys)
    if root_keys and not roots:
        return []
    if not roots:
        return list(records)

    scoped: list[Any] = []
    for record in records:
        path = str(getattr(record, "filename", "") or "").strip()
        if _path_in_roots(path, roots):
            scoped.append(record)
    return scoped


def _count_named_files_in_roots(
    *,
    roots: list[str],
    name_fragments: tuple[str, ...],
) -> int:
    lowered_fragments = tuple(fragment.strip().lower() for fragment in name_fragments if fragment.strip())
    if not lowered_fragments:
        return 0
    count = 0
    seen: set[str] = set()
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        candidates: list[Path] = []
        if root.is_file():
            candidates.append(root)
        elif root.is_dir():
            try:
                candidates.extend(sorted(root.rglob("*"), key=lambda item: str(item).lower()))
            except Exception:
                continue
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            lowered_name = candidate.name.lower()
            if any(fragment in lowered_name for fragment in lowered_fragments):
                count += 1
    return count


_SIDE_ASSET_SUFFIXES: tuple[str, ...] = (".safetensor", ".safetensors", ".pt", ".bin")


def _resolve_named_side_assets_in_roots(
    *,
    roots: list[str],
    required_name_fragment: str,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    fragment = required_name_fragment.strip().lower()
    if not fragment:
        return candidates

    for raw_root in roots:
        root = Path(raw_root)
        if root.is_file():
            lower_name = root.name.lower()
            if root.suffix.lower() in _SIDE_ASSET_SUFFIXES and fragment in lower_name:
                resolved = str(root.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)
            continue
        if not root.is_dir():
            continue
        try:
            files = sorted(root.rglob("*"), key=lambda item: str(item).lower())
        except Exception:
            continue
        for path in files:
            if not path.is_file():
                continue
            if path.suffix.lower() not in _SIDE_ASSET_SUFFIXES:
                continue
            if fragment not in path.name.lower():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    return candidates


def build_engine_dependency_checks(
    *,
    engine_capabilities: Mapping[str, Mapping[str, object]],
    model_api: Any,
) -> dict[str, dict[str, object]]:
    """Build backend-owned dependency checks for semantic engines.

    Args:
        engine_capabilities: Capability surfaces keyed by semantic engine.
        model_api: Runtime models API facade (must expose `list_checkpoints(refresh=...)`).

    Returns:
        Dict keyed by semantic engine where each value has:
        - `ready`: bool
        - `checks`: list of dependency rows (`id`, `label`, `ok`, `message`)
    """

    inventory = inventory_cache.get()
    vae_count = _count_list_entries(inventory.get("vaes"))
    text_encoder_count = _count_list_entries(inventory.get("text_encoders"))
    text_encoder_slots = _text_encoder_slots(inventory.get("text_encoders"))
    wan_model_count = _count_list_entries(inventory.get("wan22"))
    wan_metadata_count = _count_wan_metadata_repos(inventory.get("metadata"))
    wan_tenc_roots = _roots_for_keys(("wan22_tenc",))
    wan_vae_roots = _roots_for_keys(("wan22_vae",))
    wan_text_encoder_count = _count_assets_in_roots(inventory.get("text_encoders"), wan_tenc_roots)
    wan_vae_count = _count_assets_in_roots(inventory.get("vaes"), wan_vae_roots)

    result: dict[str, dict[str, object]] = {}
    for semantic_engine in sorted(engine_capabilities.keys()):
        checks: list[DependencyCheckRow] = []

        checks.append(
            DependencyCheckRow(
                id="capability_surface",
                label="Capability Surface",
                ok=True,
                message=f"Backend capability surface '{semantic_engine}' loaded.",
            )
        )

        scoped_records = _records_for_semantic_engine(model_api, semantic_engine)
        checkpoint_count = len(scoped_records)
        has_core_only_checkpoints = any(bool(getattr(record, "core_only", False)) for record in scoped_records)
        has_monolithic_checkpoints = any(not bool(getattr(record, "core_only", False)) for record in scoped_records)

        if semantic_engine in _CHECKPOINT_REQUIRED_ENGINES:
            has_checkpoint = checkpoint_count > 0
            checks.append(
                DependencyCheckRow(
                    id="checkpoint_inventory",
                    label="Model Checkpoints",
                    ok=has_checkpoint,
                    message=(
                        f"{checkpoint_count} checkpoint(s) discovered by backend registry."
                        if has_checkpoint
                        else (
                            "No checkpoints discovered by backend registry. "
                            "Add at least one checkpoint and refresh model inventory."
                        )
                    ),
                )
            )

        if semantic_engine == "netflix_void":
            base_dirs = list(resolve_netflix_void_base_dirs())
            checks.append(
                DependencyCheckRow(
                    id="base_bundle",
                    label="Base Bundle",
                    ok=len(base_dirs) == 1,
                    message=(
                        f"Exactly one Netflix VOID base bundle discovered: {base_dirs[0]}."
                        if len(base_dirs) == 1
                        else (
                            "No Netflix VOID base bundle discovered. Configure `netflix_void_base` with one directory containing "
                            "model_index.json + scheduler/text_encoder/tokenizer/transformer/vae."
                            if len(base_dirs) == 0
                            else (
                                "Multiple Netflix VOID base bundles discovered. Keep exactly one valid directory under "
                                f"`netflix_void_base`: {base_dirs!r}"
                            )
                        )
                    ),
                )
            )

            selectable_records = [record for record in scoped_records if netflix_void_record_is_publicly_selectable(record)]
            runnable_records = [record for record in scoped_records if netflix_void_record_is_publicly_runnable(record)]
            unknown_records = [
                str(getattr(record, "filename", "") or "").strip()
                for record in scoped_records
                if netflix_void_checkpoint_kind(record) == NETFLIX_VOID_KIND_UNKNOWN
            ]
            pass2_only_records = [
                str(getattr(record, "filename", "") or "").strip()
                for record in scoped_records
                if netflix_void_checkpoint_kind(record) == NETFLIX_VOID_KIND_PASS2
            ]
            overlay_pair_ok = (
                len(selectable_records) == 1
                and len(runnable_records) == 1
                and len(unknown_records) == 0
            )
            if overlay_pair_ok:
                overlay_message = (
                    "Netflix VOID literal overlay pair ready: public selector "
                    f"{str(getattr(selectable_records[0], 'filename', '') or '')!r} + sibling `void_pass2.safetensors`."
                )
            elif unknown_records:
                overlay_message = (
                    "Netflix VOID checkpoint inventory contains unsupported or ambiguously named files under the scoped roots: "
                    f"{unknown_records!r}. Keep only literal `void_pass1.safetensors` / `void_pass2.safetensors` pair members."
                )
            elif len(selectable_records) == 0 and pass2_only_records:
                overlay_message = (
                    "Netflix VOID Pass 2 overlay(s) were discovered without a public Pass 1 selector. Add literal sibling "
                    f"`void_pass1.safetensors` next to the Pass 2 file(s): {pass2_only_records!r}"
                )
            elif len(selectable_records) == 0:
                overlay_message = (
                    "No public Netflix VOID Pass 1 overlay discovered. Add literal `void_pass1.safetensors` plus sibling "
                    "`void_pass2.safetensors` under `netflix_void_ckpt`."
                )
            elif len(selectable_records) > 1:
                overlay_message = (
                    "Multiple public Netflix VOID Pass 1 overlays discovered. Tranche A supports exactly one literal sibling pair: "
                    f"{[str(getattr(record, 'filename', '') or '') for record in selectable_records]!r}"
                )
            else:
                overlay_message = (
                    "Netflix VOID Pass 1 overlay is missing literal sibling `void_pass2.safetensors` in the same directory."
                )
            checks.append(
                DependencyCheckRow(
                    id="overlay_pair",
                    label="Overlay Pair",
                    ok=overlay_pair_ok,
                    message=overlay_message,
                )
            )
            checks.append(_netflix_void_warped_noise_runtime_check())

        contract_engine = contract_owner_for_semantic_engine(semantic_engine)
        contract = contract_for_engine(contract_engine)
        if semantic_engine == "qwen_image":
            checks.append(_qwen_image_vendored_metadata_check())
        if semantic_engine == "ltx2":
            checks.append(_ltx2_vendored_metadata_check())
            if has_core_only_checkpoints and not has_monolithic_checkpoints:
                contract = contract_for_core_only(contract_engine)
            if checkpoint_count > 0:
                if has_core_only_checkpoints and has_monolithic_checkpoints:
                    checks.append(
                        DependencyCheckRow(
                            id="checkpoint_mix",
                            label="Checkpoint Mix",
                            ok=True,
                            message=(
                                "Mixed LTX2 inventory discovered: every LTX2 checkpoint still requires exactly 1 external "
                                "Gemma3-12B text encoder via sha selection, while core-only GGUF checkpoints additionally require "
                                "an external video VAE. Embeddings connectors and the combined audio bundle resolve internally from "
                                "configured LTX2 roots."
                            ),
                        )
                    )
                elif has_core_only_checkpoints:
                    checks.append(
                        DependencyCheckRow(
                            id="checkpoint_mix",
                            label="Checkpoint Mix",
                            ok=True,
                            message=(
                                "Only core-only LTX2 GGUF checkpoints are currently discovered. External video VAE and exactly 1 "
                                "external Gemma3-12B text encoder are required for generation; embeddings connectors and the "
                                "combined audio bundle resolve internally from configured LTX2 roots."
                            ),
                        )
                    )
                else:
                    checks.append(
                        DependencyCheckRow(
                            id="checkpoint_mix",
                            label="Checkpoint Mix",
                            ok=True,
                            message=(
                                "Only non-core-only LTX2 checkpoints are currently discovered. They keep the transformer, merged "
                                "connectors surface, and video/audio decoders inside the checkpoint, and still require exactly 1 "
                                "external Gemma3-12B text encoder via sha selection."
                            ),
                        )
                    )
        scoped_vae_root_keys = _VAE_ROOT_KEYS_BY_CONTRACT_OWNER.get(contract_engine, ())
        scoped_tenc_root_keys = _TEXT_ENCODER_ROOT_KEYS_BY_CONTRACT_OWNER.get(contract_engine, ())
        scoped_vae_roots = _roots_for_keys(scoped_vae_root_keys)
        scoped_tenc_roots = _roots_for_keys(scoped_tenc_root_keys)
        scoped_vae_count = (
            _count_assets_in_roots(inventory.get("vaes"), scoped_vae_roots)
            if scoped_vae_root_keys
            else vae_count
        )
        scoped_text_encoder_count = (
            _count_assets_in_roots(inventory.get("text_encoders"), scoped_tenc_roots)
            if scoped_tenc_root_keys
            else text_encoder_count
        )
        scoped_text_encoder_slots = (
            {
                str(item.get("slot") or "").strip()
                for item in inventory.get("text_encoders", [])
                if isinstance(item, dict)
                and _path_in_roots(str(item.get("path") or "").strip(), scoped_tenc_roots)
                and str(item.get("slot") or "").strip()
            }
            if scoped_tenc_root_keys
            else text_encoder_slots
        )
        if contract.requires_vae:
            has_vae = scoped_vae_count > 0
            checks.append(
                DependencyCheckRow(
                    id="vae_inventory",
                    label="VAE Inventory",
                    ok=has_vae,
                    message=(
                        f"{scoped_vae_count} VAE file(s) discovered."
                        if has_vae
                        else "No VAE files discovered. Configure VAE roots and refresh inventory."
                    ),
                )
            )
        if contract.tenc_count > 0:
            required = int(contract.tenc_count)
            has_tenc = scoped_text_encoder_count >= required
            checks.append(
                DependencyCheckRow(
                    id="text_encoder_inventory",
                    label="Text Encoder Inventory",
                    ok=has_tenc,
                    message=(
                        f"{scoped_text_encoder_count} text encoder file(s) discovered (requires >= {required})."
                        if has_tenc
                        else (
                            f"Only {scoped_text_encoder_count} text encoder file(s) discovered "
                            f"(requires >= {required}). Configure text-encoder roots and refresh inventory."
                        )
                    ),
                )
            )
            required_slots = tuple(str(slot) for slot in contract.tenc_slots)
            if required_slots:
                missing_slots = [slot for slot in required_slots if slot not in scoped_text_encoder_slots]
                has_required_slots = len(missing_slots) == 0
                checks.append(
                    DependencyCheckRow(
                        id="text_encoder_slots",
                        label="Text Encoder Slots",
                        ok=has_required_slots,
                        message=(
                            f"Required slots available: {', '.join(required_slots)}."
                            if has_required_slots
                            else (
                                "Missing required text encoder slot(s): "
                                f"{', '.join(missing_slots)}."
                            )
                        ),
                    )
                )

        if semantic_engine == "ltx2" and has_core_only_checkpoints:
            connectors_roots = _roots_for_keys(("ltx2_connectors",))
            connectors_matches = _resolve_named_side_assets_in_roots(
                roots=connectors_roots,
                required_name_fragment="embeddings_connectors",
            )
            connectors_count = len(connectors_matches)
            has_connectors = connectors_count == 1
            checks.append(
                DependencyCheckRow(
                    id="connectors_inventory",
                    label="Embeddings Connectors",
                    ok=has_connectors,
                    message=(
                        f"Exactly one LTX embeddings connectors file discovered: {connectors_matches[0]}."
                        if connectors_count == 1
                        else (
                            "No LTX embeddings connectors discovered. Configure `ltx2_connectors` roots and refresh inventory."
                            if connectors_count == 0
                            else (
                                "Multiple LTX embeddings connectors files discovered. Keep exactly one file containing "
                                f"'embeddings_connectors' under `ltx2_connectors`: {connectors_matches!r}"
                            )
                        )
                    ),
                )
            )

            audio_bundle_matches = _resolve_named_side_assets_in_roots(
                roots=scoped_vae_roots,
                required_name_fragment="audio_vae",
            )
            audio_bundle_count = len(audio_bundle_matches)
            has_audio_bundle = audio_bundle_count == 1
            checks.append(
                DependencyCheckRow(
                    id="audio_bundle_inventory",
                    label="Audio Bundle",
                    ok=has_audio_bundle,
                    message=(
                        f"Exactly one LTX audio bundle discovered: {audio_bundle_matches[0]}."
                        if audio_bundle_count == 1
                        else (
                            "No LTX audio bundle discovered. Configure `ltx2_vae` roots and refresh inventory."
                            if audio_bundle_count == 0
                            else (
                                "Multiple LTX audio bundle files discovered. Keep exactly one file containing "
                                f"'audio_vae' under `ltx2_vae`: {audio_bundle_matches!r}"
                            )
                        )
                    ),
                )
            )

        if semantic_engine == "sdxl":
            checks.append(_fooocus_inpaint_assets_check())
            checks.append(_brushnet_assets_check())

        if semantic_engine == "wan22":
            has_wan_models = wan_model_count > 0
            checks.append(
                DependencyCheckRow(
                    id="wan_models_inventory",
                    label="WAN Models",
                    ok=has_wan_models,
                    message=(
                        f"{wan_model_count} WAN GGUF model(s) discovered."
                        if has_wan_models
                        else "No WAN GGUF models discovered. Configure WAN roots and refresh inventory."
                    ),
                )
            )

            has_wan_text_encoder = wan_text_encoder_count > 0
            checks.append(
                DependencyCheckRow(
                    id="wan_text_encoder_inventory",
                    label="WAN Text Encoder",
                    ok=has_wan_text_encoder,
                    message=(
                        f"{wan_text_encoder_count} WAN text encoder file(s) discovered."
                        if has_wan_text_encoder
                        else (
                            "No text encoders discovered under WAN roots. "
                            "Configure `wan22_tenc` roots and refresh inventory."
                        )
                    ),
                )
            )

            has_wan_vae = wan_vae_count > 0
            checks.append(
                DependencyCheckRow(
                    id="wan_vae_inventory",
                    label="WAN VAE",
                    ok=has_wan_vae,
                    message=(
                        f"{wan_vae_count} WAN VAE file(s) discovered."
                        if has_wan_vae
                        else (
                            "No VAE files discovered under WAN roots. "
                            "Configure `wan22_vae` roots and refresh inventory."
                        )
                    ),
                )
            )

            has_wan_metadata = wan_metadata_count > 0
            checks.append(
                DependencyCheckRow(
                    id="wan_metadata_inventory",
                    label="WAN Metadata",
                    ok=has_wan_metadata,
                    message=(
                        f"{wan_metadata_count} WAN metadata repo(s) discovered under apps/backend/huggingface."
                        if has_wan_metadata
                        else (
                            "No WAN metadata repository discovered under apps/backend/huggingface. "
                            "Vendor a Wan2.2 metadata repo (e.g. Wan-AI/Wan2.2-I2V-A14B-Diffusers)."
                        )
                    ),
                )
            )

        status = EngineDependencyStatus.from_checks(checks)
        result[semantic_engine] = status.as_dict()

    return result


__all__ = [
    "DependencyCheckRow",
    "EngineDependencyStatus",
    "build_engine_dependency_checks",
]

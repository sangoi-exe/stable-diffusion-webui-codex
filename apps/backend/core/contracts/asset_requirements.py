"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical per-engine asset requirements (VAE usage, external VAE requirements, and text encoders) for generation requests.
Centralizes runtime VAE usage separately from external asset requirements so UI ↔ API ↔ loader stay in sync without duplicated `engine_id in (...)` logic.
    Includes sha-selected external-asset engines (e.g., FLUX.2 Klein, Qwen Image, LTX2 GGUF core-only, Z-Image, Z-Image L2P, and Anima) where VAE/text-encoder
    weights must be provided explicitly.
WAN22 engine variants (`wan22_5b`, `wan22_14b`, `wan22_14b_animate`) are modeled as explicit engine contracts with strict owner mapping, and
Netflix VOID uses an explicit base-bundle-owned contract (`netflix_void_base` + `netflix_void_ckpt`) with no external VAE/text-encoder slots.
Capability-only exact ids are intentionally absent from asset-contract owner maps so dependency lookup fails loud for non-executable ids.

Symbols (top-level; keep in sync; no ghosts):
- `TextEncoderKind` (enum): UI-friendly label for the expected text encoder selection kind.
- `EngineAssetContract` (dataclass): Runtime VAE usage and required external asset contract for a specific engine request context.
- `contract_owner_for_engine` (function): Resolve canonical asset-contract owner engine id for an API/runtime engine id.
- `contract_owner_for_semantic_engine` (function): Resolve canonical asset-contract owner engine id for a semantic engine surface.
- `contract_for_engine` (function): Base contract for an engine when the selected checkpoint is not core-only.
- `contract_for_core_only` (function): Contract for an engine when the selected checkpoint is core-only.
- `contract_for_request` (function): Resolve the effective contract for an engine request (e.g. core-only checkpoints).
- `format_text_encoder_kind_label` (function): Human label used in error messages and UI copy.
- `known_engine_ids` (function): Returns the set of engine ids covered by this contract module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from apps.backend.infra.config.env_flags import env_flag


class TextEncoderKind(str, Enum):
    NONE = "none"
    CLIP = "clip"
    SDXL = "sdxl"
    CLIP_T5 = "clip+t5"
    T5 = "t5"
    QWEN = "qwen"
    GEMMA = "gemma"
    SD3 = "sd3"


def format_text_encoder_kind_label(kind: TextEncoderKind) -> str:
    if kind is TextEncoderKind.NONE:
        return "None"
    if kind is TextEncoderKind.CLIP:
        return "CLIP"
    if kind is TextEncoderKind.SDXL:
        return "SDXL (CLIP-L + CLIP-G)"
    if kind is TextEncoderKind.CLIP_T5:
        return "CLIP + T5"
    if kind is TextEncoderKind.T5:
        return "T5"
    if kind is TextEncoderKind.QWEN:
        return "Qwen"
    if kind is TextEncoderKind.GEMMA:
        return "Gemma"
    if kind is TextEncoderKind.SD3:
        return "SD3 (CLIP-L + CLIP-G + T5)"
    return str(kind.value)


@dataclass(frozen=True, slots=True)
class EngineAssetContract:
    """Asset requirements for an engine request context."""

    requires_vae: bool
    uses_vae: bool
    tenc_slots: tuple[str, ...]
    tenc_kind: TextEncoderKind
    sha_only: bool
    tenc_slot_labels: tuple[str, ...] | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.requires_vae and not self.uses_vae:
            raise ValueError("requires_vae implies uses_vae")

        slots = tuple(str(s).strip() for s in (self.tenc_slots or ()))
        if any(not s for s in slots):
            raise ValueError("tenc_slots must not contain empty values")
        if len(set(slots)) != len(slots):
            raise ValueError("tenc_slots must not contain duplicates")
        object.__setattr__(self, "tenc_slots", slots)

        labels = self.tenc_slot_labels
        if labels is not None:
            labels = tuple(str(s).strip() for s in labels)
            if any(not s for s in labels):
                raise ValueError("tenc_slot_labels must not contain empty values")
            if len(labels) != len(slots):
                raise ValueError("tenc_slot_labels must match tenc_slots length")
            object.__setattr__(self, "tenc_slot_labels", labels)

        if self.tenc_count == 0 and self.tenc_kind is not TextEncoderKind.NONE:
            raise ValueError("tenc_kind must be NONE when tenc_slots is empty")
        if self.tenc_count > 0 and self.tenc_kind is TextEncoderKind.NONE:
            raise ValueError("tenc_kind must not be NONE when tenc_slots is non-empty")

    @property
    def requires_text_encoders(self) -> bool:
        return self.tenc_count > 0

    @property
    def tenc_count(self) -> int:
        return len(self.tenc_slots)

    def as_dict(self) -> dict[str, object]:
        return {
            "requires_vae": bool(self.requires_vae),
            "uses_vae": bool(self.uses_vae),
            "tenc_count": int(self.tenc_count),
            "tenc_slots": list(self.tenc_slots),
            "tenc_slot_labels": list(self.tenc_slot_labels or []),
            "tenc_kind": str(self.tenc_kind.value),
            "tenc_kind_label": format_text_encoder_kind_label(self.tenc_kind),
            "sha_only": bool(self.sha_only),
            "notes": str(self.notes or ""),
        }


_BASE_CONTRACTS: dict[str, EngineAssetContract] = {
    # Diffusion checkpoints embed VAE/text encoders; external assets are optional overrides.
    "sd15": EngineAssetContract(
        requires_vae=False,
        uses_vae=True,
        tenc_slots=(),
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Monolithic checkpoint; external VAE/text encoders are optional overrides.",
    ),
    "sd20": EngineAssetContract(
        requires_vae=False,
        uses_vae=True,
        tenc_slots=(),
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Monolithic checkpoint; external VAE/text encoders are optional overrides.",
    ),
    "sdxl": EngineAssetContract(
        requires_vae=False,
        uses_vae=True,
        tenc_slots=(),
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Monolithic checkpoint; external VAE/text encoders are optional overrides.",
    ),
    "sdxl_refiner": EngineAssetContract(
        requires_vae=False,
        uses_vae=True,
        tenc_slots=(),
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Monolithic checkpoint; external VAE/text encoders are optional overrides.",
    ),
    "sd35": EngineAssetContract(
        requires_vae=False,
        uses_vae=True,
        tenc_slots=(),
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Diffusers-style checkpoint; external VAE/text encoders are optional overrides.",
    ),
    # External-assets-first families.
    "flux1": EngineAssetContract(
        requires_vae=True,
        uses_vae=True,
        tenc_slots=("clip_l", "t5xxl"),
        tenc_slot_labels=("CLIP-L", "T5-XXL"),
        tenc_kind=TextEncoderKind.CLIP_T5,
        sha_only=True,
        notes="External-assets-first: requires VAE + 2 text encoders (CLIP + T5) via sha selection.",
    ),
    "flux1_kontext": EngineAssetContract(
        requires_vae=True,
        uses_vae=True,
        tenc_slots=("clip_l", "t5xxl"),
        tenc_slot_labels=("CLIP-L", "T5-XXL"),
        tenc_kind=TextEncoderKind.CLIP_T5,
        sha_only=True,
        notes="External-assets-first: requires VAE + 2 text encoders (CLIP + T5) via sha selection.",
    ),
    "flux2": EngineAssetContract(
        requires_vae=True,
        uses_vae=True,
        tenc_slots=("qwen3_4b",),
        tenc_slot_labels=("Qwen3-4B",),
        tenc_kind=TextEncoderKind.QWEN,
        sha_only=True,
        notes="External-assets-first: requires FLUX.2 VAE + 1 Qwen3-4B text encoder via sha selection.",
    ),
    "qwen_image": EngineAssetContract(
        requires_vae=True,
        uses_vae=True,
        tenc_slots=("qwen2_5_vl_7b",),
        tenc_slot_labels=("Qwen2.5-VL-7B",),
        tenc_kind=TextEncoderKind.QWEN,
        sha_only=True,
        notes="External-assets-first: requires Qwen Image VAE + 1 Qwen2.5-VL-7B multimodal text encoder via sha selection.",
    ),
    "zimage": EngineAssetContract(
        requires_vae=True,
        uses_vae=True,
        tenc_slots=("qwen3_4b",),
        tenc_slot_labels=("Qwen3-4B",),
        tenc_kind=TextEncoderKind.QWEN,
        sha_only=True,
        notes="External-assets-first: requires Flow16 VAE + 1 Qwen text encoder via sha selection.",
    ),
    "zimage_l2p": EngineAssetContract(
        requires_vae=False,
        uses_vae=False,
        tenc_slots=("qwen3_4b",),
        tenc_slot_labels=("Qwen3-4B",),
        tenc_kind=TextEncoderKind.QWEN,
        sha_only=True,
        notes="External-assets-first: pixel-space L2P core requires exactly 1 Qwen3-4B text encoder and no VAE.",
    ),
    "anima": EngineAssetContract(
        requires_vae=True,
        uses_vae=True,
        tenc_slots=("qwen3_06b",),
        tenc_slot_labels=("Qwen3-0.6B",),
        tenc_kind=TextEncoderKind.QWEN,
        sha_only=True,
        notes="External-assets-first: requires WanVAE-style VAE (3D conv; `qwen_image_vae.safetensors`) + 1 Qwen3-0.6B text encoder via sha selection.",
    ),
    "wan22_5b": EngineAssetContract(
        requires_vae=True,
        uses_vae=True,
        tenc_slots=("t5xxl",),
        tenc_slot_labels=("T5-XXL",),
        tenc_kind=TextEncoderKind.T5,
        sha_only=True,
        notes="External-assets-first: requires WAN VAE + 1 T5 text encoder via sha selection.",
    ),
    "wan22_14b": EngineAssetContract(
        requires_vae=True,
        uses_vae=True,
        tenc_slots=("t5xxl",),
        tenc_slot_labels=("T5-XXL",),
        tenc_kind=TextEncoderKind.T5,
        sha_only=True,
        notes="External-assets-first: requires WAN VAE + 1 T5 text encoder via sha selection.",
    ),
    "wan22_14b_animate": EngineAssetContract(
        requires_vae=True,
        uses_vae=True,
        tenc_slots=("t5xxl",),
        tenc_slot_labels=("T5-XXL",),
        tenc_kind=TextEncoderKind.T5,
        sha_only=True,
        notes="External-assets-first: requires WAN VAE + 1 T5 text encoder via sha selection.",
    ),
    "ltx2": EngineAssetContract(
        requires_vae=False,
        uses_vae=True,
        tenc_slots=("gemma3_12b",),
        tenc_slot_labels=("Gemma3-12B",),
        tenc_kind=TextEncoderKind.GEMMA,
        sha_only=True,
        notes=(
            "Non-core-only LTX2 checkpoint path keeps transformer / merged connector surface / video-audio decoders inside the "
            "checkpoint and still requires exactly 1 external Gemma3-12B text encoder via sha selection. The current distilled GGUF "
            "pack uses the core-only contract instead."
        ),
    ),
    "netflix_void": EngineAssetContract(
        requires_vae=False,
        uses_vae=True,
        tenc_slots=(),
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes=(
            "Base-bundle-owned video inpainting family: tokenizer/text encoder/transformer/vae/scheduler live under "
            "`netflix_void_base`, while Pass 1/Pass 2 overlays live under `netflix_void_ckpt`. External VAE/text encoders "
            "are not part of this contract."
        ),
    ),
    "svd": EngineAssetContract(
        requires_vae=False,
        uses_vae=True,
        tenc_slots=(),
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Monolithic video checkpoint; external VAE/text encoders are optional overrides.",
    ),
    "hunyuan_video": EngineAssetContract(
        requires_vae=False,
        uses_vae=True,
        tenc_slots=(),
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Monolithic video checkpoint; external VAE/text encoders are optional overrides.",
    ),
    # Chroma safetensors are treated as monolithic; GGUF selections remain core-only.
    "flux1_chroma": EngineAssetContract(
        requires_vae=False,
        uses_vae=True,
        tenc_slots=(),
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Chroma safetensors are treated as monolithic; external assets are optional overrides.",
    ),
}

_CONTRACT_OWNER_BY_ENGINE_ID: dict[str, str] = {
    "sd15": "sd15",
    "sd20": "sd20",
    "sdxl": "sdxl",
    "sdxl_refiner": "sdxl_refiner",
    "sd35": "sd35",
    "flux1": "flux1",
    "flux1_kontext": "flux1_kontext",
    "flux2": "flux2",
    "qwen_image": "qwen_image",
    "flux1_chroma": "flux1_chroma",
    "zimage": "zimage",
    "zimage_l2p": "zimage_l2p",
    "anima": "anima",
    "wan22_5b": "wan22_5b",
    "wan22_14b": "wan22_14b",
    "wan22_14b_animate": "wan22_14b_animate",
    "ltx2": "ltx2",
    "netflix_void": "netflix_void",
    "svd": "svd",
    "hunyuan_video": "hunyuan_video",
}

_CONTRACT_OWNER_BY_SEMANTIC_ENGINE: dict[str, str] = {
    "sd15": "sd15",
    "sdxl": "sdxl",
    "flux1": "flux1",
    "flux2": "flux2",
    "qwen_image": "qwen_image",
    "chroma": "flux1_chroma",
    "zimage": "zimage",
    "zimage_l2p": "zimage_l2p",
    "anima": "anima",
    "wan22": "wan22_14b",
    "ltx2": "ltx2",
    "netflix_void": "netflix_void",
    "svd": "svd",
    "hunyuan_video": "hunyuan_video",
}


def contract_owner_for_engine(engine_id: str) -> str:
    key = str(engine_id or "").strip().lower()
    if not key:
        raise ValueError("engine_id required")
    owner = _CONTRACT_OWNER_BY_ENGINE_ID.get(key)
    if owner is None:
        raise KeyError(f"Engine asset contract owner missing for engine_id={key!r}")
    return owner


def contract_owner_for_semantic_engine(semantic_engine: str) -> str:
    key = str(semantic_engine or "").strip().lower()
    if not key:
        raise ValueError("semantic_engine required")
    owner = _CONTRACT_OWNER_BY_SEMANTIC_ENGINE.get(key)
    if owner is None:
        raise KeyError(f"Engine asset contract owner missing for semantic_engine={key!r}")
    return owner


def contract_for_engine(engine_id: str) -> EngineAssetContract:
    """Return the base contract for an engine.

    This is the contract for non-core-only checkpoint selections.
    """

    owner = contract_owner_for_engine(engine_id)
    contract = _BASE_CONTRACTS.get(owner)
    if contract is None:
        raise KeyError(f"Engine asset contract missing for owner_engine_id={owner!r}")
    return contract


def contract_for_core_only(engine_id: str) -> EngineAssetContract:
    """Return the contract when the selected checkpoint is core-only."""

    owner = contract_owner_for_engine(engine_id)

    if owner in (
        "flux1",
        "flux1_kontext",
        "flux2",
        "qwen_image",
        "zimage",
        "zimage_l2p",
        "anima",
        "wan22_5b",
        "wan22_14b",
        "wan22_14b_animate",
        "svd",
        "hunyuan_video",
    ):
        return contract_for_engine(owner)

    if owner == "ltx2":
        return EngineAssetContract(
            requires_vae=True,
            uses_vae=True,
            tenc_slots=("gemma3_12b",),
            tenc_slot_labels=("Gemma3-12B",),
            tenc_kind=TextEncoderKind.GEMMA,
            sha_only=True,
            notes=(
                "Core-only LTX2 GGUF checkpoint requires an external video VAE and exactly 1 external Gemma3-12B text encoder via sha "
                "selection. Embeddings connectors and the combined audio bundle resolve internally from configured LTX2 roots; mmproj and "
                "optional upscalers are outside this base contract."
            ),
        )

    if owner == "flux1_chroma":
        return EngineAssetContract(
            requires_vae=True,
            uses_vae=True,
            tenc_slots=("t5xxl",),
            tenc_slot_labels=("T5-XXL",),
            tenc_kind=TextEncoderKind.T5,
            sha_only=True,
            notes="Core-only checkpoint: requires external VAE + 1 T5 text encoder.",
        )

    if owner in ("sd15", "sd20"):
        return EngineAssetContract(
            requires_vae=True,
            uses_vae=True,
            tenc_slots=("clip_l",),
            tenc_slot_labels=("CLIP-L",),
            tenc_kind=TextEncoderKind.CLIP,
            sha_only=True,
            notes="Core-only checkpoint: requires external VAE + 1 CLIP text encoder.",
        )

    if owner in ("sdxl", "sdxl_refiner"):
        if owner == "sdxl_refiner":
            return EngineAssetContract(
                requires_vae=True,
                uses_vae=True,
                tenc_slots=("clip_g",),
                tenc_slot_labels=("CLIP-G",),
                tenc_kind=TextEncoderKind.CLIP,
                sha_only=True,
                notes="Core-only checkpoint: requires external VAE + 1 SDXL refiner text encoder (CLIP-G).",
            )
        return EngineAssetContract(
            requires_vae=True,
            uses_vae=True,
            tenc_slots=("clip_l", "clip_g"),
            tenc_slot_labels=("CLIP-L", "CLIP-G"),
            tenc_kind=TextEncoderKind.SDXL,
            sha_only=True,
            notes="Core-only checkpoint: requires external VAE + 2 SDXL text encoders.",
        )

    if owner == "sd35":
        enable_t5 = env_flag("CODEX_SD3_ENABLE_T5", default=True)
        slots = ("clip_l", "clip_g", "t5xxl") if enable_t5 else ("clip_l", "clip_g")
        labels = ("CLIP-L", "CLIP-G", "T5-XXL") if enable_t5 else ("CLIP-L", "CLIP-G")
        return EngineAssetContract(
            requires_vae=True,
            uses_vae=True,
            tenc_slots=slots,
            tenc_slot_labels=labels,
            tenc_kind=TextEncoderKind.SD3,
            sha_only=True,
            notes=(
                "Core-only checkpoint: requires external VAE + SD3 text encoders "
                f"(tenc_count={len(slots)}; CODEX_SD3_ENABLE_T5={bool(enable_t5)})."
            ),
        )

    base = contract_for_engine(owner)
    return EngineAssetContract(
        requires_vae=True,
        uses_vae=True,
        tenc_slots=("clip_l",),
        tenc_slot_labels=("CLIP-L",),
        tenc_kind=TextEncoderKind.CLIP,
        sha_only=bool(base.sha_only),
        notes="Core-only checkpoint: requires external VAE + at least one text encoder (default contract).",
    )


def contract_for_request(*, engine_id: str, checkpoint_core_only: bool) -> EngineAssetContract:
    """Resolve the effective asset contract for an engine request."""

    if checkpoint_core_only:
        return contract_for_core_only(engine_id)
    return contract_for_engine(engine_id)


def known_engine_ids() -> tuple[str, ...]:
    """Return engine ids covered by the contract mapping."""

    return tuple(sorted(_CONTRACT_OWNER_BY_ENGINE_ID.keys()))

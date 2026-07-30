"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image Edit-2511 engine facade for the single `qwen_image` architecture family.
Validates the fixed internal `edit_2511` identity, required external Qwen Image assets, exact VAE/header contracts,
and the loader-produced core-only bundle before native img2img runtime execution.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageEngine` (class): Registered Edit-only engine facade for `qwen_image`; validates strict load contracts and delegates img2img once the native runtime is assembled.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from apps.backend.core.engine_interface import BaseInferenceEngine, EngineCapabilities, TaskType
from apps.backend.infra.config.paths import get_paths_for
from apps.backend.runtime.families.qwen_image.config import (
    QWEN_IMAGE_EDIT_VARIANT,
    QWEN_IMAGE_ENGINE_ID,
    QWEN_IMAGE_SUPPORTED_VARIANTS,
    QWEN_IMAGE_VARIANT_KEY,
    require_qwen_image_variant,
)
from apps.backend.runtime.families.qwen_image.vae import qwen_image_validate_external_vae_path
from apps.backend.runtime.logging import get_backend_logger
from apps.backend.runtime.model_registry.capabilities import ENGINE_SURFACES, SemanticEngine

if TYPE_CHECKING:
    from apps.backend.runtime.models.loader import DiffusionModelBundle


logger = get_backend_logger("backend.engines.qwen_image")


def _require_variant(options: Mapping[str, Any]) -> str:
    return require_qwen_image_variant(
        options.get(QWEN_IMAGE_VARIANT_KEY),
        context=f"Qwen Image load internal engine option {QWEN_IMAGE_VARIANT_KEY!r}",
    )


def _require_external_path(options: Mapping[str, Any], *, key: str, label: str) -> str:
    raw_path = options.get(key)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError(f"Qwen Image requires external {label} via engine option '{key}'.")
    path = Path(raw_path.strip()).expanduser()
    if not path.exists():
        raise RuntimeError(f"Qwen Image external {label} path not found: {path}")
    return str(path)


def _require_qwen_image_vae_path(options: Mapping[str, Any]) -> str:
    vae_path = _require_external_path(options, key="vae_path", label="Qwen Image VAE")
    qwen_image_vae_roots = tuple(get_paths_for("qwen_image_vae"))
    if not qwen_image_vae_roots:
        raise RuntimeError("Qwen Image engine load VAE: no qwen_image_vae roots are configured.")
    return qwen_image_validate_external_vae_path(
        vae_path,
        allowed_roots=qwen_image_vae_roots,
        context="Qwen Image engine load VAE",
    )


class QwenImageEngine(BaseInferenceEngine):
    """Qwen Image facade registered under the canonical `qwen_image` engine id."""

    engine_id = QWEN_IMAGE_ENGINE_ID

    def __init__(self) -> None:
        super().__init__()
        self._bundle: DiffusionModelBundle | None = None
        self._model_ref: str | None = None
        self._variant: str | None = None

    def capabilities(self) -> EngineCapabilities:
        surface = ENGINE_SURFACES[SemanticEngine.QWEN_IMAGE]
        tasks: list[TaskType] = []
        if surface.supports_txt2img:
            tasks.append(TaskType.TXT2IMG)
        if surface.supports_img2img:
            tasks.append(TaskType.IMG2IMG)
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=tuple(tasks),
            model_types=(QWEN_IMAGE_ENGINE_ID,),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
            extras={"variants": tuple(sorted(QWEN_IMAGE_SUPPORTED_VARIANTS))},
        )

    def load(self, model_ref: str, **options: Any) -> None:
        from apps.backend.runtime.model_registry.specs import ModelFamily
        from apps.backend.runtime.models.loader import resolve_diffusion_bundle

        if self._is_loaded:
            self.unload()

        variant = _require_variant(options)
        vae_source = options.get("vae_source")
        if vae_source != "external":
            raise RuntimeError("Qwen Image requires vae_source='external'.")
        tenc_source = options.get("tenc_source")
        if tenc_source != "external":
            raise RuntimeError("Qwen Image requires tenc_source='external'.")
        tenc_path = _require_external_path(options, key="tenc_path", label="Qwen2.5-VL-7B text encoder")
        vae_path = _require_qwen_image_vae_path(options)
        if "text_encoder_override" in options:
            raise RuntimeError("Qwen Image does not accept text_encoder_override; use tenc_path from the qwen2_5_vl_7b slot.")

        bundle = resolve_diffusion_bundle(
            model_ref,
            vae_path=vae_path,
            tenc_path=tenc_path,
            expected_family=ModelFamily.QWEN_IMAGE,
        )
        if bundle.family is not ModelFamily.QWEN_IMAGE:
            raise RuntimeError(
                "Qwen Image loader returned wrong family: "
                f"expected {ModelFamily.QWEN_IMAGE.value}, got {getattr(bundle.family, 'value', bundle.family)!r}."
            )
        bundle_variant = bundle.metadata.get(QWEN_IMAGE_VARIANT_KEY)
        if bundle_variant != variant:
            raise RuntimeError(
                "Qwen Image variant mismatch: "
                f"request variant {variant!r} cannot load metadata variant {bundle_variant!r} from {model_ref!r}."
            )

        self._bundle = bundle
        self._model_ref = str(model_ref)
        self._variant = variant
        self.mark_loaded()
        logger.info("Loaded Qwen Image Edit-2511 core bundle: model_ref=%s variant=%s", self._model_ref, variant)

    def unload(self) -> None:
        self._bundle = None
        self._model_ref = None
        self._variant = None
        self.mark_unloaded()

    def status(self) -> Mapping[str, Any]:
        data = dict(super().status())
        if self._model_ref is not None:
            data["model_ref"] = self._model_ref
        if self._variant is not None:
            data[QWEN_IMAGE_VARIANT_KEY] = self._variant
        if self._bundle is not None:
            data["bundle_source"] = self._bundle.source
        return data

    def img2img(self, request: Any, **kwargs: Any) -> Iterator[Any]:
        del request, kwargs
        self.ensure_loaded()
        if self._variant != QWEN_IMAGE_EDIT_VARIANT:
            raise RuntimeError("Qwen Image img2img edit requires loaded variant 'edit_2511'.")
        raise NotImplementedError("Qwen Image img2img edit native runtime not yet implemented")


__all__ = ["QwenImageEngine"]

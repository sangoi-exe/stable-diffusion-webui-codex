"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Qwen Image Edit-2511 engine facade for the single `qwen_image` architecture family.
Validates the fixed internal `edit_2511` identity, mandatory blocked Core streaming, required external Qwen Image
assets, and exact VAE/header contracts; assembles patcher-backed native components plus the family runtime through the
common diffusion engine and delegates img2img to the canonical use-case pipeline.

Symbols (top-level; keep in sync; no ghosts):
- `QwenImageEngine` (class): Registered Edit-only common diffusion engine with strict native component assembly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects, TextEncoderHandle
from apps.backend.engines.common.runtime_lifecycle import require_runtime
from apps.backend.infra.config.paths import get_paths_for
from apps.backend.runtime.families.qwen_image.config import (
    QWEN_IMAGE_ENGINE_ID,
    QWEN_IMAGE_SUPPORTED_VARIANTS,
    QWEN_IMAGE_VARIANT_KEY,
    require_qwen_image_variant,
)
from apps.backend.runtime.families.qwen_image.streaming import (
    require_qwen_image_streaming_activation,
)
from apps.backend.runtime.families.qwen_image.vae import qwen_image_validate_external_vae_path
from apps.backend.runtime.logging import get_backend_logger
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.model_registry.capabilities import ENGINE_SURFACES, SemanticEngine
from apps.backend.runtime.model_registry.specs import ModelFamily

if TYPE_CHECKING:
    from apps.backend.runtime.families.qwen_image.loader import QwenImageComponentAssembly
    from apps.backend.runtime.families.qwen_image.runtime import QwenImageRuntime
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


def _path_is_under(path: Path, roots: tuple[str, ...]) -> bool:
    resolved_path = path.resolve()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if root.is_file():
            if resolved_path == root:
                return True
            continue
        try:
            resolved_path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _require_qwen_image_tenc_path(options: Mapping[str, Any]) -> str:
    tenc_path = _require_external_path(options, key="tenc_path", label="Qwen2.5-VL-7B text encoder")
    roots = tuple(get_paths_for("qwen_image_tenc"))
    if not roots:
        raise RuntimeError("Qwen Image engine load text encoder: no qwen_image_tenc roots are configured.")
    path = Path(tenc_path).expanduser()
    if not _path_is_under(path, roots):
        raise RuntimeError(
            "Qwen Image engine load text encoder: path must be under qwen_image_tenc roots. "
            f"path={path} roots={roots}."
        )
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


class QwenImageEngine(CodexDiffusionEngine):
    """Qwen Image facade registered under the canonical `qwen_image` engine id."""

    engine_id = QWEN_IMAGE_ENGINE_ID
    expected_family = ModelFamily.QWEN_IMAGE

    def __init__(self) -> None:
        super().__init__()
        self._assembly: QwenImageComponentAssembly | None = None
        self._runtime: QwenImageRuntime | None = None
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
            devices=("cuda",),
            precision=("bf16",),
            extras={
                "variants": tuple(sorted(QWEN_IMAGE_SUPPORTED_VARIANTS)),
                "samplers": ("euler",),
                "schedulers": ("simple",),
            },
        )

    @property
    def required_text_encoders(self) -> tuple[str, ...]:
        return ("qwen2_5_vl_7b",)

    def load(self, model_ref: str, **options: Any) -> None:
        require_qwen_image_streaming_activation(
            options,
            swap_method=memory_management.manager.config.swap.method,
        )
        normalized_options = dict(options)
        variant = _require_variant(options)
        vae_source = options.get("vae_source")
        if vae_source != "external":
            raise RuntimeError("Qwen Image requires vae_source='external'.")
        tenc_source = options.get("tenc_source")
        if tenc_source != "external":
            raise RuntimeError("Qwen Image requires tenc_source='external'.")
        tenc_path = _require_qwen_image_tenc_path(options)
        vae_path = _require_qwen_image_vae_path(options)
        if "text_encoder_override" in options:
            raise RuntimeError("Qwen Image does not accept text_encoder_override; use tenc_path from the qwen2_5_vl_7b slot.")
        raw_model_format = options.get("model_format")
        if raw_model_format is not None and (
            not isinstance(raw_model_format, str)
            or raw_model_format.strip().lower() != "gguf"
        ):
            raise RuntimeError("Qwen Image requires model_format='gguf'.")
        raw_core_only = options.get("checkpoint_core_only")
        if raw_core_only is not None and raw_core_only is not True:
            raise RuntimeError("Qwen Image requires checkpoint_core_only=True.")

        normalized_options[QWEN_IMAGE_VARIANT_KEY] = variant
        normalized_options["tenc_path"] = tenc_path
        normalized_options["vae_path"] = vae_path
        normalized_options["model_format"] = "gguf"
        normalized_options["checkpoint_core_only"] = True
        super().load(model_ref, **normalized_options)
        logger.info("Loaded Qwen Image Edit-2511 native components: model_ref=%s variant=%s", model_ref, variant)

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, object],
    ) -> CodexObjects:
        from apps.backend.runtime.families.qwen_image.loader import load_qwen_image_components
        from apps.backend.runtime.families.qwen_image.runtime import QwenImageRuntime

        assembly = load_qwen_image_components(bundle, options=options)
        self._assembly = assembly
        self._runtime = QwenImageRuntime(assembly)
        self._variant = assembly.variant
        self.use_distilled_cfg_scale = False
        return CodexObjects(
            denoiser=assembly.denoiser,
            vae=assembly.vae,
            text_encoders={
                "qwen2_5_vl_7b": TextEncoderHandle(
                    patcher=assembly.text_encoder_patcher,
                    runtime=assembly.text_encoder,
                )
            },
        )

    def _require_runtime(self) -> QwenImageRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    @property
    def qwen_image_runtime(self) -> QwenImageRuntime:
        return self._require_runtime()

    def _on_unload(self) -> None:
        self._runtime = None
        self._assembly = None
        self._variant = None

    def status(self) -> Mapping[str, Any]:
        data = dict(super().status())
        if self._variant is not None:
            data[QWEN_IMAGE_VARIANT_KEY] = self._variant
        return data


__all__ = ["QwenImageEngine"]

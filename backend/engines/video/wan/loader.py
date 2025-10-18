from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Optional

from backend.core.exceptions import EngineLoadError


@dataclass
class WanComponents:
    # Placeholders for actual model components once integrated
    text_encoder: Any | None = None
    transformer: Any | None = None
    vae: Any | None = None
    model_dir: str | None = None
    device: str = "cpu"
    dtype: str = "fp16"


class WanLoader:
    """Lightweight loader for WAN 2.2 components.

    For MVP we only verify that a model directory exists. Real component
    construction will be added in the next iteration.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._components: Optional[WanComponents] = None

    def load(self, model_ref: str, *, device: str = "auto", dtype: str = "fp16") -> WanComponents:
        if not model_ref or not isinstance(model_ref, str):
            raise EngineLoadError("WAN loader requires a non-empty model_ref (path or repo id)")

        # Resolve local path; do not attempt remote download in MVP
        maybe_path = model_ref
        if not os.path.isabs(maybe_path):
            # Allow relative path from repo root
            maybe_path = os.path.abspath(maybe_path)

        if not os.path.isdir(maybe_path):
            # Common local layout hint
            alt = os.path.abspath(os.path.join("models", "Wan", model_ref))
            if os.path.isdir(alt):
                maybe_path = alt
            else:
                raise EngineLoadError(
                    (
                        "WAN 2.2 model path not found. Provided: %s. "
                        "Place weights under 'models/Wan/<dir>' and set Quicksettings Checkpoint to that folder, "
                        "or pass an absolute path as model_ref."
                    )
                    % model_ref
                )

        resolved_device = self._resolve_device(device)
        resolved_dtype = dtype.lower()
        if resolved_dtype not in ("fp16", "bf16", "fp32"):
            resolved_dtype = "fp16"

        self._components = WanComponents(
            text_encoder=None,
            transformer=None,
            vae=None,
            model_dir=maybe_path,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        self._logger.info(
            "WAN loader ready: dir=%s device=%s dtype=%s",
            maybe_path,
            resolved_device,
            resolved_dtype,
        )
        return self._components

    def components(self) -> WanComponents:
        if self._components is None:
            raise EngineLoadError("WAN components not loaded; call load(model_ref) first")
        return self._components

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device not in ("auto", "cpu", "cuda"):
            return "cpu"
        if device == "auto":
            try:  # local import to avoid hard dependency at import time
                import torch  # type: ignore

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return device


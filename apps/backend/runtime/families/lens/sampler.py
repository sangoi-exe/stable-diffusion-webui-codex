"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Reserved Microsoft Lens txt2img sampler owner.
Provides the named denoising seam for the parked Lens skeleton and fails loud until the real GPT-OSS/LensTransformer/VAE runtime is implemented.

Symbols (top-level; keep in sync; no ghosts):
- `sample_lens_txt2img_not_implemented` (function): Skeleton Lens sampler hook that raises the exact runtime-not-implemented error.
"""

from __future__ import annotations

from typing import Any

from .config import LENS_NOT_IMPLEMENTED_MESSAGE


def sample_lens_txt2img_not_implemented(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise NotImplementedError(LENS_NOT_IMPLEMENTED_MESSAGE)


__all__ = ["sample_lens_txt2img_not_implemented"]

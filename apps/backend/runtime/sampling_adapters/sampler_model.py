"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SamplerModel adapter for sampler-style `apply_model` callers.
Bridges `apply_model` usage to Codex diffusion models/predictors, enforcing context/y invariants and providing opt-in debug
tensor stats for Z Image and other flow runtimes. Native denoiser output is forwarded directly into predictor math without a standalone full-tensor FP32 cast.

Symbols (top-level; keep in sync; no ghosts):
- `_tensor_stats` (function): Formats quick tensor statistics for debug logging.
- `SamplerModel` (class): Adapter module exposing `apply_model`/`forward` and `memory_required` for sampler call sites.
"""

import logging
import torch
from apps.backend.runtime.logging import emit_backend_message, get_backend_logger

from apps.backend.infra.config.env_flags import env_flag, env_int
from apps.backend.runtime import attention
from .prediction import prediction_from_diffusers_scheduler


_DEBUG_LOGGER = get_backend_logger(__name__)


def _tensor_stats(label: str, tensor: torch.Tensor | None) -> str:
    if tensor is None or not torch.is_tensor(tensor):
        return f"{label}=<none>"
    with torch.no_grad():
        data = tensor.detach()
        stats = data.float()
        return (
            f"{label}:shape={tuple(data.shape)} dtype={data.dtype} dev={data.device} "
            f"min={float(stats.min().item()):.6g} max={float(stats.max().item()):.6g} "
            f"mean={float(stats.mean().item()):.6g} std={float(stats.std(unbiased=False).item()):.6g} "
            f"norm={float(stats.norm().item()):.6g}"
        )


class SamplerModel(torch.nn.Module):
    def __init__(self, model, diffusers_scheduler, predictor=None, config=None):
        super().__init__()

        self.config = config

        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        emit_backend_message(
            "SamplerModel created",
            logger=__name__,
            storage_dtype=self.storage_dtype,
            computation_dtype=self.computation_dtype,
        )

        self.diffusion_model = model

        if predictor is None:
            self.predictor = prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = predictor

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        debug_enabled = env_flag("CODEX_SAMPLER_MODEL_DEBUG") or env_flag("CODEX_SAMPLER_MODEL_DEBUG_APPLY_MODEL")
        debug_limit = env_int("CODEX_SAMPLER_MODEL_DEBUG_APPLY_MODEL_N", 3, min_value=0)
        debug_count = int(getattr(self, "_codex_apply_model_debug_count", 0) or 0)

        sigma = t
        xc = self.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.computation_dtype

        # Validate context invariants before dtype/device coercion so invalid
        # payloads fail with a clear contract error (not AttributeError).
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError(
                f"UNet context must be a 3D tensor (B,S,C); got {type(context).__name__} "
                f"shape={getattr(context, 'shape', None)}"
            )

        xc = xc.to(dtype)
        t = self.predictor.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        if debug_enabled and debug_count < debug_limit:
            try:
                sigma0 = float(sigma.detach().view(-1)[0].item()) if torch.is_tensor(sigma) else float(sigma)
            except Exception:
                sigma0 = float("nan")
            cond_flags = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
            if isinstance(cond_flags, (list, tuple)):
                cond_count = sum(1 for v in cond_flags if int(v) == 0)
                uncond_count = len(cond_flags) - cond_count
                cond_summary = f"cond={cond_count} uncond={uncond_count}"
            else:
                cond_summary = "n/a"

            extras_keys = sorted(str(k) for k in extra_conds.keys())
            emit_backend_message(
                "[sampler-model-debug] apply_model",
                logger=__name__,
                sigma=sigma0,
                prediction_type=getattr(self.predictor, "prediction_type", None),
                extras=extras_keys,
                cond_or_uncond=cond_summary,
            )
            emit_backend_message(f"[sampler-model-debug] {_tensor_stats('x', x)}", logger=__name__)
            emit_backend_message(f"[sampler-model-debug] {_tensor_stats('xc', xc)}", logger=__name__)
            emit_backend_message(f"[sampler-model-debug] {_tensor_stats('context', context)}", logger=__name__)
            if isinstance(extra_conds.get("y"), torch.Tensor):
                emit_backend_message(f"[sampler-model-debug] {_tensor_stats('y', extra_conds.get('y'))}", logger=__name__)
            if isinstance(extra_conds.get("guidance"), torch.Tensor):
                emit_backend_message(
                    f"[sampler-model-debug] {_tensor_stats('guidance', extra_conds.get('guidance'))}",
                    logger=__name__,
                )
            # transformer_options often carries sigma/cond flags; keep it compact.
            if isinstance(transformer_options, dict):
                try:
                    keys = sorted(str(k) for k in transformer_options.keys())
                    emit_backend_message("[sampler-model-debug] transformer_options keys", logger=__name__, keys=keys)
                except Exception:
                    pass
            setattr(self, "_codex_apply_model_debug_count", debug_count + 1)

        # Invariants: optional y must be consistent with diffusion model config.
        # Context rank/type was validated before coercion above.
        # Derive expected context dims from codex_config when available
        expected_ctx_dim = None
        cfg = getattr(self.diffusion_model, "codex_config", None)
        if cfg is not None:
            cd = getattr(cfg, "context_dim", None)
            if isinstance(cd, int):
                expected_ctx_dim = cd
            elif isinstance(cd, (list, tuple)) and len(cd) > 0:
                # If multiple values are present, require the feature dim to be one of them
                expected_ctx_dim = set(int(v) for v in cd if isinstance(v, int))

        feat_dim = int(context.shape[-1])
        if isinstance(expected_ctx_dim, int) and feat_dim != expected_ctx_dim:
            raise ValueError(
                f"UNet context feature dim mismatch: got {feat_dim}, expected {expected_ctx_dim}. "
                f"Hint: check SDXL concatenation (should be 2048) and UNet context_dim."
            )
        if isinstance(expected_ctx_dim, set) and expected_ctx_dim and feat_dim not in expected_ctx_dim:
            raise ValueError(
                f"UNet context feature dim {feat_dim} not in allowed set {sorted(expected_ctx_dim)}."
            )

        # If the UNet expects class/ADM conditioning, ensure 'y' is present in kwargs
        # Note: Flux uses pooled vector differently than SDXL ADM, so we only error if
        # the model needs y but doesn't have it. Extra y for flow models is OK.
        needs_y = getattr(self.diffusion_model, "num_classes", None) is not None
        has_y = "y" in kwargs and isinstance(kwargs["y"], torch.Tensor)
        if needs_y and not has_y:
            raise ValueError(
                f"UNet ADM conditioning mismatch: num_classes={getattr(self.diffusion_model,'num_classes',None)} "
                f"but y_present={has_y}. Ensure SDXL pooled vector is wired as 'y'."
            )

        # If present, enforce y feature size to match ADM channels declared in config
        if needs_y and has_y:
            y = kwargs["y"]
            adm_channels = None
            inner_cfg = getattr(self.diffusion_model, "codex_config", None)
            if inner_cfg is not None:
                adm_channels = getattr(inner_cfg, "adm_in_channels", None)
            if isinstance(adm_channels, int) and adm_channels > 0 and int(y.shape[1]) != adm_channels:
                raise ValueError(
                    f"UNet ADM feature mismatch: got y.shape[1]={int(y.shape[1])}, expected adm_in_channels={adm_channels}. "
                    f"Hint: SDXL vector should be [pooled_g, time_ids(6*256)], typically 1280+1536=2816."
                )

        if _DEBUG_LOGGER.isEnabledFor(logging.DEBUG):
            emit_backend_message(
                "apply_model",
                logger=__name__,
                level=logging.DEBUG,
                x_shape=tuple(x.shape),
                t_shape=tuple(t.shape) if hasattr(t, "shape") else (1,),
                context_shape=tuple(context.shape),
                y_shape=getattr(kwargs.get("y", None), "shape", None),
                dtype=str(dtype),
            )

        model_output = self.diffusion_model(
            xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds
        )

        if debug_enabled and debug_count < debug_limit:
            emit_backend_message(f"[sampler-model-debug] {_tensor_stats('model_output', model_output)}", logger=__name__)
        return self.predictor.calculate_denoised(sigma, model_output, x)

    def memory_required(self, input_shape):
        area = input_shape[0] * input_shape[2] * input_shape[3]
        compute_dtype_size = torch.empty((), dtype=self.computation_dtype).element_size()
        # Preserve the numerical fp32 floor for predictor-result pressure even
        # though apply_model() no longer casts the full denoiser output eagerly.
        dtype_size = max(compute_dtype_size, torch.empty((), dtype=torch.float32).element_size())

        if attention.attention_function in [attention.attention_pytorch, attention.attention_xformers]:
            scaler = 1.28
        else:
            scaler = 1.65
            if attention.get_attn_precision() == torch.float32:
                dtype_size = max(dtype_size, torch.empty((), dtype=torch.float32).element_size())

        return scaler * area * dtype_size * 16384

    def forward(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        """Standard forward method that delegates to apply_model.
        
        This is required by the memory management system which expects a 'forward' method.
        """
        return self.apply_model(x, t, c_concat=c_concat, c_crossattn=c_crossattn, 
                                control=control, transformer_options=transformer_options, **kwargs)

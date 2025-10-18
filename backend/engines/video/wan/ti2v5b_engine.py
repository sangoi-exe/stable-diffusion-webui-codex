"""Wan 2.2 TI2V 5B (Text/Image → Video) engine.

Capabilities: TXT2VID and IMG2VID using the 5B hybrid model.

MVP: Validates parameters, applies presets, loads model directory, emits
progress events, and raises explicit errors if components are missing.
Actual denoising/generation will be wired in a subsequent step.
"""

from __future__ import annotations

import time
from typing import Iterator, List

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from backend.core.engine_interface import EngineCapabilities, TaskType
from backend.core.requests import InferenceEvent, Txt2VidRequest, Img2VidRequest, ProgressEvent, ResultEvent
from backend.core.param_registry import parse_params
from backend.core.exceptions import UnsupportedTaskError, EngineLoadError, EngineExecutionError
from backend.core.enums import Mode
from backend.core.presets import apply_preset_to_request

from ...base import DiffusionEngine
from .loader import WanLoader, WanComponents
from .schedulers import apply_sampler_scheduler, allowed_samplers_for_engine


class WanTI2V5BEngine(DiffusionEngine):
    engine_id = "wan_ti2v_5b"
    _loader: WanLoader | None = None
    _comp: WanComponents | None = None

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2VID, TaskType.IMG2VID),
            model_types=("wan-2.2-ti2v-5b",),
            precision=("fp16", "bf16"),
            extras={"notes": "Stub engine; integration pending with WAN 2.2 TI2V 5B"},
        )

    def txt2vid(self, request: Txt2VidRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self.ensure_loaded()
        mode = self._resolve_mode(request)
        params = parse_params(self.engine_id, TaskType.TXT2VID, request)
        params, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.TXT2VID, params, logger=self._logger)

        start = time.perf_counter()
        comp = self._ensure_components()
        yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing WAN TI2V‑5B")

        if comp.pipeline is None:
            raise EngineExecutionError(
                "WAN TI2V‑5B requires Diffusers WanPipeline locally. Put weights at a Diffusers-style folder and set Checkpoint to it, or install diffusers main with WanPipeline."
            )

        pipe = comp.pipeline
        # Apply sampler/scheduler mapping (with warnings if unsupported)
        outcome = apply_sampler_scheduler(pipe, str(getattr(params, "sampler", "Automatic")), str(getattr(params, "scheduler", "Automatic")))
        for w in outcome.warnings:
            self._logger.warning("wan_ti2v_5b: %s", w)
        try:
            # Map parameters
            num_steps = int(getattr(params, "steps", 12) or 12)
            num_frames = int(getattr(params, "num_frames", 16) or 16)
            width = int(getattr(params, "width", 768) or 768)
            height = int(getattr(params, "height", 432) or 432)
            guidance = getattr(params, "guidance_scale", None)

            # Run pipeline (txt2vid)
            yield ProgressEvent(stage="run", percent=5.0, message="Running pipeline")
            out = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                height=height,
                width=width,
                guidance_scale=guidance,
            )

            images = list(out.frames[0]) if hasattr(out, "frames") else []
            elapsed = time.perf_counter() - start
            vram = 0
            if torch is not None and torch.cuda.is_available():  # pragma: no cover
                try:
                    vram = int(torch.cuda.max_memory_allocated() // (1024 * 1024))
                except Exception:
                    vram = 0
            info = {
                "engine": self.engine_id,
                "task": "txt2vid",
                "elapsed": round(elapsed, 3),
                "vram_mb_peak": vram,
                "frames": len(images),
                "width": width,
                "height": height,
                "steps": num_steps,
                "num_frames": num_frames,
                "sampler_in": outcome.sampler_in,
                "scheduler_in": outcome.scheduler_in,
                "sampler_effective": outcome.sampler_effective,
                "scheduler_effective": outcome.scheduler_effective,
            }
            yield ResultEvent(payload={"images": images, "info": self._to_json(info)})
        except Exception as exc:  # noqa: BLE001
            raise EngineExecutionError(str(exc)) from exc

    def img2vid(self, request: Img2VidRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self.ensure_loaded()
        mode = self._resolve_mode(request)
        params = parse_params(self.engine_id, TaskType.IMG2VID, request)
        params, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.IMG2VID, params, logger=self._logger)

        if getattr(params, "init_image", None) is None:
            raise EngineExecutionError("wan_ti2v_5b/img2vid requires 'init_image'")

        comp = self._ensure_components()
        yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing WAN TI2V‑5B")

        if comp.pipeline is None:
            raise EngineExecutionError(
                "WAN TI2V‑5B requires Diffusers WanPipeline locally. Put weights at a Diffusers-style folder and set Checkpoint to it, or install diffusers main with WanPipeline."
            )

        pipe = comp.pipeline
        outcome = apply_sampler_scheduler(pipe, str(getattr(params, "sampler", "Automatic")), str(getattr(params, "scheduler", "Automatic")))
        for w in outcome.warnings:
            self._logger.warning("wan_ti2v_5b: %s", w)
        try:
            start = time.perf_counter()
            num_steps = int(getattr(params, "steps", 12) or 12)
            num_frames = int(getattr(params, "num_frames", 16) or 16)
            width = int(getattr(params, "width", 768) or 768)
            height = int(getattr(params, "height", 432) or 432)
            guidance = getattr(params, "guidance_scale", None)
            init = getattr(params, "init_image", None)

            yield ProgressEvent(stage="run", percent=5.0, message="Running pipeline")
            out = pipe(
                image=init,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                height=height,
                width=width,
                guidance_scale=guidance,
            )

            images = list(out.frames[0]) if hasattr(out, "frames") else []
            elapsed = time.perf_counter() - start
            vram = 0
            if torch is not None and torch.cuda.is_available():  # pragma: no cover
                try:
                    vram = int(torch.cuda.max_memory_allocated() // (1024 * 1024))
                except Exception:
                    vram = 0
            info = {
                "engine": self.engine_id,
                "task": "img2vid",
                "elapsed": round(elapsed, 3),
                "vram_mb_peak": vram,
                "frames": len(images),
                "width": width,
                "height": height,
                "steps": num_steps,
                "num_frames": num_frames,
                "sampler_in": outcome.sampler_in,
                "scheduler_in": outcome.scheduler_in,
                "sampler_effective": outcome.sampler_effective,
                "scheduler_effective": outcome.scheduler_effective,
            }
            yield ResultEvent(payload={"images": images, "info": self._to_json(info)})
        except Exception as exc:  # noqa: BLE001
            raise EngineExecutionError(str(exc)) from exc

    def load(self, model_ref: str, **options: object) -> None:  # type: ignore[override]
        if not model_ref:
            raise EngineLoadError("wan_ti2v_5b load failed: empty model_ref")
        self._loader = WanLoader(self._logger)
        comp = self._loader.load(model_ref, device=str(options.get("device", "auto")), dtype=str(options.get("dtype", "fp16")))
        self._comp = comp
        self.mark_loaded()

    # ------------------------------------------------------------------
    def _ensure_components(self) -> WanComponents:
        if self._comp is None or self._loader is None:
            raise EngineLoadError("WAN components not loaded; call load(model_ref) first")
        return self._comp

    @staticmethod
    def _resolve_mode(request: object) -> Mode:
        try:
            meta = getattr(request, "metadata", {})
            mode_value = meta.get("mode", "Normal") if isinstance(meta, dict) else "Normal"
            return Mode(mode_value)  # type: ignore[arg-type]
        except Exception:
            return Mode.NORMAL

    def _simulate_progress(self, steps: int) -> Iterator[InferenceEvent]:
        # Emits deterministic progress events and basic VRAM metrics; no actual gen
        total = max(1, int(steps))
        for i in range(total):
            pct = round(100.0 * (i + 1) / total, 2)
            vram = 0
            if torch is not None and torch.cuda.is_available():  # pragma: no cover
                try:
                    vram = int(torch.cuda.max_memory_allocated() // (1024 * 1024))
                except Exception:
                    vram = 0
            yield ProgressEvent(stage="denoise", percent=pct, step=i + 1, total_steps=total, data={"vram_mb": vram})

    @staticmethod
    def _to_json(obj: object) -> str:
        try:
            import json

            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return "{}"

"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Task orchestration for the dedicated SeedVR2 video-upscale endpoint.
Owns task status, inference-gate coordination, cancellation boundaries, event forwarding, terminal result storage, and final task-snapshot
cleanup while the canonical use case owns source-video processing and export.

Symbols (top-level; keep in sync; no ghosts):
- `run_video_upscale_task` (function): Starts one dedicated SeedVR2 video-upscale task worker.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

from apps.backend.core.requests import ProgressEvent, ResultEvent
from apps.backend.interfaces.api.inference_gate import acquire_inference_gate, release_inference_gate, single_flight_enabled
from apps.backend.interfaces.api.public_errors import (
    build_cancelled_task_error,
    build_missing_result_task_error,
    build_public_task_error,
)
from apps.backend.interfaces.api.task_registry import TaskCancelMode, TaskEntry
from apps.backend.runtime.logging import get_backend_logger
from apps.backend.use_cases.video_upscale import (
    VideoUpscaleRequest,
    cleanup_video_upscale_source_admission,
    run_video_upscale,
)


logger = get_backend_logger(__name__)


def run_video_upscale_task(
    *,
    task_id: str,
    request: VideoUpscaleRequest,
    entry: TaskEntry,
) -> None:
    """Start the task worker for one dedicated SeedVR2 video-upscale request."""

    def push(event: dict[str, object]) -> None:
        entry.push_event(event)

    push({"type": "status", "stage": "queued"})

    def worker() -> None:
        acquired = False
        success = False
        try:
            if single_flight_enabled():
                push({"type": "status", "stage": "waiting_for_inference"})

            acquired = acquire_inference_gate(should_cancel=lambda: bool(entry.cancel_requested))
            if not acquired:
                entry.error = build_cancelled_task_error()
                return

            push({"type": "status", "stage": "running"})
            from apps.backend.interfaces.api.device_selection import apply_primary_device

            apply_primary_device(request.device)
            if entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE:
                entry.error = build_cancelled_task_error()
                return

            received_result = False
            for event in run_video_upscale(
                request,
                should_cancel=lambda: bool(
                    entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE
                ),
            ):
                if isinstance(event, ProgressEvent):
                    if entry.cancel_requested and entry.cancel_mode is TaskCancelMode.IMMEDIATE:
                        entry.error = build_cancelled_task_error()
                        return
                    progress_payload: dict[str, object] = {
                        "type": "progress",
                        "stage": event.stage,
                        "percent": event.percent,
                        "step": event.step,
                        "total_steps": event.total_steps,
                        "eta_seconds": event.eta_seconds,
                    }
                    if event.message is not None:
                        progress_payload["message"] = event.message
                    if event.data:
                        progress_payload["data"] = dict(event.data)
                    push(progress_payload)
                    continue
                if not isinstance(event, ResultEvent):
                    raise RuntimeError(f"SeedVR2 video-upscale emitted unsupported event {type(event).__name__}.")

                payload = event.payload
                if not isinstance(payload, Mapping):
                    raise RuntimeError("SeedVR2 video-upscale result payload must be an object.")
                info = payload.get("info")
                video = payload.get("video")
                if not isinstance(info, Mapping) or not isinstance(video, Mapping):
                    raise RuntimeError("SeedVR2 video-upscale completed without info and video result objects.")
                rel_path = video.get("rel_path")
                mime = video.get("mime")
                if not isinstance(rel_path, str) or not rel_path or not isinstance(mime, str) or not mime:
                    raise RuntimeError("SeedVR2 video-upscale completed without a saved video artifact descriptor.")

                entry.result = {
                    "status": "completed",
                    "result": {
                        "images": [],
                        "info": dict(info),
                        "video": {"rel_path": rel_path, "mime": mime},
                    },
                }
                received_result = True

            if not received_result:
                raise RuntimeError("SeedVR2 video-upscale completed without a result event.")
            success = True
        except Exception as err:  # pragma: no cover - surfaces runtime failures through the task contract
            entry.error = build_public_task_error(err)
            success = False
        finally:
            try:
                cleanup_video_upscale_source_admission(request.source)
            except Exception as cleanup_error:
                logger.error(
                    "video-upscale task-work cleanup failed (task_id=%s): %s",
                    task_id,
                    cleanup_error,
                    exc_info=False,
                )
                if entry.error is None:
                    entry.error = build_public_task_error(cleanup_error)
                success = False
            if success:
                result_obj = entry.result.get("result") if isinstance(entry.result, dict) else None
                if not isinstance(result_obj, dict):
                    entry.error = build_missing_result_task_error("video upscale task")
                    success = False
            entry.mark_finished(success=success)
            entry.schedule_cleanup(task_id)
            if acquired:
                try:
                    release_inference_gate()
                except Exception as exc:
                    logger.warning(
                        "inference gate release failed in video-upscale worker (task_id=%s): %s",
                        task_id,
                        exc,
                        exc_info=False,
                    )

    threading.Thread(target=worker, name=f"video-upscale-task-{task_id}", daemon=True).start()

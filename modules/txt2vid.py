from __future__ import annotations

import gradio as gr

from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html

from backend.core.engine_interface import TaskType
from backend.core.orchestrator import InferenceOrchestrator
from backend.core.requests import Txt2VidRequest, ResultEvent


def txt2vid_from_json(id_task: str, request: gr.Request, payload, *args):
    if isinstance(payload, str):
        import json as _json
        payload = _json.loads(payload)
    if not isinstance(payload, dict) or payload.get("__strict_version") != 1:
        raise ValueError("Invalid or missing __strict_version in payload; frontend must send strict JSON")

    prompt = payload.get('txt2vid_prompt', '')
    negative_prompt = payload.get('txt2vid_neg_prompt', '')
    width = int(payload.get('txt2vid_width', 768))
    height = int(payload.get('txt2vid_height', 432))
    steps = int(payload.get('txt2vid_steps', 12))
    fps = int(payload.get('txt2vid_fps', 24))
    num_frames = int(payload.get('txt2vid_num_frames', 16))
    sampler = str(payload.get('txt2vid_sampling', 'Euler'))
    scheduler = str(payload.get('txt2vid_scheduler', 'Automatic'))
    seed = int(payload.get('txt2vid_seed', -1))
    motion_strength = float(payload.get('txt2vid_motion_strength', 0.5)) if 'txt2vid_motion_strength' in payload else None

    req = Txt2VidRequest(
        task=TaskType.TXT2VID,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        fps=fps,
        num_frames=num_frames,
        sampler=sampler,
        scheduler=scheduler,
        seed=seed,
        guidance_scale=float(payload.get('txt2vid_cfg_scale', 7.0)),
        metadata={
            "mode": getattr(shared.opts, 'codex_mode', 'Normal'),
            "styles": payload.get('txt2vid_styles', []),
        },
        motion_strength=motion_strength,
    )

    engine_key = getattr(shared.opts, 'codex_engine', 'hunyuan_video')
    model_ref = getattr(shared.opts, 'sd_model_checkpoint', None)

    orch = InferenceOrchestrator()
    images = []
    info_js = "{}"
    engine_opts = {"export_video": bool(getattr(shared.opts, 'codex_export_video', False))}
    for ev in orch.run(TaskType.TXT2VID, str(engine_key), req, model_ref=model_ref, engine_options=engine_opts):
        if isinstance(ev, ResultEvent):
            payload = ev.payload or {}
            images = payload.get("images", [])
            info_js = payload.get("info", "{}")

    if opts.do_not_show_images:
        images = []

    return images, info_js, plaintext_to_html(""), plaintext_to_html("", classname="comments")

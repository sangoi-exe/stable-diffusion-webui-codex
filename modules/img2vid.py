from __future__ import annotations

import gradio as gr
from PIL import Image

from modules import images as _images
from modules.shared import opts
import modules.shared as shared
from modules.ui import plaintext_to_html

from backend.core.engine_interface import TaskType
from backend.core.orchestrator import InferenceOrchestrator
from backend.core.requests import Img2VidRequest, ResultEvent


def img2vid_from_json(id_task: str, request: gr.Request, init_image, *rest):
    if not rest:
        raise ValueError("Missing JSON payload in img2vid_from_json")
    payload = rest[-1]
    if isinstance(payload, str):
        import json as _json
        payload = _json.loads(payload)
    if not isinstance(payload, dict) or payload.get("__strict_version") != 1:
        raise ValueError("Invalid or missing __strict_version in payload; frontend must send strict JSON")

    prompt = payload.get('img2vid_prompt', '')
    negative_prompt = payload.get('img2vid_neg_prompt', '')
    width = int(payload.get('img2vid_width', 768))
    height = int(payload.get('img2vid_height', 432))
    steps = int(payload.get('img2vid_steps', 12))
    fps = int(payload.get('img2vid_fps', 24))
    num_frames = int(payload.get('img2vid_num_frames', 16))
    sampler = str(payload.get('img2vid_sampling', 'Euler'))
    scheduler = str(payload.get('img2vid_scheduler', 'Automatic'))
    seed = int(payload.get('img2vid_seed', -1))
    motion_strength = float(payload.get('img2vid_motion_strength', 0.5)) if 'img2vid_motion_strength' in payload else None

    base_image = None
    try:
        if isinstance(init_image, Image.Image):
            base_image = init_image
        elif hasattr(init_image, 'name'):
            base_image = _images.read(init_image.name)
    except Exception:
        base_image = None

    req = Img2VidRequest(
        task=TaskType.IMG2VID,
        prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=base_image,
        width=width,
        height=height,
        steps=steps,
        fps=fps,
        num_frames=num_frames,
        sampler=sampler,
        scheduler=scheduler,
        seed=seed,
        guidance_scale=float(payload.get('img2vid_cfg_scale', 7.0)),
        motion_strength=motion_strength,
        metadata={
            "mode": getattr(shared.opts, 'codex_mode', 'Normal'),
            "styles": payload.get('img2vid_styles', []),
        },
    )

    engine_key = getattr(shared.opts, 'codex_engine', 'svd')
    model_ref = getattr(shared.opts, 'sd_model_checkpoint', None)

    orch = InferenceOrchestrator()
    images = []
    info_js = "{}"
    engine_opts = {"export_video": bool(getattr(shared.opts, 'codex_export_video', False))}
    for ev in orch.run(TaskType.IMG2VID, str(engine_key), req, model_ref=model_ref, engine_options=engine_opts):
        if isinstance(ev, ResultEvent):
            payload = ev.payload or {}
            images = payload.get("images", [])
            info_js = payload.get("info", "{}")

    if opts.do_not_show_images:
        images = []

    return images, info_js, plaintext_to_html(""), plaintext_to_html("", classname="comments")

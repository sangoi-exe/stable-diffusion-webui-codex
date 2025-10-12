from __future__ import annotations

import time
import modules.shared as shared
from modules.progress import current_task


class ProgressService:
    """Compute progress/ETA and (optionally) current preview image.

    Returns plain dicts suitable for response models; caller can wrap in pydantic.
    """

    def __init__(self, media_service):
        self.media = media_service

    def compute(self, skip_current_image: bool = False):
        # Derived from modules/api/api.py progressapi
        if shared.state.job_count == 0:
            return {
                "progress": 0,
                "eta_relative": 0,
                "state": shared.state.dict(),
                "textinfo": shared.state.textinfo,
                "current_task": current_task,
                "current_image": None,
            }

        progress = 0.01

        if shared.state.job_count > 0:
            progress += shared.state.job_no / shared.state.job_count
        if shared.state.sampling_steps > 0:
            progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start / max(progress, 1e-6))
        eta_relative = eta - time_since_start

        progress = min(progress, 1)

        shared.state.set_current_image()

        current_image = None
        if shared.state.current_image and not skip_current_image:
            current_image = self.media.encode_image(shared.state.current_image)

        return {
            "progress": progress,
            "eta_relative": eta_relative,
            "state": shared.state.dict(),
            "current_image": current_image,
            "textinfo": shared.state.textinfo,
            "current_task": current_task,
        }


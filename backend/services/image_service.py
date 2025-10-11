from __future__ import annotations

from contextlib import closing, nullcontext
from typing import Callable, Optional

import modules.shared as shared
from modules.processing import process_images
from modules.processing import StableDiffusionProcessing  # for type hints only
from modules.api.api import process_extra_images  # reuse existing helper
from modules.progress import add_task_to_queue, start_task, finish_task


class ImageService:
    """Encapsulates Stable Diffusion image generation flows.

    This service owns the common execution path for txt2img/img2img style
    requests, without knowledge of HTTP frameworks or request models.
    """

    def execute_generation(
        self,
        p_factory: Callable[[], StableDiffusionProcessing],
        *,
        script_runner,
        selectable_scripts,
        script_args,
        task_id: str,
        job_label: str,
        outpath_samples: str,
        outpath_grids: str,
        prepare_p: Optional[Callable[[StableDiffusionProcessing], None]] = None,
        queue_lock=None,
    ):
        """Create, configure and execute a processing pipeline.

        Returns the `Processed` object from `process_images()` or script runner.
        """

        add_task_to_queue(task_id)

        lock_ctx = queue_lock if queue_lock is not None else nullcontext()
        with lock_ctx:
            with closing(p_factory()) as p:
                p.is_api = True
                p.scripts = script_runner
                p.outpath_grids = outpath_grids
                p.outpath_samples = outpath_samples

                if callable(prepare_p):
                    prepare_p(p)

                try:
                    shared.state.begin(job=job_label)
                    start_task(task_id)
                    if selectable_scripts is not None:
                        p.script_args = script_args
                        processed = script_runner.run(p, *p.script_args)
                    else:
                        p.script_args = tuple(script_args)
                        processed = process_images(p)
                    process_extra_images(processed)
                    finish_task(task_id)
                finally:
                    shared.state.end()
                    shared.total_tqdm.clear()

        return processed


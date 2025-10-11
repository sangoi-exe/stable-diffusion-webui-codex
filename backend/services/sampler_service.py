from __future__ import annotations

from fastapi.exceptions import HTTPException

from modules import sd_samplers


class SamplerService:
    """Sampler/Scheduler resolution and validation."""

    @staticmethod
    def resolve(sampler_name_or_index, scheduler: str | None):
        """Return a tuple (sampler_name, scheduler_name).

        Normalizes aliases and indices using sd_samplers helpers.
        """
        sampler, sched = sd_samplers.get_sampler_and_scheduler(
            sampler_name_or_index, scheduler
        )
        return sampler, sched

    @staticmethod
    def ensure_valid_sampler(name: str) -> str:
        config = sd_samplers.all_samplers_map.get(name, None)
        if config is None:
            raise HTTPException(status_code=400, detail="Sampler not found")
        return name


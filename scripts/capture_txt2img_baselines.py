from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ORIGINAL_MODULE_NAME = __name__
MODULE_STEM = Path(__file__).stem
if ORIGINAL_MODULE_NAME.endswith(".py"):
    module_obj = sys.modules.get(ORIGINAL_MODULE_NAME)
    if module_obj is not None:
        sys.modules[MODULE_STEM] = module_obj
        globals()["__name__"] = MODULE_STEM
elif ORIGINAL_MODULE_NAME == "__main__":
    module_obj = sys.modules.get(ORIGINAL_MODULE_NAME)
    if module_obj is not None:
        sys.modules.setdefault(MODULE_STEM, module_obj)

# Ensure Stable Diffusion command-line parser tolerates script-specific flags.
os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")

try:
    from tqdm import tqdm
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "tqdm is required to run capture_txt2img_baselines.py. "
        "Install it with `pip install tqdm`."
    ) from exc

LOGGER = logging.getLogger("capture_txt2img_baselines")
DEFAULT_OUTPUT_DIR = Path("tests/backend/fixtures/txt2img")


def _as_float(value: Any, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Field "{field_name}" must be a number, received: {value!r}') from exc


def _as_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Field "{field_name}" must be an integer, received: {value!r}') from exc


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False
    raise ValueError(f"Cannot interpret boolean value from {value!r}")


def _sanitize_name(name: str) -> str:
    """Return a filesystem-friendly slug for the scenario name."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "scenario"


@dataclass(frozen=True)
class HiresSettings:
    hr_scale: float = 2.0
    hr_upscaler: Optional[str] = None
    hr_second_pass_steps: int = 0
    hr_resize_x: int = 0
    hr_resize_y: int = 0
    hr_checkpoint_name: Optional[str] = None
    hr_additional_modules: Optional[List[str]] = None
    hr_sampler_name: Optional[str] = None
    hr_scheduler: Optional[str] = None
    hr_prompt: str = ""
    hr_negative_prompt: str = ""
    hr_cfg: float = 1.0
    hr_distilled_cfg: float = 3.5

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HiresSettings":
        payload = dict(payload or {})

        modules = payload.get("hr_additional_modules")
        if modules is not None and not isinstance(modules, list):
            raise ValueError("hires.hr_additional_modules must be a list if provided.")

        return cls(
            hr_scale=_as_float(payload.get("hr_scale", cls.hr_scale), field_name="hires.hr_scale"),
            hr_upscaler=payload.get("hr_upscaler"),
            hr_second_pass_steps=_as_int(payload.get("hr_second_pass_steps", cls.hr_second_pass_steps), field_name="hires.hr_second_pass_steps"),
            hr_resize_x=_as_int(payload.get("hr_resize_x", cls.hr_resize_x), field_name="hires.hr_resize_x"),
            hr_resize_y=_as_int(payload.get("hr_resize_y", cls.hr_resize_y), field_name="hires.hr_resize_y"),
            hr_checkpoint_name=payload.get("hr_checkpoint_name"),
            hr_additional_modules=list(modules) if modules else None,
            hr_sampler_name=payload.get("hr_sampler_name"),
            hr_scheduler=payload.get("hr_scheduler"),
            hr_prompt=str(payload.get("hr_prompt", cls.hr_prompt)),
            hr_negative_prompt=str(payload.get("hr_negative_prompt", cls.hr_negative_prompt)),
            hr_cfg=_as_float(payload.get("hr_cfg", cls.hr_cfg), field_name="hires.hr_cfg"),
            hr_distilled_cfg=_as_float(payload.get("hr_distilled_cfg", cls.hr_distilled_cfg), field_name="hires.hr_distilled_cfg"),
        )


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt: str
    negative_prompt: str
    seed: int
    subseed: int
    subseed_strength: float
    sampler_name: str
    scheduler: Optional[str]
    steps: int
    cfg_scale: float
    distilled_cfg_scale: float
    width: int
    height: int
    batch_size: int
    n_iter: int
    enable_hr: bool
    denoising_strength: float
    hires: HiresSettings
    override_settings: Dict[str, Any]
    styles: List[str]
    script_args: List[Any]
    script_index: int
    tags: List[str]
    firstpass_image: Optional[str]
    raw: Dict[str, Any] = field(repr=False)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Scenario":
        required_keys = ("name", "prompt", "seed", "sampler_name", "steps", "cfg_scale", "width", "height")
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise ValueError(f"Scenario is missing required keys: {', '.join(missing)}")

        override_settings = payload.get("override_settings") or {}
        if not isinstance(override_settings, dict):
            raise ValueError("override_settings must be a mapping of option name to value.")

        script_args = payload.get("script_args") or []
        if not isinstance(script_args, list):
            raise ValueError("script_args must be a list when provided.")

        script_index = _as_int(payload.get("script_index", 0), field_name="script_index")
        
        styles = payload.get("styles") or []
        if not isinstance(styles, list):
            raise ValueError("styles must be a list when provided.")

        tags = payload.get("tags") or []
        if not isinstance(tags, list):
            raise ValueError("tags must be a list when provided.")

        firstpass_image = payload.get("firstpass_image")
        if firstpass_image is not None and not isinstance(firstpass_image, str):
            raise ValueError("firstpass_image, when provided, must be a string path.")

        hires_payload = payload.get("hires", {})
        hires = HiresSettings.from_dict(hires_payload)

        return cls(
            name=str(payload["name"]).strip() or "unnamed-scenario",
            prompt=str(payload["prompt"]),
            negative_prompt=str(payload.get("negative_prompt", "")),
            seed=_as_int(payload["seed"], field_name="seed"),
            subseed=_as_int(payload.get("subseed", -1), field_name="subseed"),
            subseed_strength=_as_float(payload.get("subseed_strength", 0.0), field_name="subseed_strength"),
            sampler_name=str(payload["sampler_name"]),
            scheduler=payload.get("scheduler"),
            steps=_as_int(payload["steps"], field_name="steps"),
            cfg_scale=_as_float(payload["cfg_scale"], field_name="cfg_scale"),
            distilled_cfg_scale=_as_float(payload.get("distilled_cfg_scale", 3.5), field_name="distilled_cfg_scale"),
            width=_as_int(payload["width"], field_name="width"),
            height=_as_int(payload["height"], field_name="height"),
            batch_size=_as_int(payload.get("batch_size", 1), field_name="batch_size"),
            n_iter=_as_int(payload.get("n_iter", 1), field_name="n_iter"),
            enable_hr=_as_bool(payload.get("enable_hr", False)),
            denoising_strength=_as_float(payload.get("denoising_strength", 0.75), field_name="denoising_strength"),
            hires=hires,
            override_settings=dict(override_settings),
            styles=list(styles),
            script_args=list(script_args),
            script_index=script_index,
            tags=list(tags),
            firstpass_image=firstpass_image,
            raw=dict(payload),
        )

    @property
    def slug(self) -> str:
        return _sanitize_name(self.name)

    def to_processing_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "styles": list(self.styles),
            "seed": self.seed,
            "subseed": self.subseed,
            "subseed_strength": self.subseed_strength,
            "sampler_name": self.sampler_name,
            "scheduler": self.scheduler,
            "batch_size": self.batch_size,
            "n_iter": self.n_iter,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "distilled_cfg_scale": self.distilled_cfg_scale,
            "width": self.width,
            "height": self.height,
            "enable_hr": self.enable_hr,
            "denoising_strength": self.denoising_strength,
            "hr_scale": self.hires.hr_scale,
            "hr_upscaler": self.hires.hr_upscaler,
            "hr_second_pass_steps": self.hires.hr_second_pass_steps,
            "hr_resize_x": self.hires.hr_resize_x,
            "hr_resize_y": self.hires.hr_resize_y,
            "hr_checkpoint_name": self.hires.hr_checkpoint_name,
            "hr_additional_modules": self.hires.hr_additional_modules,
            "hr_sampler_name": self.hires.hr_sampler_name,
            "hr_scheduler": self.hires.hr_scheduler,
            "hr_prompt": self.hires.hr_prompt,
            "hr_negative_prompt": self.hires.hr_negative_prompt,
            "hr_cfg": self.hires.hr_cfg,
            "hr_distilled_cfg": self.hires.hr_distilled_cfg,
            "override_settings": dict(self.override_settings),
            "do_not_save_samples": True,
            "do_not_save_grid": True,
        }
        return kwargs


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture deterministic txt2img baselines driven by Stable Diffusion "
            "WebUI Forge's backend runtime."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSON configuration describing baseline scenarios.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where artifacts will be written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenario_filters",
        help="Run only the specified scenario(s); may be supplied multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and report planned runs without executing generation.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining scenarios even if one fails.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scenario directories instead of aborting when outputs exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_configuration(config_path: Path) -> List[Scenario]:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON configuration {config_path}: {exc}") from exc

    scenarios_payload = raw.get("scenarios")
    if not isinstance(scenarios_payload, list) or not scenarios_payload:
        raise SystemExit("Configuration must define a non-empty \"scenarios\" list.")

    scenarios: List[Scenario] = []
    for item in scenarios_payload:
        if not isinstance(item, dict):
            raise SystemExit("Each scenario entry must be an object.")
        scenarios.append(Scenario.from_dict(item))
    return scenarios


def filter_scenarios(scenarios: List[Scenario], filters: Optional[Iterable[str]]) -> List[Scenario]:
    if not filters:
        return scenarios
    filter_set = {item.strip().lower() for item in filters}
    filtered = [scenario for scenario in scenarios if scenario.name.lower() in filter_set or scenario.slug.lower() in filter_set]
    unknown = filter_set.difference({scenario.name.lower() for scenario in filtered}.union({scenario.slug.lower() for scenario in filtered}))
    if unknown:
        LOGGER.warning("Ignoring unknown scenario filters: %s", ", ".join(sorted(unknown)))
    if not filtered:
        raise SystemExit("No scenarios matched the provided filters.")
    return filtered


def ensure_runtime_initialized() -> None:
    from modules import initialize, shared
    from modules.sd_models import model_data
    from modules_forge import main_entry

    if getattr(shared, "opts", None) is not None and getattr(shared, "sd_model", None):
        LOGGER.debug("Shared runtime already initialised; reusing existing state.")
        return

    LOGGER.info("Initialising Stable Diffusion runtime (imports, models, extensions).")
    initialize.imports()
    initialize.check_versions()
    initialize.initialize()

    if not model_data.forge_loading_parameters.get("checkpoint_info"):
        main_entry.refresh_model_loading_parameters()

    if shared.sd_model is None or not model_data.forge_loading_parameters.get("checkpoint_info"):
        raise SystemExit(
            "Stable Diffusion model failed to load. "
            "Verify that a checkpoint is selected (UI > Settings > Stable Diffusion > Model) "
            "or include \"sd_model_checkpoint\" inside scenario override_settings."
        )


def _load_firstpass_image(path: str):
    from PIL import Image

    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"firstpass_image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def _assign_firstpass_image(processing, scenario: Scenario) -> None:
    if not scenario.firstpass_image:
        return
    processing.firstpass_image = _load_firstpass_image(scenario.firstpass_image)
    LOGGER.debug("Loaded firstpass image for scenario %s from %s", scenario.name, scenario.firstpass_image)


def execute_scenario(scenario: Scenario, output_dir: Path) -> Dict[str, Any]:
    from PIL import PngImagePlugin
    from modules import scripts, shared
    from modules.processing import StableDiffusionProcessingTxt2Img, process_images
    from modules_forge import main_entry

    requested_checkpoint = scenario.override_settings.get("sd_model_checkpoint")
    requested_modules = scenario.override_settings.get("forge_additional_modules")
    previous_checkpoint = shared.opts.data.get("sd_model_checkpoint")
    previous_modules = getattr(shared.opts, "forge_additional_modules", None)
    checkpoint_changed = False
    modules_changed = False

    if requested_modules is not None:
        modules_changed = main_entry.modules_change(requested_modules, save=False, refresh=True)

    if requested_checkpoint:
        checkpoint_changed = main_entry.checkpoint_change(requested_checkpoint, save=False, refresh=True)

    try:
        kwargs = scenario.to_processing_kwargs()
        kwargs.setdefault("outpath_samples", str(output_dir))
        kwargs.setdefault("outpath_grids", str(output_dir / "grids"))

        with closing(StableDiffusionProcessingTxt2Img(**kwargs)) as processing:
            processing.scripts = scripts.scripts_txt2img
            combined_script_args = [scenario.script_index]
            combined_script_args.extend(scenario.script_args)
            processing.script_args = combined_script_args
            processing.user = "baseline-capture"
            _assign_firstpass_image(processing, scenario)

            shared.state.interrupted = False
            shared.state.skipped = False
            shared.state.job = f"txt2img baseline – {scenario.name}"

            LOGGER.debug("Starting generation for scenario %s", scenario.name)

            processed = scripts.scripts_txt2img.run(processing, *processing.script_args)
            if processed is None:
                processed = process_images(processing)

        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        extras_dir = output_dir / "extra_images"
        images_dir.mkdir(exist_ok=True, parents=True)
        extras_dir.mkdir(exist_ok=True, parents=True)

        run_metadata: Dict[str, Any] = {
            "scenario": scenario.raw,
            "scenario_slug": scenario.slug,
            "infotexts": processed.infotexts,
            "info": processed.info,
            "comments": processed.comments,
            "js": json.loads(processed.js()),
            "output": [],
            "extra_images": [],
        }

        for index, image in enumerate(processed.images):
            filename = images_dir / f"{scenario.slug}_{index:02d}.png"
            png_info = PngImagePlugin.PngInfo()
            infotext = processed.infotexts[index] if index < len(processed.infotexts) else processed.info
            if infotext:
                png_info.add_text("parameters", infotext)
            image.save(filename, format="PNG", pnginfo=png_info)
            run_metadata["output"].append({"path": str(filename), "infotext": infotext})

        for index, image in enumerate(processed.extra_images):
            filename = extras_dir / f"{scenario.slug}_extra_{index:02d}.png"
            image.save(filename, format="PNG")
            run_metadata["extra_images"].append(str(filename))

        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

        LOGGER.info(
            "Captured scenario %s (%d primary image(s), %d extra image(s)).",
            scenario.name,
            len(processed.images),
            len(processed.extra_images),
        )

        return run_metadata

    finally:
        if checkpoint_changed:
            restore_target = previous_checkpoint or shared.opts.data.get("sd_model_checkpoint")
            if restore_target:
                main_entry.checkpoint_change(restore_target, save=False, refresh=True)
            else:
                main_entry.refresh_model_loading_parameters()
        if modules_changed:
            restore_modules = previous_modules if isinstance(previous_modules, (list, tuple)) else []
            main_entry.modules_change(restore_modules, save=False, refresh=True)


def maybe_overwrite_directory(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        return
    if overwrite:
        LOGGER.debug("Overwriting existing directory %s", path)
        for entry in path.iterdir():
            if entry.is_file():
                entry.unlink()
            elif entry.is_dir():
                maybe_overwrite_directory(entry, overwrite=True)
                entry.rmdir()
        return
    raise SystemExit(f"Output directory already exists for scenario: {path}. Use --overwrite to replace it.")


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    LOGGER.setLevel(level)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    LOGGER.info("Arguments: %s", args)

    scenarios = load_configuration(args.config)
    scenarios = filter_scenarios(scenarios, args.scenario_filters)

    LOGGER.info("Loaded %d scenario(s) from %s", len(scenarios), args.config)

    if args.dry_run:
        LOGGER.info("Dry run requested. The following scenarios would be executed:")
        for scenario in scenarios:
            LOGGER.info(
                " - %s (slug=%s, steps=%s, sampler=%s, width=%s, height=%s, enable_hr=%s)",
                scenario.name,
                scenario.slug,
                scenario.steps,
                scenario.sampler_name,
                scenario.width,
                scenario.height,
                scenario.enable_hr,
            )
        return 0

    LOGGER.info("Ensuring runtime is initialised")
    ensure_runtime_initialized()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    for scenario in tqdm(scenarios, desc="txt2img baselines", unit="scenario"):
        LOGGER.info(
            "Executing scenario '%s' -> %s",
            scenario.name,
            args.output_dir / scenario.slug,
        )
        scenario_dir = args.output_dir / scenario.slug
        try:
            maybe_overwrite_directory(scenario_dir, overwrite=args.overwrite)
            execute_scenario(scenario, scenario_dir)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Scenario %s failed: %s", scenario.name, exc)
            failures.append(scenario.name)
            if not args.continue_on_error:
                LOGGER.error("Aborting due to failure; rerun with --continue-on-error to skip failures.")
                break

    if failures:
        LOGGER.error("The following scenarios failed: %s", ", ".join(failures))
        return 1

    LOGGER.info("All scenarios completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

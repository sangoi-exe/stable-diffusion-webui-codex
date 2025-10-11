import pathlib
import sys
import types

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide lightweight stubs for heavy modules imported by txt2img.
modules_package = sys.modules.setdefault("modules", types.ModuleType("modules"))
modules_package.__path__ = []

devices_module = types.ModuleType("modules.devices")
devices_module.cpu = torch.device("cpu")
devices_module.torch_gc = lambda: None
sys.modules.setdefault("modules.devices", devices_module)

sd_samplers_module = types.ModuleType("modules.sd_samplers")
sd_samplers_module.create_sampler = lambda *args, **kwargs: None
sys.modules.setdefault("modules.sd_samplers", sd_samplers_module)

class _SkipWritingToConfig:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

sd_models_module = types.ModuleType("modules.sd_models")
sd_models_module.apply_token_merging = lambda *args, **kwargs: None
sd_models_module.SkipWritingToConfig = _SkipWritingToConfig
sd_models_module.forge_model_reload = lambda: None
sys.modules.setdefault("modules.sd_models", sd_models_module)

sd_samplers_common_module = types.ModuleType("modules.sd_samplers_common")
sd_samplers_common_module.approximation_indexes = {}
sd_samplers_common_module.decode_first_stage = lambda _model, batch: batch
sd_samplers_common_module.images_tensor_to_samples = lambda tensor, *_args, **_kwargs: tensor
sys.modules.setdefault("modules.sd_samplers_common", sd_samplers_common_module)

rng_module = types.ModuleType("modules.rng")


class _StubImageRNG:
    created = []

    def __init__(self, shape, seeds, **_kwargs):
        self.shape = tuple(int(dim) for dim in shape)
        self.seeds = list(seeds)
        self.calls = 0
        self.__class__.created.append(
            {
                "shape": self.shape,
                "seeds": self.seeds,
                "kwargs": dict(_kwargs),
            }
        )

    def next(self):
        self.calls += 1
        batch = len(self.seeds) or 1
        return torch.zeros((batch, *self.shape), dtype=torch.float32)


rng_module.ImageRNG = _StubImageRNG
sys.modules.setdefault("modules.rng", rng_module)

extra_networks_module = types.ModuleType("modules.extra_networks")
extra_networks_module.invoke_log = []


def _activate(processing, data):
    extra_networks_module.invoke_log.append((processing, data))


extra_networks_module.activate = _activate
sys.modules.setdefault("modules.extra_networks", extra_networks_module)

scripts_module = types.ModuleType("modules.scripts")


class _PostSampleArgs:
    def __init__(self, samples):
        self.samples = samples


scripts_module.PostSampleArgs = _PostSampleArgs
sys.modules.setdefault("modules.scripts", scripts_module)

shared_module = types.ModuleType("modules.shared")
shared_module.opts = types.SimpleNamespace(sd_vae_encode_method="Full")
shared_module.shared = types.SimpleNamespace(device=torch.device("cpu"), state=None)
sys.modules.setdefault("modules.shared", shared_module)

modules_forge_module = types.ModuleType("modules_forge")
main_entry_stub = types.ModuleType("modules_forge.main_entry")
main_entry_stub.modules_change = lambda *args, **kwargs: False
main_entry_stub.checkpoint_change = lambda *args, **kwargs: False
main_entry_stub.refresh_model_loading_parameters = lambda: None
modules_forge_module.main_entry = main_entry_stub
sys.modules.setdefault("modules_forge", modules_forge_module)
sys.modules.setdefault("modules_forge.main_entry", main_entry_stub)

for missing in ("psutil", "gguf"):
    if missing not in sys.modules:
        sys.modules[missing] = types.ModuleType(missing)

if "pytz" not in sys.modules:
    sys.modules["pytz"] = types.ModuleType("pytz")

if "piexif" not in sys.modules:
    piexif_module = types.ModuleType("piexif")
    piexif_module.__path__ = []
    sys.modules["piexif"] = piexif_module
else:
    piexif_module = sys.modules["piexif"]

sys.modules.setdefault("piexif.helper", types.ModuleType("piexif.helper"))
piexif_module.dump = getattr(piexif_module, "dump", lambda *args, **kwargs: b"")
piexif_module.insert = getattr(piexif_module, "insert", lambda *args, **kwargs: None)
piexif_module.load = getattr(piexif_module, "load", lambda _data: {})
piexif_module.ExifIFD = getattr(
    piexif_module,
    "ExifIFD",
    types.SimpleNamespace(UserComment=37510),
)

piexif_helper_module = sys.modules["piexif.helper"]

class _UserComment:
    @staticmethod
    def dump(*_args, **_kwargs):
        return b""

    @staticmethod
    def load(_data):
        return ""

piexif_helper_module.UserComment = getattr(
    piexif_helper_module,
    "UserComment",
    _UserComment,
)

if "pillow_avif" not in sys.modules:
    sys.modules["pillow_avif"] = types.ModuleType("pillow_avif")

if "gradio" not in sys.modules:
    sys.modules["gradio"] = types.ModuleType("gradio")

if "safetensors" not in sys.modules:
    safetensors_module = types.ModuleType("safetensors")
    sys.modules["safetensors"] = safetensors_module
    sys.modules["safetensors.torch"] = types.ModuleType("safetensors.torch")

if "torchdiffeq" not in sys.modules:
    torchdiffeq_module = types.ModuleType("torchdiffeq")
    torchdiffeq_module.odeint = lambda *args, **kwargs: None
    sys.modules["torchdiffeq"] = torchdiffeq_module

if "torchsde" not in sys.modules:
    torchsde_module = types.ModuleType("torchsde")
    torchsde_module._brownian = types.SimpleNamespace(
        brownian_interval=types.SimpleNamespace(_randn=lambda *args, **kwargs: None)
    )
    sys.modules["torchsde"] = torchsde_module

if "diffusers" not in sys.modules:
    diffusers_module = types.ModuleType("diffusers")
    diffusers_module.__path__ = []
    diffusers_module.FlowMatchEulerDiscreteScheduler = type(
        "FlowMatchEulerDiscreteScheduler",
        (),
        {},
    )
    sys.modules["diffusers"] = diffusers_module
else:
    diffusers_module = sys.modules["diffusers"]

pipelines_module = types.ModuleType("diffusers.pipelines")
pipelines_module.__path__ = []
sys.modules.setdefault("diffusers.pipelines", pipelines_module)

flux_module = types.ModuleType("diffusers.pipelines.flux")
flux_module.__path__ = []
sys.modules.setdefault("diffusers.pipelines.flux", flux_module)

pipeline_flux_module = types.ModuleType("diffusers.pipelines.flux.pipeline_flux")
pipeline_flux_module.calculate_shift = lambda *args, **kwargs: None
sys.modules.setdefault("diffusers.pipelines.flux.pipeline_flux", pipeline_flux_module)

# Provide a lightweight memory management stub to avoid CUDA initialisation during import.
memory_management_stub = types.ModuleType("backend.memory_management")
memory_management_stub.get_torch_device = lambda: torch.device("cpu")
memory_management_stub.soft_empty_cache = lambda: None
memory_management_stub.text_encoder_device = lambda: torch.device("cpu")
memory_management_stub.unet_dtype = lambda: torch.float32
memory_management_stub.vae_dtype = lambda: torch.float32
memory_management_stub.mps_mode = lambda: False
sys.modules.setdefault("backend.memory_management", memory_management_stub)

if "gguf" in sys.modules:
    gguf_module = sys.modules["gguf"]
else:
    gguf_module = types.ModuleType("gguf")
    sys.modules["gguf"] = gguf_module

if not hasattr(gguf_module, "GGMLQuantizationType"):
    quant_names = [
        "Q2_K",
        "Q3_K",
        "Q4_0",
        "Q4_K",
        "Q4_1",
        "Q5_0",
        "Q5_1",
        "Q5_K",
        "Q6_K",
        "Q8_0",
        "BF16",
    ]

    gguf_module.GGMLQuantizationType = types.SimpleNamespace(
        **{name: name for name in quant_names}
    )

    class _Quant:
        @staticmethod
        def bake(_tensor):
            return None

        @staticmethod
        def dequantize_pytorch(tensor):
            return tensor

    for quant_name in quant_names:
        setattr(gguf_module, quant_name, _Quant)

from backend.diffusion_engine import txt2img


class DummyForgeObjects:
    def __init__(self):
        self.copies = 0
        self.vae = types.SimpleNamespace(latent_channels=4)

    def shallow_copy(self):
        self.copies += 1
        return self


class DummySampler:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def sample(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class DummyScripts:
    def __init__(self):
        self.calls = []
        self.post_samples = []
        self.before_batches = []
        self.process_batches = []

    def process_before_every_sampling(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def post_sample(self, _processing, args):
        self.post_samples.append(args.samples)

    def before_process_batch(self, _processing, **kwargs):
        self.before_batches.append(kwargs)

    def process_batch(self, _processing, **kwargs):
        self.process_batches.append(kwargs)


class DummyRNG:
    def __init__(self, tensor):
        self.tensor = tensor
        self.call_count = 0

    def next(self):
        self.call_count += 1
        return self.tensor


class DummyProcessing:
    def __init__(self):
        self.sampler_name = "Euler"
        self.sd_model = types.SimpleNamespace(
            forge_objects_after_applying_lora=DummyForgeObjects(),
            forge_objects_original=DummyForgeObjects(),
            forge_objects=None,
            use_distilled_cfg_scale=False,
        )
        self.rng = DummyRNG(torch.zeros((1, 1, 1, 1)))
        self.modified_noise = None
        self.scripts = DummyScripts()
        self.extra_generation_params = {}
        self.enable_hr = False
        self.latent_scale_mode = None
        self.txt2img_image_conditioning_calls = []
        self.firstpass_image = None
        self.hr_additional_modules = None
        self.hr_checkpoint_name = "Use same checkpoint"
        self.hr_distilled_cfg = 0.0
        self.extra_network_data = ["network-a"]
        self.disable_extra_networks = False
        self.width = 512
        self.height = 512
        self.seeds = [1]
        self.subseeds = [2]
        self.subseed_strength = 0.0
        self.seed_resize_from_h = 0
        self.seed_resize_from_w = 0

    def get_token_merging_ratio(self):
        return 0.0

    def txt2img_image_conditioning(self, noise):
        self.txt2img_image_conditioning_calls.append(noise)
        return "image-conditioning"


@pytest.fixture(autouse=True)
def _isolate_opts(monkeypatch):
    monkeypatch.setattr(txt2img, "apply_token_merging", lambda *args, **kwargs: None)
    monkeypatch.setattr(txt2img.devices, "torch_gc", lambda: None)
    _StubImageRNG.created.clear()
    extra_networks_module.invoke_log.clear()
    return None


def _runtime(processing=None, **kwargs):
    if processing is None:
        processing = DummyProcessing()
    return txt2img.Txt2ImgRuntime(
        processing=processing,
        conditioning="cond",
        unconditional_conditioning="uncond",
        seeds=[1],
        subseeds=[2],
        subseed_strength=0.0,
        prompts=["prompt"],
        **kwargs,
    )


def test_generate_without_hr_runs_sampler(monkeypatch):
    processing = DummyProcessing()
    sampler = DummySampler(torch.ones((1, 1, 1, 1)))

    monkeypatch.setattr(txt2img.sd_samplers, "create_sampler", lambda *_, **__: sampler)
    merge_calls = []
    monkeypatch.setattr(
        txt2img,
        "apply_token_merging",
        lambda model, ratio: merge_calls.append((model, ratio)),
    )

    runtime = _runtime(processing=processing)
    result = runtime.generate()

    assert torch.equal(result, sampler.result)
    assert sampler.calls, "sampler.sample should be invoked"

    assert processing.sd_model.forge_objects is processing.sd_model.forge_objects_after_applying_lora
    assert processing.scripts.calls, "scripts.process_before_every_sampling should be called"
    called_noise = processing.scripts.calls[0][1]["noise"]
    assert torch.equal(called_noise, processing.txt2img_image_conditioning_calls[0])
    assert processing.sd_model.forge_objects_after_applying_lora.copies == 1
    assert processing.sd_model.forge_objects_original.copies == 1
    assert merge_calls == [(processing.sd_model, 0.0)]
    assert processing.scripts.before_batches[0]["prompts"] == ["prompt"]
    assert processing.scripts.process_batches[0]["seeds"] == [1]
    assert extra_networks_module.invoke_log == [(processing, ["network-a"])]


def test_generate_uses_first_pass_artifacts(monkeypatch):
    processing = DummyProcessing()
    processing.enable_hr = True
    sampler = DummySampler(torch.ones((1, 1, 1, 1)))
    monkeypatch.setattr(txt2img.sd_samplers, "create_sampler", lambda *_, **__: sampler)

    expected_samples = torch.ones((1, 1, 1, 1))
    expected_decoded = torch.full((1, 1, 1, 1), 2.0)

    monkeypatch.setattr(
        txt2img,
        "_prepare_first_pass_from_image",
        lambda _proc: (expected_samples, expected_decoded),
    )

    reload_calls = []
    monkeypatch.setattr(txt2img, "_reload_for_hires", lambda proc: reload_calls.append(proc))

    def sample_hr_pass(samples, decoded, seeds, subseeds, subseed_strength, prompts):
        assert torch.equal(samples, expected_samples)
        assert torch.equal(decoded, expected_decoded)
        assert seeds == [1]
        assert subseeds == [2]
        assert subseed_strength == 0.0
        assert prompts == ["prompt"]
        return "hires-result"

    processing.sample_hr_pass = sample_hr_pass

    runtime = _runtime(processing=processing)
    result = runtime.generate()

    assert result == "hires-result"
    assert not sampler.calls, "sampler.sample should not run when first pass data exists"
    assert reload_calls == [processing]


def test_generate_decodes_for_hr_when_needed(monkeypatch):
    processing = DummyProcessing()
    processing.enable_hr = True
    sampler = DummySampler(torch.ones((1, 1, 1, 1)))
    monkeypatch.setattr(txt2img.sd_samplers, "create_sampler", lambda *_, **__: sampler)

    monkeypatch.setattr(
        txt2img,
        "_prepare_first_pass_from_image",
        lambda _proc: (None, None),
    )

    decoded = torch.full((1, 1, 1, 1), 5.0)
    monkeypatch.setattr(
        txt2img,
        "_decode_latent_batch",
        lambda _model, samples, target_device=None: decoded,
    )

    reload_calls = []
    monkeypatch.setattr(txt2img, "_reload_for_hires", lambda proc: reload_calls.append(proc))

    def sample_hr_pass(samples, decoded_samples, *_args):
        assert torch.equal(samples, sampler.result)
        assert torch.equal(decoded_samples, decoded)
        return "decoded-hr"

    processing.sample_hr_pass = sample_hr_pass

    runtime = _runtime(processing=processing)
    result = runtime.generate()

    assert result == "decoded-hr"
    assert reload_calls == [processing]
    assert sampler.calls, "sampler.sample should run to produce the base latents"


def test_generate_prefers_modified_noise(monkeypatch):
    processing = DummyProcessing()

    class RecordingSampler(DummySampler):
        def __init__(self, result):
            super().__init__(result)
            self.used_noise = None

        def sample(self, processing_arg, noise, *args, **kwargs):
            self.used_noise = noise
            return super().sample(processing_arg, noise, *args, **kwargs)

    sampler = RecordingSampler(torch.ones((1, 1, 1, 1)))
    monkeypatch.setattr(txt2img.sd_samplers, "create_sampler", lambda *_, **__: sampler)

    expected_noise = torch.full((1, 1, 1, 1), 7.0)
    processing.modified_noise = expected_noise

    runtime = _runtime(processing=processing)
    result = runtime.generate()

    assert torch.equal(result, sampler.result)
    assert sampler.used_noise is expected_noise
    assert processing.modified_noise is None
    assert processing.txt2img_image_conditioning_calls[-1] is expected_noise


def test_generate_handles_missing_scripts(monkeypatch):
    processing = DummyProcessing()
    processing.scripts = None

    sampler_result = torch.ones((1, 1, 1, 1))
    monkeypatch.setattr(
        txt2img.sd_samplers,
        "create_sampler",
        lambda *_, **__: DummySampler(sampler_result),
    )

    runtime = _runtime(processing=processing)
    result = runtime.generate()

    assert torch.equal(result, sampler_result)


def test_reload_for_hires_switches_and_restores(monkeypatch):
    processing = DummyProcessing()
    processing.hr_additional_modules = ["module-new"]
    processing.hr_checkpoint_name = "checkpoint-new"
    processing.enable_hr = True
    processing.sd_model.use_distilled_cfg_scale = True
    processing.hr_distilled_cfg = 3.5

    modules_before = ["module-base"]
    checkpoint_before = "checkpoint-base"

    monkeypatch.setattr(
        txt2img.opts, "forge_additional_modules", list(modules_before), raising=False
    )
    monkeypatch.setattr(
        txt2img.opts, "sd_model_checkpoint", checkpoint_before, raising=False
    )

    modules_calls = []
    checkpoint_calls = []
    refresh_calls = []
    reload_calls = []

    def modules_change(values, save=True, refresh=True):
        modules_calls.append((tuple(values), save, refresh))
        txt2img.opts.forge_additional_modules = list(values)
        return values != modules_before

    def checkpoint_change(name, save=True, refresh=True):
        checkpoint_calls.append((name, save, refresh))
        changed = name != txt2img.opts.sd_model_checkpoint
        txt2img.opts.sd_model_checkpoint = name
        return changed

    monkeypatch.setattr(txt2img.main_entry, "modules_change", modules_change)
    monkeypatch.setattr(txt2img.main_entry, "checkpoint_change", checkpoint_change)
    monkeypatch.setattr(
        txt2img.main_entry,
        "refresh_model_loading_parameters",
        lambda: refresh_calls.append(True),
    )
    monkeypatch.setattr(
        txt2img.sd_models,
        "forge_model_reload",
        lambda: reload_calls.append(True),
    )

    txt2img._reload_for_hires(processing)

    assert modules_calls[0][0] == tuple(processing.hr_additional_modules)
    assert modules_calls[-1][0] == tuple(modules_before)
    assert checkpoint_calls[0][0] == processing.hr_checkpoint_name
    assert checkpoint_calls[-1][0] == checkpoint_before
    assert reload_calls == [True]
    assert len(refresh_calls) == 2
    assert txt2img.opts.forge_additional_modules == modules_before
    assert txt2img.opts.sd_model_checkpoint == checkpoint_before
    assert processing.firstpass_use_distilled_cfg_scale is processing.sd_model.use_distilled_cfg_scale
    assert (
        processing.extra_generation_params["Hires Distilled CFG Scale"]
        == processing.hr_distilled_cfg
    )


def test_generate_updates_shared_state_and_triggers_gc(monkeypatch):
    processing = DummyProcessing()
    sampler = DummySampler(torch.ones((1, 1, 1, 1)))
    monkeypatch.setattr(txt2img.sd_samplers, "create_sampler", lambda *_, **__: sampler)

    gc_calls = []
    monkeypatch.setattr(txt2img.devices, "torch_gc", lambda: gc_calls.append(None))

    nextjob_calls = []

    def _nextjob():
        nextjob_calls.append(True)

    shared_state = types.SimpleNamespace(current_latent=None, nextjob=_nextjob)
    monkeypatch.setattr(txt2img.shared, "state", shared_state, raising=False)

    runtime = _runtime(processing=processing)
    result = runtime.generate()

    assert shared_state.current_latent is result
    assert gc_calls, "devices.torch_gc should be invoked after sampling"
    assert nextjob_calls == [True]


def test_generate_initializes_rng(monkeypatch):
    processing = DummyProcessing()
    sampler = DummySampler(torch.zeros((1, 1, 1, 1)))
    monkeypatch.setattr(txt2img.sd_samplers, "create_sampler", lambda *_, **__: sampler)

    runtime = _runtime(processing=processing)
    runtime.generate()

    assert _StubImageRNG.created
    created = _StubImageRNG.created[0]
    assert created["shape"] == (4, 64, 64)
    assert created["seeds"] == [1]
    assert created["kwargs"]["subseeds"] == [2]


def test_generate_applies_post_sample_hooks(monkeypatch):
    processing = DummyProcessing()

    class RecordingScripts(DummyScripts):
        def post_sample(self, _processing, args):
            args.samples = torch.full_like(args.samples, 2.0)
            super().post_sample(_processing, args)

    processing.scripts = RecordingScripts()

    sampler_output = torch.ones((1, 1, 1, 1))
    sampler = DummySampler(sampler_output)
    monkeypatch.setattr(txt2img.sd_samplers, "create_sampler", lambda *_, **__: sampler)

    runtime = _runtime(processing=processing)
    result = runtime.generate()

    assert torch.equal(result, torch.full_like(sampler_output, 2.0))
    assert processing.scripts.post_samples, "post_sample should have been invoked"


def test_generate_skips_extra_networks_when_disabled(monkeypatch):
    processing = DummyProcessing()
    processing.disable_extra_networks = True
    sampler = DummySampler(torch.zeros((1, 1, 1, 1)))
    monkeypatch.setattr(txt2img.sd_samplers, "create_sampler", lambda *_, **__: sampler)

    runtime = _runtime(processing=processing)
    runtime.generate()

    assert not extra_networks_module.invoke_log

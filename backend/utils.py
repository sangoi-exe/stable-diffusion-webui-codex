import json
import os

import gguf
import safetensors.torch
from safetensors.torch import safe_open
from collections.abc import MutableMapping
import torch

import backend.misc.checkpoint_pickle
from backend.operations_gguf import ParameterGGUF
from collections.abc import MutableMapping


class KeyPrefixView(MutableMapping):
    """Lightweight mapping view that exposes `base` keys with a fixed prefix.

    - Does not materialize tensor values; delegates to `base[key_without_prefix]` on access.
    - Deletions and sets propagate to the underlying mapping.
    - Useful to avoid rebuilding huge state_dicts on CPU.
    """

    def __init__(self, base: MutableMapping, prefix: str):
        self._base = base
        self._prefix = prefix

    def _strip(self, k: str) -> str:
        if not k.startswith(self._prefix):
            raise KeyError(k)
        return k[len(self._prefix):]

    def __getitem__(self, k: str):
        return self._base[self._strip(k)]

    def __setitem__(self, k: str, v):
        self._base[self._strip(k)] = v

    def __delitem__(self, k: str):
        del self._base[self._strip(k)]

    def __iter__(self):
        for k in self._base.keys():
            yield f"{self._prefix}{k}"

    def __len__(self) -> int:
        try:
            return len(self._base.keys())
        except Exception:
            # Fallback: iterate
            return sum(1 for _ in self.__iter__())


class FilterPrefixView(MutableMapping):
    """View over keys under a given prefix, optionally re-prefixed lazily.

    - base: mapping with original keys (e.g., LazySafetensorsDict or KeyPrefixView)
    - prefix: filter only keys that start with this
    - new_prefix: keys presented by this view will start with new_prefix instead
    """

    def __init__(self, base: MutableMapping, prefix: str, new_prefix: str = ""):
        self._base = base
        self._prefix = prefix
        self._new_prefix = new_prefix

    def _to_base_key(self, k: str) -> str:
        if self._new_prefix and k.startswith(self._new_prefix):
            return self._prefix + k[len(self._new_prefix):]
        if not self._new_prefix and k.startswith(self._prefix):
            return k
        # Accept direct pass-through for loaders that query with base keys
        return k

    def __getitem__(self, k: str):
        return self._base[self._to_base_key(k)]

    def __setitem__(self, k: str, v):
        self._base[self._to_base_key(k)] = v

    def __delitem__(self, k: str):
        del self._base[self._to_base_key(k)]

    def __iter__(self):
        for k in self._base.keys():
            if k.startswith(self._prefix):
                out = self._new_prefix + k[len(self._prefix):]
                yield out

    def __len__(self) -> int:
        c = 0
        for _ in self.__iter__():
            c += 1
        return c


class CastOnGetView(MutableMapping):
    """Mapping view that casts tensor values on CPU to a target dtype on access.

    Useful to avoid fragile CPU bf16/fp16 ops during preprocessing. Only casts
    floating tensors matching `from_dtypes` and `device_type`.
    """

    def __init__(self, base: MutableMapping, *, device_type: str = "cpu", from_dtypes=None, to_dtype=None):
        import torch as _torch
        self._base = base
        self._device_type = device_type
        self._from = tuple(from_dtypes) if from_dtypes is not None else (_torch.bfloat16, _torch.float16)
        self._to = to_dtype or _torch.float32

    def __getitem__(self, k: str):
        import torch as _torch
        v = self._base[k]
        if isinstance(v, _torch.Tensor):
            try:
                if v.device.type == self._device_type and v.dtype in self._from:
                    return v.to(self._to)
            except Exception:
                return v
        return v

    def __setitem__(self, k: str, v):
        self._base[k] = v

    def __delitem__(self, k: str):
        del self._base[k]

    def __iter__(self):
        return iter(self._base.keys())

    def __len__(self) -> int:
        try:
            return len(self._base.keys())
        except Exception:
            return sum(1 for _ in self.__iter__())


def read_arbitrary_config(directory):
    config_path = os.path.join(directory, 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json file found in the directory: {directory}")

    with open(config_path, 'rt', encoding='utf-8') as file:
        config_data = json.load(file)

    return config_data


def load_torch_file(ckpt, safe_load=False, device=None):
    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        device = torch.device("cpu")

    checkpoint_path = str(ckpt)
    suffix = os.path.splitext(checkpoint_path)[1].lower()

    if suffix == ".safetensors":
        return LazySafetensorsDict(checkpoint_path, device=device.type)
    if suffix == ".gguf":
        return _load_gguf_state_dict(checkpoint_path)

    pl_sd = _load_pickled_checkpoint(checkpoint_path, device, safe_load)

    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    if "state_dict" in pl_sd:
        return pl_sd["state_dict"]
    return pl_sd


def _load_gguf_state_dict(path):
    reader = gguf.GGUFReader(path)
    state_dict = {}
    for tensor in reader.tensors:
        state_dict[str(tensor.name)] = ParameterGGUF(tensor)
    return state_dict


class LazySafetensorsDict(MutableMapping):
    """Lazy, mutable mapping backed by a .safetensors file.

    - Keys come from the file; values are loaded on demand with safe_open.get_tensor.
    - Supports overlay writes and deletions without touching the underlying file.
    - Device: only CPU tensors are produced (parity with previous loader).
    """

    def __init__(self, filepath: str, device: str = "cpu"):
        self.filepath = filepath
        self.device = device or "cpu"
        self._overlay = {}          # in-memory writes/overrides
        self._deleted = set()       # keys logically removed
        self._keys_cache = None     # cached set of underlying keys

    def _base_keys(self):
        if self._keys_cache is None:
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                self._keys_cache = set(f.keys())
        return self._keys_cache

    # Mapping protocol
    def __getitem__(self, key):
        if key in self._overlay:
            return self._overlay[key]
        if key in self._deleted:
            raise KeyError(key)
        if key not in self._base_keys():
            raise KeyError(key)
        with safe_open(self.filepath, framework="pt", device="cpu") as f:
            t = f.get_tensor(key)
        return t

    def __setitem__(self, key, value):
        self._overlay[key] = value
        if self._keys_cache is None and key not in self._deleted:
            # do not expand base key set; overlay keys are separate
            pass
        if key in self._deleted:
            self._deleted.remove(key)

    def __delitem__(self, key):
        if key in self._overlay:
            del self._overlay[key]
        else:
            # mark as deleted logically
            self._deleted.add(key)

    def __iter__(self):
        base = (k for k in self._base_keys() if k not in self._deleted)
        # overlay can shadow base
        for k in base:
            if k not in self._overlay:
                yield k
        for k in self._overlay.keys():
            yield k

    def __len__(self):
        return len([k for k in self._base_keys() if k not in self._deleted and k not in self._overlay]) + len(self._overlay)

    # Convenience helpers
    def keys(self):
        return list(iter(self))

    def items(self):
        for k in self:
            yield k, self[k]



def _load_pickled_checkpoint(path, device, safe_load):
    if safe_load and not _torch_supports_weights_only():
        print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
        safe_load = False

    if safe_load:
        return torch.load(path, map_location=device, weights_only=True)

    return torch.load(path, map_location=device, pickle_module=backend.misc.checkpoint_pickle)


def _torch_supports_weights_only():
    try:
        return 'weights_only' in torch.load.__code__.co_varnames
    except AttributeError:
        return False


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], torch.nn.Parameter(value, requires_grad=False))


def set_attr_raw(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)


def copy_to_param(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def get_attr_with_parent(obj, attr):
    attrs = attr.split(".")
    parent = obj
    name = None
    for name in attrs:
        parent = obj
        obj = getattr(obj, name)
    return parent, name, obj


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def tensor2parameter(x):
    if isinstance(x, torch.nn.Parameter):
        return x
    else:
        return torch.nn.Parameter(x, requires_grad=False)


def fp16_fix(x):
    # An interesting trick to avoid fp16 overflow
    # Source: https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/1114
    # Related: https://github.com/comfyanonymous/ComfyUI/blob/f1d6cef71c70719cc3ed45a2455a4e5ac910cd5e/comfy/ldm/flux/layers.py#L180

    if x.dtype in [torch.float16]:
        return x.clip(-32768.0, 32768.0)
    return x


def dtype_to_element_size(dtype):
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def nested_compute_size(obj, element_size):
    module_mem = 0

    if isinstance(obj, dict):
        for key in obj:
            module_mem += nested_compute_size(obj[key], element_size)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for i in range(len(obj)):
            module_mem += nested_compute_size(obj[i], element_size)
    elif isinstance(obj, torch.Tensor):
        module_mem += obj.nelement() * element_size

    return module_mem


def nested_move_to_device(obj, **kwargs):
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = nested_move_to_device(obj[key], **kwargs)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = nested_move_to_device(obj[i], **kwargs)
    elif isinstance(obj, tuple):
        obj = tuple(nested_move_to_device(i, **kwargs) for i in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.to(**kwargs)
    return obj


def get_state_dict_after_quant(model, prefix=''):
    for m in model.modules():
        if hasattr(m, 'weight') and hasattr(m.weight, 'bnb_quantized'):
            if not m.weight.bnb_quantized:
                original_device = m.weight.device
                m.cuda()
                m.to(original_device)

    sd = model.state_dict()
    sd = {(prefix + k): v.clone() for k, v in sd.items()}
    return sd


def beautiful_print_gguf_state_dict_statics(state_dict):
    from gguf.constants import GGMLQuantizationType
    type_counts = {}
    for k, v in state_dict.items():
        gguf_cls = getattr(v, 'gguf_cls', None)
        if gguf_cls is not None:
            type_name = gguf_cls.__name__
            if type_name in type_counts:
                type_counts[type_name] += 1
            else:
                type_counts[type_name] = 1
    print(f'GGUF state dict: {type_counts}')
    return

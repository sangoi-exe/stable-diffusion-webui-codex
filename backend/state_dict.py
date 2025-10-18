import torch
from backend.utils import FilterPrefixView
import logging
from . import torch_trace as _trace

_log = logging.getLogger("backend.state_dict")


def load_state_dict(model, sd, ignore_errors=[], log_name=None, ignore_start=None):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [x for x in missing if x not in ignore_errors]
    unexpected = [x for x in unexpected if x not in ignore_errors]

    if isinstance(ignore_start, str):
        missing = [x for x in missing if not x.startswith(ignore_start)]
        unexpected = [x for x in unexpected if not x.startswith(ignore_start)]

    log_name = log_name or type(model).__name__
    if len(missing) > 0:
        print(f'{log_name} Missing: {missing}')
        _log.debug("%s missing_count=%d", log_name, len(missing))
    if len(unexpected) > 0:
        print(f'{log_name} Unexpected: {unexpected}')
        _log.debug("%s unexpected_count=%d", log_name, len(unexpected))
    _trace.event("load_state_dict_done", name=log_name, missing=len(missing), unexpected=len(unexpected))
    return


def state_dict_has(sd, prefix):
    return any(x.startswith(prefix) for x in sd.keys())


def filter_state_dict_with_prefix(sd, prefix, new_prefix=''):
    """Return a lazy view filtered by `prefix`, optionally re-prefixed.

    Avoid materializing tensors while building the subset; deletion from the
    base mapping is skipped to preserve laziness and stability.
    """
    return FilterPrefixView(sd, prefix, new_prefix)


def try_filter_state_dict(sd, prefix_list, new_prefix=''):
    for prefix in prefix_list:
        if state_dict_has(sd, prefix):
            return filter_state_dict_with_prefix(sd, prefix, new_prefix)
    return FilterPrefixView(sd, "__no_such_prefix__/")  # empty view


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]
    return sd


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def safe_load_state_dict(model, sd, *, log_name=None):
    """Conservative loader: iterates model keys and copies tensors one by one.

    Avoids materializing all tensors and reduces device/dtype edge cases.
    Emits periodic trace events. Returns (missing, unexpected) like nn.Module.load_state_dict.
    """
    import torch
    from collections.abc import Mapping
    log_name = log_name or type(model).__name__

    model_state = model.state_dict()
    model_keys = list(model_state.keys())
    sd_keys = list(sd.keys()) if isinstance(sd, Mapping) and hasattr(sd, 'keys') else []

    missing = []
    loaded = 0
    for k in model_keys:
        try:
            t = sd[k]
        except Exception:
            missing.append(k)
            continue
        p = model_state.get(k)
        if not isinstance(t, torch.Tensor) or p is None:
            missing.append(k)
            continue
        try:
            if t.device.type != 'cpu':
                t_cpu = t.detach().to('cpu')
            else:
                t_cpu = t
            t_cast = t_cpu.to(dtype=p.dtype)
            p.copy_(t_cast.to(device=p.device))
        except Exception:
            _log.exception("safe_load_state_dict: failed key=%s", k)
            missing.append(k)
            continue
        loaded += 1
        if loaded % 200 == 0:
            _trace.event("load_state_dict_progress", name=log_name, loaded=loaded)

    unexpected = [k for k in sd_keys if k not in model_keys]
    if missing:
        print(f'{log_name} Missing: {missing}')
        _log.debug("%s missing_count=%d", log_name, len(missing))
    if unexpected:
        print(f'{log_name} Unexpected: {unexpected}')
        _log.debug("%s unexpected_count=%d", log_name, len(unexpected))
    _trace.event("load_state_dict_done", name=log_name, missing=len(missing), unexpected=len(unexpected))
    return missing, unexpected

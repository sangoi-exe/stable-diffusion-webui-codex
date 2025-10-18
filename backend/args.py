import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID")

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--all-in-fp32", action="store_true")
fp_group.add_argument("--all-in-fp16", action="store_true")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--unet-in-bf16", action="store_true")
fpunet_group.add_argument("--unet-in-fp16", action="store_true")
fpunet_group.add_argument("--unet-in-fp8-e4m3fn", action="store_true")
fpunet_group.add_argument("--unet-in-fp8-e5m2", action="store_true")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--vae-in-fp16", action="store_true")
fpvae_group.add_argument("--vae-in-fp32", action="store_true")
fpvae_group.add_argument("--vae-in-bf16", action="store_true")

parser.add_argument("--vae-in-cpu", action="store_true")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--clip-in-fp8-e4m3fn", action="store_true")
fpte_group.add_argument("--clip-in-fp8-e5m2", action="store_true")
fpte_group.add_argument("--clip-in-fp16", action="store_true")
fpte_group.add_argument("--clip-in-fp32", action="store_true")

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--attention-split", action="store_true")
attn_group.add_argument("--attention-quad", action="store_true")
attn_group.add_argument("--attention-pytorch", action="store_true")

upcast = parser.add_mutually_exclusive_group()
upcast.add_argument("--force-upcast-attention", action="store_true")
upcast.add_argument("--disable-attention-upcast", action="store_true")

parser.add_argument("--disable-xformers", action="store_true")

parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1)
parser.add_argument("--disable-ipex-hijack", action="store_true")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--always-gpu", action="store_true")
vram_group.add_argument("--always-high-vram", action="store_true")
vram_group.add_argument("--always-normal-vram", action="store_true")
vram_group.add_argument("--always-low-vram", action="store_true")
vram_group.add_argument("--always-no-vram", action="store_true")
vram_group.add_argument("--always-cpu", action="store_true")

parser.add_argument("--always-offload-from-vram", action="store_true")
parser.add_argument("--pytorch-deterministic", action="store_true")

parser.add_argument("--cuda-malloc", action="store_true")
parser.add_argument("--cuda-stream", action="store_true")
parser.add_argument("--pin-shared-memory", action="store_true")

parser.add_argument("--disable-gpu-warning", action="store_true")

# Online resolution for tokenizer assets (enabled by default). Use this flag to force strict offline mode.
parser.add_argument("--disable-online-tokenizer", action="store_true")

# Swap / device policy
parser.add_argument("--swap-policy", choices=["never", "cpu", "shared"], default="cpu",
                    help="Offload policy when VRAM is insufficient: never (abort), cpu (host), shared (pinned host)")
parser.add_argument("--swap-method", choices=["blocked", "async"], default="blocked",
                    help="Data transfer mode: blocked (no CUDA streams) or async (CUDA streams)")
parser.add_argument("--gpu-prefer-construct", action="store_true",
                    help="Prefer constructing models directly on GPU (fallback to policy on OOM)")

args = parser.parse_known_args()[0]

# Environment overrides (webui.settings.bat or process env)
# Prefer explicit dtype selections via env without requiring CLI flags.
_env = os.environ

def _truthy(v: str | None) -> bool:
    if not v:
        return False
    t = v.strip().lower()
    return t in ("1", "true", "yes", "on")


def _set_unet_dtype(val: str | None) -> None:
    if not val:
        return
    v = val.strip().lower()
    # Clear mutually exclusive flags first
    args.unet_in_bf16 = False
    args.unet_in_fp16 = False
    args.unet_in_fp8_e4m3fn = False
    args.unet_in_fp8_e5m2 = False
    if v == "bf16" or v == "bfloat16":
        args.unet_in_bf16 = True
    elif v == "fp16" or v == "half":
        args.unet_in_fp16 = True
    elif v in ("fp8_e4m3fn", "fp8-e4m3fn", "fp8_e4"):  # shorthand tolerant
        args.unet_in_fp8_e4m3fn = True
    elif v in ("fp8_e5m2", "fp8-e5m2", "fp8_e5"):
        args.unet_in_fp8_e5m2 = True
    elif v == "fp32" or v == "float" or v == "single":
        # No explicit unet_in_fp32 flag; leaving all False makes unet fall back to fp32.
        pass


def _set_vae_dtype(val: str | None) -> None:
    if not val:
        return
    v = val.strip().lower()
    # There are direct flags for VAE
    if v == "bf16" or v == "bfloat16":
        args.vae_in_bf16 = True
        args.vae_in_fp16 = False
        args.vae_in_fp32 = False
    elif v == "fp16" or v == "half":
        args.vae_in_bf16 = False
        args.vae_in_fp16 = True
        args.vae_in_fp32 = False
    elif v == "fp32" or v == "float" or v == "single":
        args.vae_in_bf16 = False
        args.vae_in_fp16 = False
        args.vae_in_fp32 = True


# Global overrides
if _truthy(_env.get("CODEX_VAE_IN_CPU")):
    args.vae_in_cpu = True

_set_unet_dtype(_env.get("CODEX_UNET_DTYPE") or _env.get("WEBUI_UNET_DTYPE"))
_set_vae_dtype(_env.get("CODEX_VAE_DTYPE") or _env.get("WEBUI_VAE_DTYPE"))

# Global all-fp32 override if user insists
if _truthy(_env.get("CODEX_ALL_IN_FP32")):
    args.all_in_fp32 = True

# Swap/device policy overrides via env
_sp = (_env.get("CODEX_SWAP_POLICY") or _env.get("WEBUI_SWAP_POLICY") or "").lower()
if _sp in ("never", "cpu", "shared"):
    args.swap_policy = _sp

_sm = (_env.get("CODEX_SWAP_METHOD") or _env.get("WEBUI_SWAP_METHOD") or "").lower()
if _sm in ("blocked", "async"):
    args.swap_method = _sm

if _truthy(_env.get("CODEX_GPU_PREFER_CONSTRUCT")):
    args.gpu_prefer_construct = True

# Some dynamic args that may be changed by webui rather than cmd flags.
dynamic_args = dict(
    embedding_dir='./embeddings',
    emphasis_name='original'
)

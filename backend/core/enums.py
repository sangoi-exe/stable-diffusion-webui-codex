from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    NORMAL = "Normal"
    LCM = "LCM"
    TURBO = "Turbo"
    LIGHTNING = "Lightning"


class EngineKey(str, Enum):
    SD15 = "sd15"
    SDXL = "sdxl"
    FLUX = "flux"
    SVD = "svd"
    HUNYUAN_VIDEO = "hunyuan_video"
    WAN_TI2V_5B = "wan_ti2v_5b"
    WAN_T2V_14B = "wan_t2v_14b"
    WAN_I2V_14B = "wan_i2v_14b"


class SamplerName(str, Enum):
    EULER_A = "Euler a"
    EULER = "Euler"
    DPM2M_SDE = "DPM++ 2M SDE"
    DPM2M = "DPM++ 2M"
    DDIM = "DDIM"
    PLMS = "PLMS"
    # Fallback for dynamic lists; engines should validate choices.
    AUTOMATIC = "Automatic"


class SchedulerName(str, Enum):
    AUTOMATIC = "Automatic"
    KARRAS = "Karras"
    SIMPLE = "Simple"
    EXPONENTIAL = "Exponential"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FLOAT8_E4M3FN = "float8-e4m3fn"
    FLOAT8_E5M2 = "float8-e5m2"
    NF4 = "nf4"
    FP4 = "fp4"
    GGUF = "gguf"

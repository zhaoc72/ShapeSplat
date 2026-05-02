from __future__ import annotations

from contextlib import nullcontext

import torch

from .device import resolve_runtime_device


def get_amp_config(cfg: dict) -> dict:
    """解析 AMP 配置。

    中文注释：AMP 默认关闭；只有用户显式 mixed_precision=true 且 CUDA 可用时才启用，
    避免默认数值行为变化影响复现。
    """
    info = resolve_runtime_device(cfg)
    enabled = bool(cfg.get("runtime", {}).get("mixed_precision", False)) and info.device.type == "cuda"
    name = str(cfg.get("runtime", {}).get("amp_dtype", "float16")).lower()
    dtype = torch.bfloat16 if name == "bfloat16" else torch.float16
    return {"enabled": enabled, "dtype": dtype, "device_type": info.device.type}


def amp_autocast(cfg: dict):
    amp = get_amp_config(cfg)
    if amp["enabled"]:
        return torch.autocast("cuda", dtype=amp["dtype"])
    return nullcontext()

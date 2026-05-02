from __future__ import annotations

import torch


def get_gpu_memory_info(device=None) -> dict:
    """返回显存信息；CPU 或无 CUDA 时不报错。"""
    if not torch.cuda.is_available():
        return {"available": False, "allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    dev = torch.device(device or "cuda:0")
    out = {
        "available": True,
        "device": str(dev),
        "allocated_mb": float(torch.cuda.memory_allocated(dev) / 1024**2),
        "reserved_mb": float(torch.cuda.memory_reserved(dev) / 1024**2),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated(dev) / 1024**2),
    }
    try:
        free, total = torch.cuda.mem_get_info(dev)
        out["free_mb"] = float(free / 1024**2)
        out["total_mb"] = float(total / 1024**2)
    except Exception as exc:
        out["mem_get_info_error"] = str(exc)
    return out


def log_gpu_memory(prefix: str = "") -> dict:
    # 中文注释：长批量实验中用于定位显存泄漏或峰值过高。
    info = get_gpu_memory_info()
    print(f"{prefix}GPU memory: {info}")
    return info


def empty_cuda_cache_if_needed(cfg: dict) -> None:
    if torch.cuda.is_available() and cfg.get("runtime", {}).get("empty_cache_between_images", True):
        torch.cuda.empty_cache()


def set_gpu_memory_fraction_if_configured(cfg: dict) -> None:
    frac = cfg.get("runtime", {}).get("max_gpu_memory_fraction")
    if frac is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(frac))

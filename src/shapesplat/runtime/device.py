from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class DeviceInfo:
    requested: str
    resolved: str
    device: torch.device
    cuda_available: bool
    cuda_device_index: int | None
    gpu_name: str | None
    compute_capability: tuple[int, int] | None
    torch_cuda_version: str | None
    warning_messages: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)


DEFAULT_RUNTIME = {
    "device": "auto",
    "cuda_device": 0,
    "prefer_gpu": True,
    "allow_cpu_fallback": False,
    "require_cuda_for_experiments": False,
    "mixed_precision": False,
    "amp_dtype": "float16",
    "benchmark_cudnn": True,
    "deterministic": False,
    "empty_cache_between_images": True,
    "max_gpu_memory_fraction": None,
    "warn_if_cuda_unavailable": True,
    "fail_if_cuda_requested_but_unavailable": True,
    "log_gpu_memory": True,
}


def _runtime_cfg(cfg: dict | str | None) -> dict:
    if isinstance(cfg, str):
        out = dict(DEFAULT_RUNTIME)
        out["device"] = cfg
        return out
    out = dict(DEFAULT_RUNTIME)
    if isinstance(cfg, dict):
        out.update(cfg.get("runtime", {}) or {})
        # 中文注释：兼容旧配置，只有 runtime.device 缺省时才读取顶层 device。
        if "device" not in (cfg.get("runtime", {}) or {}) and cfg.get("device") is not None:
            out["device"] = cfg.get("device")
    return out


def resolve_runtime_device(cfg: dict | str | None) -> DeviceInfo:
    """解析运行设备。

    中文注释：Windows + RTX 5090 环境不能只看 torch.cuda.is_available；
    这里先解析设备，真正 kernel 兼容性由 cuda_check 的 matmul/backward 测试确认。
    """
    rcfg = _runtime_cfg(cfg)
    requested = str(rcfg.get("device", "auto")).lower()
    cuda_index = int(rcfg.get("cuda_device", 0))
    allow_fallback = bool(rcfg.get("allow_cpu_fallback", False))
    prefer_gpu = bool(rcfg.get("prefer_gpu", True))
    cuda_available = bool(torch.cuda.is_available())
    warnings: list[str] = []
    errors: list[str] = []

    resolved = "cpu"
    if requested == "cpu":
        resolved = "cpu"
    elif requested == "auto":
        if cuda_available and prefer_gpu:
            resolved = f"cuda:{cuda_index}"
        else:
            resolved = "cpu"
            if rcfg.get("warn_if_cuda_unavailable", True) and prefer_gpu:
                warnings.append("CUDA is unavailable; auto runtime resolved to CPU.")
    elif requested.startswith("cuda"):
        if cuda_available:
            resolved = requested if ":" in requested else f"cuda:{cuda_index}"
        elif allow_fallback:
            resolved = "cpu"
            warnings.append("CUDA was requested but unavailable; allow_cpu_fallback=true so CPU is used.")
        else:
            errors.append("CUDA was requested but torch.cuda.is_available() is false. CPU fallback is disabled.")
            resolved = f"cuda:{cuda_index}"
    else:
        errors.append(f"Unknown runtime.device={requested!r}; expected auto/cuda/cpu.")

    device = torch.device(resolved)
    gpu_name = None
    cc = None
    torch_cuda = getattr(torch.version, "cuda", None)
    if device.type == "cuda" and cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(device)
            cc = tuple(int(x) for x in torch.cuda.get_device_capability(device))
            if "5090" in gpu_name or cc >= (12, 0):
                warnings.append("RTX 5090 / sm_120-like GPU detected; verify your PyTorch CUDA build supports Blackwell kernels.")
        except Exception as exc:
            errors.append(f"Failed to query CUDA device {device}: {exc}")

    return DeviceInfo(
        requested=requested,
        resolved=str(device),
        device=device,
        cuda_available=cuda_available,
        cuda_device_index=(device.index if device.type == "cuda" else None),
        gpu_name=gpu_name,
        compute_capability=cc,
        torch_cuda_version=torch_cuda,
        warning_messages=warnings,
        error_messages=errors,
    )


def get_torch_device(cfg: dict | str | None) -> torch.device:
    return resolve_runtime_device(cfg).device

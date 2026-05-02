from __future__ import annotations

import argparse
from pathlib import Path

from .cuda_check import assert_cuda_usable_if_required, check_cuda_runtime
from .device import resolve_runtime_device
from .memory import set_gpu_memory_fraction_if_configured
from .summary import make_runtime_summary, save_runtime_summary


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    """给脚本补充统一 GPU runtime 参数。"""
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--runtime-summary", action="store_true")


def runtime_overrides_from_args(args) -> dict:
    """提取可在 load_config 解析设备前应用的 runtime 覆盖。"""
    runtime: dict = {}
    if getattr(args, "device", None):
        runtime["device"] = args.device
    if getattr(args, "cuda_device", None) is not None:
        runtime["cuda_device"] = int(args.cuda_device)
    if getattr(args, "allow_cpu_fallback", False):
        runtime["allow_cpu_fallback"] = True
        runtime["require_cuda_for_experiments"] = False
    if getattr(args, "require_cuda", False):
        runtime["require_cuda_for_experiments"] = True
        runtime["device"] = "cuda"
    if getattr(args, "mixed_precision", False):
        runtime["mixed_precision"] = True
    return runtime


def apply_runtime_cli_overrides(cfg: dict, args) -> dict:
    """应用 CLI 覆盖。

    中文注释：只有用户显式传参时才覆盖配置，避免破坏旧实验命令默认行为。
    """
    runtime = cfg.setdefault("runtime", {})
    runtime.update(runtime_overrides_from_args(args))
    info = resolve_runtime_device(cfg)
    if info.error_messages:
        raise RuntimeError("; ".join(info.error_messages))
    cfg["device"] = str(info.device)
    return cfg


def prepare_runtime_for_run(cfg: dict, out_dir: str | Path | None = None, save_summary: bool = False) -> dict:
    """打印并可选保存 runtime summary。"""
    # 中文注释：显式 require_cuda 时，必须通过真实 CUDA kernel smoke test。
    assert_cuda_usable_if_required(cfg)
    set_gpu_memory_fraction_if_configured(cfg)
    info = resolve_runtime_device(cfg)
    cuda_report = check_cuda_runtime(cfg)
    print(f"Runtime device: requested={info.requested}, resolved={info.resolved}")
    print(f"CUDA smoke status: {cuda_report.get('status')}")
    if info.gpu_name:
        print(f"GPU: {info.gpu_name}, capability={info.compute_capability}, torch_cuda={info.torch_cuda_version}")
    for warning in info.warning_messages + cuda_report.get("warnings", []):
        print(f"runtime warning: {warning}")
    if save_summary and out_dir is not None:
        summary = make_runtime_summary(cfg)
        save_runtime_summary(summary, out_dir)
        return summary
    return {"device_info": info, "cuda_runtime": cuda_report}

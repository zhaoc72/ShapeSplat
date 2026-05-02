from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def run_nvidia_smi() -> dict:
    """Windows 下轻量调用 nvidia-smi；不存在时返回 available=false。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "path": None}
    try:
        proc = subprocess.run([exe], capture_output=True, text=True, timeout=10, shell=False)
        return {"available": proc.returncode == 0, "path": exe, "return_code": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-1000:]}
    except Exception as exc:
        return {"available": False, "path": exe, "error": str(exc)}


def collect_runtime_environment() -> dict:
    """收集 Python/Conda/PyTorch/CUDA 环境，便于 RTX 5090 本地诊断。"""
    info = {
        "os": os.name,
        "platform": platform.platform(),
        "python_version": sys.version,
        "sys_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": run_nvidia_smi(),
    }
    try:
        import torch

        gpus = []
        for idx in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
            gpus.append({"index": idx, "name": torch.cuda.get_device_name(idx), "compute_capability": list(torch.cuda.get_device_capability(idx))})
        info.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                "gpu_count": len(gpus),
                "gpus": gpus,
            }
        )
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info

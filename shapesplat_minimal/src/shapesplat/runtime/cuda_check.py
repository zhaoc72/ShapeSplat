from __future__ import annotations

import torch

from .device import resolve_runtime_device


def run_cuda_kernel_smoke_test(device: torch.device, matmul_size: int = 512) -> dict:
    """运行真实 CUDA kernel smoke test。

    中文注释：RTX 5090 上可能出现 CUDA 可见但 kernel 架构不匹配，所以这里实际做
    randn、matmul、backward 和 synchronize，而不是只相信 is_available。
    """
    result = {
        "passed": False,
        "device": str(device),
        "gpu_name": None,
        "compute_capability": None,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "error": None,
        "warnings": [],
    }
    if device.type != "cuda":
        result.update({"passed": True, "warnings": ["CPU mode; CUDA kernel smoke test skipped."]})
        return result
    try:
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["compute_capability"] = list(torch.cuda.get_device_capability(device))
        n = max(16, int(matmul_size))
        x = torch.randn((n, n), device=device, dtype=torch.float32, requires_grad=True)
        y = torch.randn((n, n), device=device, dtype=torch.float32)
        z = (x @ y).square().mean()
        z.backward()
        torch.cuda.synchronize(device)
        result["passed"] = True
        if result["compute_capability"] and tuple(result["compute_capability"]) >= (12, 0):
            result["warnings"].append("Blackwell/sm_120-class GPU detected; keep PyTorch and NVIDIA driver aligned.")
    except RuntimeError as exc:
        result["error"] = (
            f"{exc}\nCUDA kernel smoke test failed. PyTorch CUDA build may not support this GPU architecture; "
            "for RTX 5090 / sm_120, check Blackwell support, NVIDIA driver, and conda environment."
        )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def check_cuda_runtime(cfg: dict, matmul_size: int = 512) -> dict:
    info = resolve_runtime_device(cfg)
    row = {
        "requested": info.requested,
        "resolved": info.resolved,
        "cuda_available": info.cuda_available,
        "warnings": list(info.warning_messages),
        "errors": list(info.error_messages),
    }
    if info.device.type != "cuda":
        row["status"] = "cpu"
        row["smoke_test"] = {"passed": True, "warnings": ["CUDA test skipped because resolved device is CPU."]}
        return row
    smoke = run_cuda_kernel_smoke_test(info.device, matmul_size=matmul_size)
    row["smoke_test"] = smoke
    row["status"] = "cuda_ok" if smoke.get("passed") else "cuda_failed"
    if not smoke.get("passed"):
        row["errors"].append(smoke.get("error") or "CUDA smoke test failed.")
    return row


def assert_cuda_usable_if_required(cfg: dict) -> None:
    runtime = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    if not runtime.get("require_cuda_for_experiments", False):
        return
    report = check_cuda_runtime(cfg)
    if report.get("status") != "cuda_ok":
        raise RuntimeError(f"CUDA is required but not usable: {report}")

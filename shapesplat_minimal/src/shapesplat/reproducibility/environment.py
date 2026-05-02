from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

from shapesplat.runtime.env import collect_runtime_environment


def collect_environment_info() -> dict:
    """收集轻量环境信息，用于复现实验运行环境。"""

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
        },
    }
    try:
        import torch

        info.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "device_count": int(torch.cuda.device_count()),
            }
        )
        # 中文注释：run metadata 中记录更完整的 CUDA / conda / nvidia-smi 摘要，便于复现实验设备。
        info["runtime_environment"] = collect_runtime_environment()
    except Exception as exc:
        info.update({"torch_version": None, "cuda_available": False, "cuda_version": None, "device_count": 0, "torch_error": str(exc)})
    return info


def _git(root: Path, args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def try_collect_git_info(root: str | Path = ".") -> dict:
    """尝试收集 git commit/branch/dirty files；没有 git 时不影响实验。"""

    root = Path(root)
    try:
        commit = _git(root, ["rev-parse", "--short", "HEAD"])
        branch = _git(root, ["branch", "--show-current"])
        status = _git(root, ["status", "--short"])
        return {
            "available": True,
            "commit": commit,
            "branch": branch,
            "dirty_files": [line for line in status.splitlines() if line.strip()],
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}

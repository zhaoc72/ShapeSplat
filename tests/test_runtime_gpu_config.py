from __future__ import annotations

import sys
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.runtime.amp import amp_autocast, get_amp_config
from shapesplat.runtime.cuda_check import check_cuda_runtime
from shapesplat.runtime.device import resolve_runtime_device
from shapesplat.runtime.memory import get_gpu_memory_info
from shapesplat.runtime.summary import make_runtime_summary, save_runtime_summary


def _local_tmp(name: str) -> Path:
    root = ROOT / "outputs" / "test_runtime_gpu_config_tmp" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_resolve_runtime_device_cpu():
    info = resolve_runtime_device({"runtime": {"device": "cpu"}})
    assert info.device.type == "cpu"
    assert info.resolved == "cpu"


def test_resolve_runtime_device_auto():
    info = resolve_runtime_device({"runtime": {"device": "auto"}})
    assert info.device.type in {"cpu", "cuda"}


def test_cuda_check_cpu_mode():
    report = check_cuda_runtime({"runtime": {"device": "cpu"}})
    assert report["status"] == "cpu"
    assert report["smoke_test"]["passed"]


def test_amp_context_cpu():
    cfg = {"runtime": {"device": "cpu", "mixed_precision": True}}
    amp = get_amp_config(cfg)
    assert amp["device_type"] == "cpu"
    with amp_autocast(cfg):
        x = torch.ones(1)
    assert float(x.item()) == 1.0


def test_memory_info_no_cuda():
    info = get_gpu_memory_info()
    assert "allocated_mb" in info


def test_runtime_summary():
    cfg = {"runtime": {"device": "cpu"}}
    summary = make_runtime_summary(cfg)
    tmp_path = _local_tmp("runtime_summary")
    save_runtime_summary(summary, tmp_path)
    assert (tmp_path / "runtime_summary.json").exists()
    assert (tmp_path / "runtime_summary.md").exists()

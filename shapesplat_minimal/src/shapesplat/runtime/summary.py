from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .cuda_check import check_cuda_runtime
from .device import resolve_runtime_device
from .env import collect_runtime_environment
from .memory import get_gpu_memory_info


def _jsonable_device_info(info) -> dict:
    row = asdict(info)
    row["device"] = str(info.device)
    if row.get("compute_capability") is not None:
        row["compute_capability"] = list(row["compute_capability"])
    return row


def make_runtime_summary(cfg: dict) -> dict:
    """汇总运行时诊断，用于每次实验保存。"""
    info = resolve_runtime_device(cfg)
    return {
        "device_info": _jsonable_device_info(info),
        "cuda_runtime": check_cuda_runtime(cfg),
        "memory": get_gpu_memory_info(info.device if info.device.type == "cuda" else None),
        "environment": collect_runtime_environment(),
    }


def save_runtime_summary(summary: dict, out_dir: str | Path) -> None:
    """保存 JSON/Markdown runtime summary。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runtime_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    device = summary.get("device_info", {})
    cuda = summary.get("cuda_runtime", {})
    lines = [
        "# Runtime Summary",
        "",
        f"- requested device: {device.get('requested')}",
        f"- resolved device: {device.get('resolved')}",
        f"- gpu name: {device.get('gpu_name')}",
        f"- compute capability: {device.get('compute_capability')}",
        f"- torch cuda: {device.get('torch_cuda_version')}",
        f"- cuda smoke status: {cuda.get('status')}",
        "",
        "## Warnings",
    ]
    for w in device.get("warning_messages", []) + cuda.get("warnings", []):
        lines.append(f"- {w}")
    lines.append("")
    lines.append("## Errors")
    for e in device.get("error_messages", []) + cuda.get("errors", []):
        lines.append(f"- {e}")
    (out / "runtime_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

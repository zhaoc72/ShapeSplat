from __future__ import annotations

import json
from pathlib import Path


KEYS = [
    "AttrAcc",
    "AttrAcc_mean",
    "Leakage",
    "Leakage_mean",
    "InstIoU_mean",
    "InstIoU_mean_mean",
    "EditLocality",
    "EditLocality_mean",
    "num_success",
    "num_failed",
]


def _flatten(prefix: str, obj, out: dict) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}_{k}" if prefix else k, v, out)
    elif isinstance(obj, (int, float, str)):
        out[prefix] = obj


def extract_metrics_summary(out_dir: str | Path) -> dict:
    """从输出目录提取 registry 使用的关键指标摘要。"""

    out_dir = Path(out_dir)
    candidates = [
        "metrics.json",
        "summary.json",
        "per_method_summary.json",
        "ablation_summary.json",
        "baseline_summary.json",
        "external_baseline_summary.json",
    ]
    summary = {}
    for name in candidates:
        path = out_dir / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        flat = {}
        _flatten("", data, flat)
        # comparison summary 常以 method 为第一层；优先把 ours 的关键指标提升到顶层，
        # 这样 list_runs.py 可以直接显示主方法摘要，同时保留原始 method 前缀指标。
        if isinstance(data, dict) and isinstance(data.get("ours"), dict):
            ours_flat = {}
            _flatten("", data["ours"], ours_flat)
            for key, value in ours_flat.items():
                if key in KEYS or any(key.endswith(k) for k in KEYS):
                    summary[key] = value
        for key, value in flat.items():
            tail = key.split("_")[-1]
            if key in KEYS or tail in KEYS or any(key.endswith(k) for k in KEYS):
                summary[key] = value
        for key in KEYS:
            if key in flat:
                summary[key] = flat[key]
    return summary

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

import torch


def _jsonable(value: Any) -> Any:
    """把 tensor / 标量转成 JSON 可保存对象。"""
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return [float(v) for v in value.detach().cpu().flatten()]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_metrics_json(metrics: dict, path: str | Path) -> None:
    """保存 metrics.json，自动创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2, ensure_ascii=False)


def print_metrics(metrics: dict) -> None:
    """按 key 排序打印指标；list 只打印长度和均值，避免终端输出过长。"""
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, list):
            mean = sum(float(v) for v in value) / max(1, len(value))
            print(f"{key}: list(len={len(value)}, mean={mean:.6f})")
        else:
            print(f"{key}: {value}")


def merge_metrics(*dicts: dict) -> dict:
    """合并多个 metrics dict；如果 key 冲突，后者覆盖前者。"""
    out = {}
    for d in dicts:
        out.update(d)
    return out


def load_metrics_json(path: str | Path) -> dict:
    """读取 metrics.json。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_metrics(metrics: dict) -> dict:
    """把 metrics 展平成适合 CSV 的一行。

    list 类型保存 mean，避免 per-object 列数随物体数量变化。
    """
    row = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            row[key] = sum(float(v) for v in value) / max(1, len(value))
        elif isinstance(value, (int, float, str)) or value is None:
            row[key] = value
    return row


def save_metrics_csv(rows: list[dict], path: str | Path) -> None:
    """保存 metrics summary CSV，自动收集所有 key。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

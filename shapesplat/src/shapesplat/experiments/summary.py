from __future__ import annotations

import math
from pathlib import Path

from shapesplat.evaluation.report import flatten_metrics, save_metrics_csv, save_metrics_json


def _numeric_values(rows: list[dict], key: str) -> list[float]:
    vals = []
    for row in rows:
        if row.get("status") != "success":
            continue
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            vals.append(float(value))
    return vals


def summarize_rows(rows: list[dict]) -> dict:
    """对 per-image metrics 做轻量汇总。

    summary.csv 主要用于论文表格的初步整理：这里只统计数值型字段的 mean/std，
    list 型 per-object 指标会在 flatten_metrics 中转为 mean 后进入 per-image csv。
    """
    num_success = sum(1 for row in rows if row.get("status") == "success")
    num_failed = len(rows) - num_success
    summary: dict = {"num_total": len(rows), "num_success": num_success, "num_failed": num_failed}

    keys: list[str] = []
    for row in rows:
        flat = flatten_metrics(row)
        for key, value in flat.items():
            if key not in keys and isinstance(value, (int, float)):
                keys.append(key)

    for key in keys:
        vals = _numeric_values([flatten_metrics(row) for row in rows], key)
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        summary[key] = mean
        summary[f"{key}_std"] = math.sqrt(var)
    return summary


def save_batch_summary(rows: list[dict], out_dir: str | Path) -> dict:
    """保存 batch experiment 的 per-image 与 aggregate summary。"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    flat_rows = [flatten_metrics(row) for row in rows]
    summary = summarize_rows(rows)
    save_metrics_json(rows, out_path / "per_image_metrics.json")
    save_metrics_csv(flat_rows, out_path / "per_image_metrics.csv")
    save_metrics_json(summary, out_path / "summary.json")
    save_metrics_csv([summary], out_path / "summary.csv")
    return summary

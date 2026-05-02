from __future__ import annotations

import math
from pathlib import Path

from shapesplat.evaluation.report import flatten_metrics, save_metrics_csv, save_metrics_json


def summarize_comparison_rows(rows: list[dict]) -> dict:
    """按 method 汇总 comparison 结果。

    per_method_summary.csv 是后续论文表格的基础：每个 method 一行，指标给出
    mean/std。list 类型指标先在 flatten_metrics 中转成 mean。
    """

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method", "unknown")), []).append(row)

    summary: dict[str, dict] = {}
    for method, method_rows in grouped.items():
        ok_rows = [r for r in method_rows if r.get("status") == "success"]
        item: dict = {
            "method": method,
            "num_success": len(ok_rows),
            "num_failed": len(method_rows) - len(ok_rows),
        }
        keys: list[str] = []
        for row in ok_rows:
            for key, value in flatten_metrics(row).items():
                if key not in keys and isinstance(value, (int, float)) and math.isfinite(float(value)):
                    if key not in {"num_success", "num_failed"}:
                        keys.append(key)
        for key in keys:
            vals = []
            for row in ok_rows:
                value = flatten_metrics(row).get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    vals.append(float(value))
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                item[f"{key}_mean"] = mean
                item[f"{key}_std"] = math.sqrt(var)
        summary[method] = item
    return summary


def save_comparison_summary(rows: list[dict], out_dir: str | Path) -> dict:
    """保存 per-image rows 和 per-method summary。"""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_rows = [flatten_metrics(row) for row in rows]
    summary = summarize_comparison_rows(rows)
    save_metrics_json(rows, out_dir / "per_image_comparison.json")
    save_metrics_csv(flat_rows, out_dir / "per_image_comparison.csv")
    save_metrics_json(summary, out_dir / "per_method_summary.json")
    save_metrics_csv(list(summary.values()), out_dir / "per_method_summary.csv")
    return summary


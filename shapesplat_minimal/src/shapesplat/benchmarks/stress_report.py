from __future__ import annotations

import math
from pathlib import Path

from shapesplat.evaluation.report import save_metrics_csv, save_metrics_json


STRESS_COLUMNS = [
    "subset",
    "num_images",
    "AttrAcc_mean",
    "Leakage_mean",
    "InstIoU_mean_mean",
    "SwapRateProxy_mean",
    "OrderAccProxy_mean",
    "OcclusionRecallProxy_mean",
    "EditLocality_mean",
]


def _to_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def summarize_stress_rows(rows: list[dict]) -> dict:
    """按 subset 汇总 stress 指标，用于快速发现困难场景 failure mode。"""

    metrics = [
        "AttrAcc",
        "Leakage",
        "InstIoU_mean",
        "SwapRateProxy",
        "OrderAccProxy",
        "OcclusionRecallProxy",
        "EditLocality",
    ]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        subset = str(row.get("Subset") or row.get("subset") or "unknown")
        groups.setdefault(subset, []).append(row)
    summary = {}
    for subset, items in sorted(groups.items()):
        out = {"subset": subset, "num_images": len(items)}
        for metric in metrics:
            vals = [_to_float(item.get(metric)) for item in items]
            vals = [v for v in vals if v is not None]
            out[f"{metric}_mean"] = sum(vals) / len(vals) if vals else None
        summary[subset] = out
    return summary


def save_stress_summary(rows: list[dict], out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_stress_rows(rows)
    summary_rows = list(summary.values())
    save_metrics_json(rows, out / "stress_per_image.json")
    save_metrics_csv(rows, out / "stress_per_image.csv")
    save_metrics_json(summary, out / "stress_subset_summary.json")
    save_metrics_csv([{col: row.get(col, "") for col in STRESS_COLUMNS} for row in summary_rows], out / "stress_subset_summary.csv")
    return summary


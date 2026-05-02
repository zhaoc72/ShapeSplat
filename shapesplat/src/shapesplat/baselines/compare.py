from __future__ import annotations

import json
from pathlib import Path

import torch

from shapesplat.baselines.evaluate_baseline import evaluate_baseline_prediction
from shapesplat.evaluation.report import flatten_metrics, save_metrics_csv, save_metrics_json


COMPARISON_KEYS = [
    "method",
    "InstIoU_mean",
    "IsoIoU_mean",
    "AttrAcc",
    "AttrPurity_mean",
    "Leakage",
    "ForegroundAlphaError",
    "ForegroundRGBL1",
]


def save_comparison_table(rows: list[dict], path: str | Path) -> None:
    """保存轻量 comparison CSV，不依赖 pandas。"""

    path = Path(path)
    csv_rows = []
    for row in rows:
        flat = flatten_metrics(row)
        csv_rows.append({key: flat.get(key, "") for key in COMPARISON_KEYS})
    save_metrics_csv(csv_rows, path)


def compare_methods_for_image(
    image: torch.Tensor,
    masks: torch.Tensor,
    methods: dict[str, dict],
    out_dir: str | Path,
) -> list[dict]:
    """对同一张图的多个 baseline prediction 做统一评估并汇总。"""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, prediction in methods.items():
        metrics = evaluate_baseline_prediction(prediction, masks, image=image)
        row = {"method": method, **metrics}
        rows.append(row)
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        save_metrics_json(row, method_dir / "metrics.json")
    save_metrics_json(rows, out_dir / "comparison.json")
    save_comparison_table(rows, out_dir / "comparison.csv")
    return rows


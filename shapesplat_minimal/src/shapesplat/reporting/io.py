from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict | list:
    """读取 JSON 结果文件。"""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: str | Path) -> list[dict]:
    """读取 CSV rows，不依赖 pandas。"""

    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_json(obj: Any, path: str | Path) -> None:
    """保存 JSON，并自动创建父目录。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv_rows(rows: list[dict], path: str | Path) -> None:
    """保存 CSV rows，自动收集所有列。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def find_experiment_outputs(root: str | Path) -> dict[str, Path]:
    """在实验目录下查找常见输出文件。

    comparison runner 产出 per_method_summary/per_image_comparison；
    ablation runner 产出 ablation_summary；baseline runner 产出 baseline_summary；
    dataset runner 产出 summary。缺失文件不会报错，方便对不同 runner 复用。
    """

    root = Path(root)
    candidates = {
        "comparison_summary": root / "per_method_summary.json",
        "per_image_comparison": root / "per_image_comparison.json",
        "ablation_summary": root / "ablation_summary.json",
        "baseline_summary": root / "baseline_summary.json",
        "dataset_summary": root / "summary.json",
        "stress_subset_summary": root / "stress_subset_summary.json",
        "stress_per_image": root / "stress_per_image.json",
        "comparison_csv": root / "comparison.csv",
        "metrics": root / "metrics.json",
    }
    found = {key: path for key, path in candidates.items() if path.exists()}
    # 递归查找单图 metrics，便于对不规则输出目录做基本诊断。
    metrics_files = sorted(root.glob("**/metrics.json"))
    if metrics_files:
        found["metrics_files"] = metrics_files
    return found

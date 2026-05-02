from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from shapesplat.reproducibility.run_info import RunInfo


def append_run_registry(
    run_info: RunInfo,
    registry_path: str | Path = "runs/run_registry.jsonl",
    metrics_summary: dict | None = None,
) -> None:
    """追加全局 run registry。

    这是轻量实验追踪，不替代 MLflow/W&B。
    """

    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    item = asdict(run_info)
    item["metrics_summary"] = metrics_summary or {}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_run_registry(registry_path: str | Path = "runs/run_registry.jsonl") -> list[dict]:
    path = Path(registry_path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_runs(
    registry_path: str | Path = "runs/run_registry.jsonl",
    run_type: str | None = None,
    status: str | None = None,
) -> list[dict]:
    rows = load_run_registry(registry_path)
    if run_type is not None:
        rows = [r for r in rows if r.get("run_type") == run_type]
    if status is not None:
        rows = [r for r in rows if r.get("status") == status]
    return rows


def print_run_registry(rows: list[dict], max_rows: int = 20) -> None:
    cols = ["run_id", "run_type", "status", "timestamp", "output_dir", "AttrAcc_mean", "Leakage_mean"]
    view = rows[-max_rows:]
    widths = {c: len(c) for c in cols}
    flat = []
    for row in view:
        metrics = row.get("metrics_summary", {}) or {}
        item = {**row, **metrics}
        flat.append(item)
        for c in cols:
            widths[c] = max(widths[c], len(str(item.get(c, ""))))
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for item in flat:
        print(" | ".join(str(item.get(c, "")).ljust(widths[c]) for c in cols))


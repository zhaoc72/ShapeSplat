from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KEYS = [
    "num_success",
    "num_failed",
    "InstIoU_mean",
    "AttrAcc",
    "AttrPurity_mean",
    "Leakage",
    "ForegroundAlphaError",
    "CollateralL1",
    "EditLocality",
    "DeletionResidual",
]


def _load_summary(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[0] if data else {}
        return data
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[0] if rows else {}
    raise ValueError(f"Unsupported summary file: {path}")


def print_metrics_table(path: str | Path) -> None:
    """打印轻量 batch metrics 表格，不依赖 pandas。"""
    summary = _load_summary(Path(path))
    rows = [(key, summary[key]) for key in KEYS if key in summary]
    width = max([len("metric")] + [len(k) for k, _ in rows])
    print(f"{'metric'.ljust(width)} | value")
    print(f"{'-' * width}-+-{'-' * 12}")
    for key, value in rows:
        print(f"{key.ljust(width)} | {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact metrics table from summary.json/csv.")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    print_metrics_table(args.summary)


if __name__ == "__main__":
    main()

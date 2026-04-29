from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COLUMNS = ["name", "AttrAcc", "AttrPurity_mean", "Leakage", "InstIoU_mean", "CollateralL1", "EditLocality"]


def load_summary(path: str | Path) -> list[dict]:
    """读取 ablation_summary.json 或 ablation_summary.csv。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"summary 不存在: {p.resolve()}")
    if p.suffix.lower() == ".csv":
        with open(p, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def print_table(rows: list[dict]) -> None:
    """轻量结果查看工具，不依赖 pandas。"""
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in COLUMNS}
    header = " | ".join(c.ljust(widths[c]) for c in COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(row.get(c, "")).ljust(widths[c]) for c in COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact ShapeSplat++ ablation table.")
    parser.add_argument("--summary", default="outputs/ablations/ablation_summary.json", help="summary json/csv 路径")
    args = parser.parse_args()
    print_table(load_summary(args.summary))


if __name__ == "__main__":
    main()

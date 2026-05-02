from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from shapesplat.benchmarks.standard.validator import validate_benchmark_manifest


def summarize_benchmark(manifest_path: str | Path) -> dict:
    report = validate_benchmark_manifest(manifest_path)
    rows = report.get("rows", [])
    split_counts = Counter()
    subset_counts = Counter()
    num_objects = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            split_counts[row.get("split") or ""] += 1
            subset_counts[row.get("subset") or ""] += 1
    for row in rows:
        if row.get("num_masks") is not None:
            num_objects.append(int(row["num_masks"]))
    return {
        "num_images": len(rows),
        "split_counts": dict(split_counts),
        "subset_counts": dict(subset_counts),
        "avg_num_objects": sum(num_objects) / max(1, len(num_objects)),
        "min_num_objects": min(num_objects) if num_objects else None,
        "max_num_objects": max(num_objects) if num_objects else None,
        "validation_valid": report.get("valid", False),
    }

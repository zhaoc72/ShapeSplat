from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from shapesplat.datasets.benchmark.manifest_v2 import load_benchmark_manifest
from shapesplat.evaluation.report import save_metrics_csv


def summarize_benchmark_v2(manifest_path: str | Path) -> dict:
    """生成 paper-ready benchmark 摘要，统计 split/subset/cache/optional GT 覆盖率。"""

    records = load_benchmark_manifest(manifest_path)
    nums = [r.num_objects for r in records if r.num_objects is not None]
    diagnostics = Counter()
    for r in records:
        for k, v in r.diagnostics.items():
            if str(v).lower() in {"true", "1", "yes"}:
                diagnostics[k] += 1
    return {
        "num_images": len(records),
        "split_counts": dict(Counter(r.split or "unknown" for r in records)),
        "subset_counts": dict(Counter(r.subset or "unknown" for r in records)),
        "source_dataset_counts": dict(Counter(r.source_dataset or "unknown" for r in records)),
        "average_num_objects": (sum(nums) / len(nums)) if nums else None,
        "min_num_objects": min(nums) if nums else None,
        "max_num_objects": max(nums) if nums else None,
        "num_with_depth": sum(1 for r in records if r.depth_path),
        "num_with_camera": sum(1 for r in records if r.camera_path),
        "num_with_gt_pointcloud": sum(1 for r in records if r.gt_pointcloud_path),
        "num_with_gt_mesh": sum(1 for r in records if r.gt_mesh_path),
        "num_with_frontend_cache": sum(1 for r in records if r.frontend_cache_dir),
        "diagnostic_counts": dict(diagnostics),
    }


def save_benchmark_summary(summary: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [{"metric": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v} for k, v in summary.items()]
    save_metrics_csv(rows, out / "benchmark_summary.csv")


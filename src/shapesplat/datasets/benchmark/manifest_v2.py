from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from shapesplat.datasets.benchmark.schema import DIAGNOSTIC_COLUMNS, get_all_known_columns


@dataclass
class BenchmarkRecord:
    """正式 benchmark manifest 的核心记录。

    optional GT/cache/diagnostic 字段都允许为空；这样 toy/example/stress 和真实数据
    可以共用同一个协议，而不会因为 GT 缺失影响 same-mask 实验。
    """

    image_id: str
    image_path: str
    mask_path: str
    split: str
    metadata_path: Optional[str] = None
    subset: Optional[str] = None
    category: Optional[str] = None
    num_objects: Optional[int] = None
    scene_id: Optional[str] = None
    source_dataset: Optional[str] = None
    depth_path: Optional[str] = None
    camera_path: Optional[str] = None
    gt_pointcloud_path: Optional[str] = None
    gt_mesh_path: Optional[str] = None
    visible_pointcloud_path: Optional[str] = None
    hidden_pointcloud_path: Optional[str] = None
    frontend_cache_dir: Optional[str] = None
    frontend_cache_status: Optional[str] = None
    diagnostics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def _clean(value):
    return None if value in (None, "", "null", "None") else value


def _resolve_path(value, manifest_dir: Path) -> Optional[str]:
    value = _clean(value)
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest_dir / path
    return str(path)


def row_to_record(row: dict, manifest_dir: str | Path) -> BenchmarkRecord:
    """CSV row -> BenchmarkRecord；相对路径按 manifest 所在目录解析。"""

    mdir = Path(manifest_dir)
    known = set(get_all_known_columns())
    diagnostics = {k: row.get(k) for k in DIAGNOSTIC_COLUMNS if _clean(row.get(k)) is not None}
    extra = {k: v for k, v in row.items() if k not in known and _clean(v) is not None}
    num_objects = _clean(row.get("num_objects"))
    return BenchmarkRecord(
        image_id=str(row.get("image_id", "")),
        image_path=_resolve_path(row.get("image_path"), mdir) or "",
        mask_path=_resolve_path(row.get("mask_path"), mdir) or "",
        split=str(row.get("split", "")),
        metadata_path=_resolve_path(row.get("metadata_path"), mdir),
        subset=_clean(row.get("subset")),
        category=_clean(row.get("category")),
        num_objects=int(num_objects) if num_objects is not None else None,
        scene_id=_clean(row.get("scene_id")),
        source_dataset=_clean(row.get("source_dataset")),
        depth_path=_resolve_path(row.get("depth_path"), mdir),
        camera_path=_resolve_path(row.get("camera_path"), mdir),
        gt_pointcloud_path=_resolve_path(row.get("gt_pointcloud_path"), mdir),
        gt_mesh_path=_resolve_path(row.get("gt_mesh_path"), mdir),
        visible_pointcloud_path=_resolve_path(row.get("visible_pointcloud_path"), mdir),
        hidden_pointcloud_path=_resolve_path(row.get("hidden_pointcloud_path"), mdir),
        frontend_cache_dir=_resolve_path(row.get("frontend_cache_dir"), mdir),
        frontend_cache_status=_clean(row.get("frontend_cache_status")),
        diagnostics=diagnostics,
        extra=extra,
    )


def record_to_row(record: BenchmarkRecord) -> dict:
    """BenchmarkRecord -> CSV row；调用方可决定写相对路径或绝对路径。"""

    row = {
        "image_id": record.image_id,
        "image_path": record.image_path,
        "mask_path": record.mask_path,
        "split": record.split,
        "metadata_path": record.metadata_path or "",
        "subset": record.subset or "",
        "category": record.category or "",
        "num_objects": "" if record.num_objects is None else str(record.num_objects),
        "scene_id": record.scene_id or "",
        "source_dataset": record.source_dataset or "",
        "depth_path": record.depth_path or "",
        "camera_path": record.camera_path or "",
        "gt_pointcloud_path": record.gt_pointcloud_path or "",
        "gt_mesh_path": record.gt_mesh_path or "",
        "visible_pointcloud_path": record.visible_pointcloud_path or "",
        "hidden_pointcloud_path": record.hidden_pointcloud_path or "",
        "frontend_cache_dir": record.frontend_cache_dir or "",
        "frontend_cache_status": record.frontend_cache_status or "",
    }
    row.update(record.diagnostics)
    row.update(record.extra)
    return row


def load_benchmark_manifest(path: str | Path) -> list[BenchmarkRecord]:
    path = Path(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [row_to_record(row, path.parent) for row in csv.DictReader(f)]


def save_benchmark_manifest(records: list[BenchmarkRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record_to_row(r) for r in records]
    fields = get_all_known_columns()
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from shapesplat.datasets.benchmark.manifest_v2 import BenchmarkRecord, save_benchmark_manifest
from shapesplat.datasets.benchmark.splits import create_split_file


def _copy_or_ref(src: Path, dst: Path, copy_files: bool, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        if dst.exists() and not overwrite:
            return str(dst)
        shutil.copy2(src, dst)
        return str(dst)
    return str(src)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path)


def write_benchmark_info(out_dir: str | Path, info: dict) -> None:
    """写 benchmark_info.json；记录协议版本、来源和构建时间。"""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "v2", "created_at": datetime.now().isoformat(), **info}
    (out / "benchmark_info.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_benchmark_from_existing_manifest(source_manifest: str | Path, out_dir: str | Path, copy_files: bool = True, overwrite: bool = False, source_dataset: str = "custom") -> Path:
    """从已有 same-mask manifest 构建 benchmark v2。

    builder 只整理数据协议，不生成新 mask，也不负责真实 GT；masks 应是 retained visible masks。
    """

    source_manifest = Path(source_manifest)
    out = Path(out_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    for name in ["images", "masks", "metadata", "depth", "cameras", "gt_pointclouds", "gt_meshes", "frontend_cache"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    with open(source_manifest, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    records: list[BenchmarkRecord] = []
    for row in rows:
        image_src = Path(row["image_path"])
        mask_src = Path(row["mask_path"])
        if not image_src.is_absolute():
            image_src = source_manifest.parent / image_src
        if not mask_src.is_absolute():
            mask_src = source_manifest.parent / mask_src
        image_dst = out / "images" / image_src.name
        mask_dst = out / "masks" / mask_src.name
        _copy_or_ref(image_src, image_dst, copy_files, overwrite)
        _copy_or_ref(mask_src, mask_dst, copy_files, overwrite)
        meta_path = None
        if row.get("metadata_path"):
            meta_src = Path(row["metadata_path"])
            if not meta_src.is_absolute():
                meta_src = source_manifest.parent / meta_src
            if meta_src.exists():
                meta_dst = out / "metadata" / meta_src.name
                _copy_or_ref(meta_src, meta_dst, copy_files, overwrite)
                meta_path = _rel(meta_dst, out)
        records.append(
            BenchmarkRecord(
                image_id=row.get("image_id") or image_src.stem,
                image_path=_rel(image_dst if copy_files else image_src, out),
                mask_path=_rel(mask_dst if copy_files else mask_src, out),
                split=row.get("split") or "test",
                metadata_path=meta_path,
                subset=row.get("subset") or "default",
                category=row.get("category") or "",
                num_objects=int(row["num_objects"]) if row.get("num_objects") else None,
                scene_id=row.get("scene_id") or row.get("image_id") or image_src.stem,
                source_dataset=source_dataset,
                frontend_cache_dir=row.get("frontend_cache_dir") or None,
            )
        )
    manifest = out / "manifest.csv"
    save_benchmark_manifest(records, manifest)
    create_split_file(records, out / "splits.json")
    write_benchmark_info(out, {"name": out.name, "source_dataset": source_dataset, "num_images": len(records), "notes": "built from existing manifest"})
    return manifest


def build_benchmark_from_folders(image_dir: str | Path, mask_dir: str | Path, out_dir: str | Path, metadata_dir: str | Path | None = None, source_dataset: str = "custom_folder", split: str = "test", overwrite: bool = False) -> Path:
    """从 images/masks[/metadata] 文件夹构建 benchmark v2，按同名文件匹配。"""

    image_dir = Path(image_dir).resolve()
    mask_dir = Path(mask_dir).resolve()
    out = Path(out_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    tmp = out / "_folder_source_manifest.csv"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}):
        mask = next((mask_dir / f"{image.stem}{ext}" for ext in [".npy", ".npz", ".png"] if (mask_dir / f"{image.stem}{ext}").exists()), None)
        if mask is None:
            continue
        meta = None
        if metadata_dir:
            metadata_dir = Path(metadata_dir).resolve()
            meta = next((metadata_dir / f"{image.stem}{ext}" for ext in [".json"] if (metadata_dir / f"{image.stem}{ext}").exists()), None)
        rows.append({"image_id": image.stem, "image_path": str(image), "mask_path": str(mask), "metadata_path": str(meta or ""), "split": split, "subset": "default"})
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        fields = ["image_id", "image_path", "mask_path", "metadata_path", "split", "subset"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return build_benchmark_from_existing_manifest(tmp, out, copy_files=True, overwrite=False, source_dataset=source_dataset)

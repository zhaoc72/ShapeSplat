from __future__ import annotations

import csv
import shutil
from pathlib import Path

from shapesplat.benchmarks.standard.manifest_schema import normalize_manifest_row


def _copy_or_reference(src: Path, dst: Path, copy_files: bool, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        if dst.exists() and not overwrite:
            return str(dst)
        shutil.copy2(src, dst)
        return str(dst)
    return str(src)


def build_same_mask_benchmark(source_manifest: str | Path, out_dir: str | Path, copy_files: bool = True, overwrite: bool = False) -> Path:
    """把已有 manifest 整理为标准 same-mask benchmark 目录。

    builder 不生成新 mask，只复制/引用已有 retained visible masks。
    """
    src_manifest = Path(source_manifest)
    out = Path(out_dir)
    image_dir, mask_dir, meta_dir = out / "images", out / "masks", out / "metadata"
    out.mkdir(parents=True, exist_ok=True)
    with open(src_manifest, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out_rows = []
    for raw in rows:
        row = normalize_manifest_row(raw, src_manifest.parent)
        image_src, mask_src = Path(row["image_path"]), Path(row["mask_path"])
        image_dst = image_dir / image_src.name
        mask_dst = mask_dir / mask_src.name
        std = dict(row)
        std["image_path"] = str(Path("images") / image_src.name)
        std["mask_path"] = str(Path("masks") / mask_src.name)
        _copy_or_reference(image_src, image_dst, copy_files, overwrite)
        _copy_or_reference(mask_src, mask_dst, copy_files, overwrite)
        if row.get("metadata_path"):
            meta_src = Path(row["metadata_path"])
            if meta_src.exists():
                meta_dst = meta_dir / meta_src.name
                _copy_or_reference(meta_src, meta_dst, copy_files, overwrite)
                std["metadata_path"] = str(Path("metadata") / meta_src.name)
        out_rows.append(std)
    manifest = out / "manifest.csv"
    keys = ["image_id", "image_path", "mask_path", "metadata_path", "split", "subset", "category", "num_objects"]
    for r in out_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(out_rows)
    return manifest


def build_from_folder(image_dir: str | Path, mask_dir: str | Path, out_dir: str | Path, image_exts: list[str] | None = None) -> Path:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    out = Path(out_dir)
    image_exts = image_exts or [".png", ".jpg", ".jpeg"]
    rows = []
    for img in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in image_exts):
        mask = next((mask_dir / f"{img.stem}{ext}" for ext in [".npy", ".npz", ".png"] if (mask_dir / f"{img.stem}{ext}").exists()), None)
        if mask is None:
            continue
        rows.append({"image_id": img.stem, "image_path": str(img), "mask_path": str(mask), "split": "test"})
    tmp_manifest = out / "_source_manifest.csv"
    out.mkdir(parents=True, exist_ok=True)
    with open(tmp_manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "mask_path", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return build_same_mask_benchmark(tmp_manifest, out, copy_files=True, overwrite=True)

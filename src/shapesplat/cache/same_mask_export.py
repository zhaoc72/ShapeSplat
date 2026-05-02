from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from shapesplat.benchmarks.standard.validator import validate_benchmark_manifest
from shapesplat.cache.frontend_cache import load_frontend_cache_manifest
from shapesplat.datasets.manifest import load_manifest


def cache_to_same_mask_dataset(image_manifest: str | Path, cache_manifest: str | Path, out_dir: str | Path, copy_images: bool = True, overwrite: bool = False) -> Path:
    """把 frontend cache 中的 retained visible masks 固化成 same-mask dataset。"""
    out = Path(out_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    cache = load_frontend_cache_manifest(cache_manifest)
    rows = []
    for record in load_manifest(image_manifest):
        if record.image_id not in cache:
            continue
        c = cache[record.image_id]
        image_src = Path(record.image_path)
        image_dst = out / "images" / image_src.name
        if copy_images:
            shutil.copy2(image_src, image_dst)
            image_path = f"images/{image_dst.name}"
        else:
            image_path = str(image_src)
        masks = np.load(c.masks_path)
        mask_path = out / "masks" / f"{record.image_id}.npy"
        np.save(mask_path, masks.astype("float32"))
        meta_path = out / "metadata" / f"{record.image_id}.json"
        meta = {"image_id": record.image_id, "frontend_cache_dir": c.cache_dir, "cache_meta_path": c.meta_path}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append(
            {
                "image_id": record.image_id,
                "image_path": image_path,
                "mask_path": f"masks/{mask_path.name}",
                "metadata_path": f"metadata/{meta_path.name}",
                "frontend_cache_dir": c.cache_dir,
                "split": record.split,
                "subset": record.metadata.get("subset", ""),
                "num_objects": str(int(masks.shape[0])),
            }
        )
    manifest_out = out / "manifest.csv"
    fields = ["image_id", "image_path", "mask_path", "metadata_path", "frontend_cache_dir", "split", "subset", "num_objects"]
    with open(manifest_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    validate_benchmark_manifest(manifest_out)
    return manifest_out

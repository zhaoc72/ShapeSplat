from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from shapesplat.cache.frontend_cache import load_frontend_cache_manifest
from shapesplat.datasets.benchmark.builder_v2 import write_benchmark_info
from shapesplat.datasets.benchmark.manifest_v2 import BenchmarkRecord, load_benchmark_manifest, save_benchmark_manifest
from shapesplat.datasets.benchmark.splits import create_split_file


def bind_frontend_cache_to_benchmark(benchmark_manifest: str | Path, cache_manifest: str | Path, out_manifest: str | Path) -> Path:
    """按 image_id 将 frontend cache manifest 绑定到 benchmark v2 manifest。"""

    cache = load_frontend_cache_manifest(cache_manifest)
    records = load_benchmark_manifest(benchmark_manifest)
    for record in records:
        if record.image_id in cache:
            record.frontend_cache_dir = cache[record.image_id].cache_dir
            record.frontend_cache_status = cache[record.image_id].status
        else:
            record.frontend_cache_status = "missing"
    save_benchmark_manifest(records, out_manifest)
    return Path(out_manifest)


def export_cached_masks_to_benchmark(image_manifest: str | Path, cache_manifest: str | Path, out_dir: str | Path, copy_images: bool = True, overwrite: bool = False) -> Path:
    """把真实前端缓存中的 retained masks 固化为 same-mask benchmark v2。"""

    out = Path(out_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    for name in ["images", "masks", "metadata"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    cache = load_frontend_cache_manifest(cache_manifest)
    records = []
    from shapesplat.datasets.benchmark.manifest_v2 import load_benchmark_manifest as _load_v2
    try:
        src_records = _load_v2(image_manifest)
    except Exception:
        from shapesplat.datasets.manifest import load_manifest
        src_records = load_manifest(image_manifest)
    for src in src_records:
        image_id = src.image_id
        if image_id not in cache:
            continue
        c = cache[image_id]
        image_src = Path(src.image_path)
        image_dst = out / "images" / image_src.name
        if copy_images:
            shutil.copy2(image_src, image_dst)
            image_path = f"images/{image_dst.name}"
        else:
            image_path = str(image_src)
        masks = np.load(c.masks_path)
        mask_path = out / "masks" / f"{image_id}.npy"
        np.save(mask_path, masks.astype("float32"))
        meta_path = out / "metadata" / f"{image_id}.json"
        meta_path.write_text(json.dumps({"image_id": image_id, "frontend_cache_dir": c.cache_dir}, indent=2), encoding="utf-8")
        records.append(
            BenchmarkRecord(
                image_id=image_id,
                image_path=image_path,
                mask_path=f"masks/{mask_path.name}",
                split=getattr(src, "split", "test") or "test",
                metadata_path=f"metadata/{meta_path.name}",
                subset=getattr(src, "subset", None) or getattr(src, "metadata", {}).get("subset", "cached"),
                num_objects=int(masks.shape[0]),
                scene_id=getattr(src, "scene_id", None) or image_id,
                source_dataset=getattr(src, "source_dataset", None) or "cached_frontend",
                frontend_cache_dir=c.cache_dir,
                frontend_cache_status=c.status,
            )
        )
    manifest = out / "manifest.csv"
    save_benchmark_manifest(records, manifest)
    create_split_file(records, out / "splits.json")
    write_benchmark_info(out, {"name": out.name, "source_dataset": "cached_frontend", "num_images": len(records), "notes": "masks exported from frontend cache"})
    return manifest


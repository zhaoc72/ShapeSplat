from __future__ import annotations

import json
from pathlib import Path

from shapesplat.cache.frontend_cache import FrontendCacheManifestRecord, frontend_cache_exists, save_frontend_output
from shapesplat.config import load_config
from shapesplat.datasets.image_dataset import build_dataset_from_manifest
from shapesplat.evaluation.report import save_metrics_csv
from shapesplat.frontend.pipeline import build_frontend


def cache_frontend_outputs(config: str, manifest: str, out_cache: str, max_images=None, save_dino_features=False, skip_existing=False, mask_source=None, use_existing=False) -> tuple[list[dict], list[FrontendCacheManifestRecord]]:
    """批量缓存 frontend outputs；供 CLI 和 CO3Dv2 real frontend cache 共用。"""
    cfg = load_config(config)
    if mask_source:
        cfg.setdefault("frontend", {})["mask_source"] = mask_source
    dataset = build_dataset_from_manifest(manifest, cfg=cfg)
    rows: list[dict] = []
    records: list[FrontendCacheManifestRecord] = []
    out_root = Path(out_cache)
    out_root.mkdir(parents=True, exist_ok=True)
    total = min(len(dataset), int(max_images)) if max_images is not None else len(dataset)
    for idx in range(total):
        item = dataset[idx]
        image_id = item["image_id"]
        cache_dir = out_root / image_id
        try:
            if (skip_existing or use_existing) and frontend_cache_exists(cache_dir):
                rows.append({"image_id": image_id, "status": "skipped", "cache_dir": str(cache_dir)})
                records.append(
                    FrontendCacheManifestRecord(
                        image_id=image_id,
                        image_path=item["image_path"],
                        cache_dir=str(cache_dir),
                        masks_path=str(cache_dir / "masks.npy"),
                        descriptors_path=str(cache_dir / "descriptors.npy"),
                        depth_path=str(cache_dir / "depth.npy"),
                        meta_path=str(cache_dir / "frontend_meta.json"),
                        status="valid",
                    )
                )
                continue
            front = build_frontend(item["image"], cfg, record=item["record"])
            front.metadata["config_path"] = str(config)
            if cfg.get("image", {}).get("resize_mode") == "keep_aspect":
                front.metadata["cache_resolution_tag"] = f"highres_long{cfg.get('image', {}).get('long_side')}_dino{cfg.get('frontend', {}).get('dino_input_size')}"
            save_frontend_output(front, cache_dir, image_id=image_id, save_dino_features=save_dino_features)
            records.append(
                FrontendCacheManifestRecord(
                    image_id=image_id,
                    image_path=item["image_path"],
                    cache_dir=str(cache_dir),
                    masks_path=str(cache_dir / "masks.npy"),
                    descriptors_path=str(cache_dir / "descriptors.npy"),
                    depth_path=str(cache_dir / "depth.npy"),
                    meta_path=str(cache_dir / "frontend_meta.json"),
                    status="valid",
                    num_masks=int(front.masks.shape[0]),
                    descriptor_dim=int(front.descriptors.shape[1]),
                )
            )
            rows.append({"image_id": image_id, "status": "success", "num_masks": int(front.masks.shape[0]), "descriptor_dim": int(front.descriptors.shape[1]), "cache_dir": str(cache_dir)})
            print(f"[{idx + 1}/{total}] cached {image_id}")
        except Exception as exc:
            err = {"image_id": image_id, "status": "failed", "error": str(exc), "cache_dir": str(cache_dir)}
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
            rows.append(err)
            print(f"[{idx + 1}/{total}] failed {image_id}: {exc}")
    summary = {"num_images": len(rows), "num_success": sum(r["status"] == "success" for r in rows), "rows": rows}
    (out_root / "frontend_cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_metrics_csv(rows, out_root / "frontend_cache_summary.csv")
    return rows, records

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.cache.frontend_cache import FrontendCacheManifestRecord, frontend_cache_exists, save_frontend_output, write_frontend_cache_manifest
from shapesplat.cache.validate_cache import validate_frontend_cache_manifest
from shapesplat.config import load_config
from shapesplat.datasets.image_dataset import build_dataset_from_manifest
from shapesplat.evaluation.report import save_metrics_csv
from shapesplat.frontend.pipeline import build_frontend


def cache_frontend_outputs(config: str, manifest: str, out_cache: str, max_images=None, save_dino_features=False, skip_existing=False, mask_source=None, use_existing=False) -> tuple[list[dict], list[FrontendCacheManifestRecord]]:
    """批量缓存前端输出，避免真实 SAM / DINO / Depth 重复推理。"""
    cfg = load_config(config)
    if mask_source:
        cfg.setdefault("frontend", {})["mask_source"] = mask_source
    dataset = build_dataset_from_manifest(manifest, cfg["image"]["size"])
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
            rows.append(
                {
                    "image_id": image_id,
                    "status": "success",
                    "num_masks": int(front.masks.shape[0]),
                    "descriptor_dim": int(front.descriptors.shape[1]),
                    "cache_dir": str(cache_dir),
                }
            )
            print(f"[{idx+1}/{total}] cached {image_id}")
        except Exception as exc:
            err = {"image_id": image_id, "status": "failed", "error": str(exc), "cache_dir": str(cache_dir)}
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
            rows.append(err)
            print(f"[{idx+1}/{total}] failed {image_id}: {exc}")
    summary = {"num_images": len(rows), "num_success": sum(r["status"] == "success" for r in rows), "rows": rows}
    (out_root / "frontend_cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_metrics_csv(rows, out_root / "frontend_cache_summary.csv")
    return rows, records


def update_dataset_manifest_with_cache(src_manifest: str | Path, records: list[FrontendCacheManifestRecord], out_path: str | Path) -> None:
    """给原 manifest 增加 frontend_cache_dir 列，供后续 runner 直接读取。"""

    # 写入 dataset manifest 时使用绝对 cache 路径，避免 manifest 被移动后相对路径二次解析。
    cache_by_id = {r.image_id: str(Path(r.cache_dir).resolve()) for r in records}
    with open(src_manifest, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "frontend_cache_dir" not in fields:
        fields.append("frontend_cache_dir")
    for row in rows:
        if row.get("image_id") in cache_by_id:
            row["frontend_cache_dir"] = cache_by_id[row["image_id"]]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frontend outputs for a manifest dataset.")
    parser.add_argument("--config", default="configs/local_real_frontend.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-cache", default="outputs/frontend_cache")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-dino-features", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--mask-source", default=None)
    parser.add_argument("--backend-summary", action="store_true")
    parser.add_argument("--write-manifest", default=None)
    parser.add_argument("--update-dataset-manifest", default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--use-existing", action="store_true")
    args = parser.parse_args()
    rows, records = cache_frontend_outputs(
        args.config,
        args.manifest,
        args.out_cache,
        args.max_images,
        args.save_dino_features,
        args.skip_existing,
        args.mask_source,
        args.use_existing,
    )
    manifest_path = args.write_manifest or str(Path(args.out_cache) / "cache_manifest.csv")
    write_frontend_cache_manifest(records, manifest_path)
    if args.update_dataset_manifest:
        update_dataset_manifest_with_cache(args.manifest, records, args.update_dataset_manifest)
    if args.validate:
        report = validate_frontend_cache_manifest(manifest_path)
        (Path(args.out_cache) / "cache_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

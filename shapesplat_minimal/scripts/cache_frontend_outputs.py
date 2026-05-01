from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.cache.frontend_cache import frontend_cache_exists, save_frontend_output
from shapesplat.config import load_config
from shapesplat.datasets.image_dataset import build_dataset_from_manifest
from shapesplat.evaluation.report import save_metrics_csv
from shapesplat.frontend.pipeline import build_frontend


def cache_frontend_outputs(config: str, manifest: str, out_cache: str, max_images=None, save_dino_features=False, skip_existing=False, mask_source=None) -> list[dict]:
    """批量缓存前端输出，避免真实 SAM / DINO / Depth 重复推理。"""
    cfg = load_config(config)
    if mask_source:
        cfg.setdefault("frontend", {})["mask_source"] = mask_source
    dataset = build_dataset_from_manifest(manifest, cfg["image"]["size"])
    rows: list[dict] = []
    out_root = Path(out_cache)
    out_root.mkdir(parents=True, exist_ok=True)
    total = min(len(dataset), int(max_images)) if max_images is not None else len(dataset)
    for idx in range(total):
        item = dataset[idx]
        image_id = item["image_id"]
        cache_dir = out_root / image_id
        try:
            if skip_existing and frontend_cache_exists(cache_dir):
                rows.append({"image_id": image_id, "status": "skipped", "cache_dir": str(cache_dir)})
                continue
            front = build_frontend(item["image"], cfg, record=item["record"])
            save_frontend_output(front, cache_dir, image_id=image_id, save_dino_features=save_dino_features)
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
    return rows


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
    args = parser.parse_args()
    cache_frontend_outputs(args.config, args.manifest, args.out_cache, args.max_images, args.save_dino_features, args.skip_existing, args.mask_source)


if __name__ == "__main__":
    main()

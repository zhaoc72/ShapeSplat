from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.datasets.image_dataset import build_dataset_from_manifest
from shapesplat.experiments.co3dv2_real_frontend import best_iou_and_coverage, check_checkpoint_path
from shapesplat.frontend.sam3_real import RealSAM3Wrapper
from shapesplat.utils.visualization import save_mask_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare optional SAM3 masks against CO3Dv2 file masks.")
    parser.add_argument("--config", default="configs/co3dv2_real_frontend.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/check_sam3_vs_co3d_masks")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    runtime_overrides = {"device": args.device, "require_cuda_for_experiments": args.device == "cuda"} if args.device else None
    if args.device == "cpu":
        runtime_overrides["allow_cpu_fallback"] = True
        runtime_overrides["require_cuda_for_experiments"] = False
    cfg = load_config(args.config, runtime_overrides=runtime_overrides)
    fcfg = cfg.setdefault("frontend", {})
    if args.sam_checkpoint:
        fcfg["sam3_checkpoint"] = args.sam_checkpoint
    if args.device:
        fcfg["sam3_device"] = args.device
    fcfg["mask_source"] = "sam"
    fcfg["sam_backend"] = "real"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = check_checkpoint_path(fcfg.get("sam3_checkpoint"), allow_missing=args.allow_missing)
    if not ckpt["exists"]:
        report = {"status": "missing_checkpoint", "checkpoint": ckpt}
        (out / "sam3_mask_diagnostic_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.allow_missing:
            print(report)
            return
        raise FileNotFoundError(ckpt["error"])

    dataset = build_dataset_from_manifest(args.manifest, image_size=int(cfg["image"]["size"]))
    sam = RealSAM3Wrapper(cfg)
    rows = []
    for idx in range(min(len(dataset), args.max_images)):
        item = dataset[idx]
        image = load_image(item["image_path"], size=int(cfg["image"]["size"]))
        co3d_masks = np.load(item["record"].metadata["mask_path"])
        try:
            sam_masks = sam.predict_masks(image).masks.detach().cpu().numpy()
            row = {"image_id": item["image_id"], "status": "success", **best_iou_and_coverage(co3d_masks, sam_masks)}
            save_mask_grid(sam.predict_masks(image).masks, out / f"{item['image_id']}_sam_masks.png")
        except Exception as exc:
            row = {"image_id": item["image_id"], "status": "failed", "error": str(exc)}
        rows.append(row)
    with open(out / "per_image_sam3_mask_diagnostic.csv", "w", encoding="utf-8", newline="") as f:
        fields = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "per_image_sam3_mask_diagnostic.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"num_images": len(rows), "num_success": sum(r.get("status") == "success" for r in rows)}, indent=2))


if __name__ == "__main__":
    main()

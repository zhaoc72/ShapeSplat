from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.baselines.compare import compare_methods_for_image
from shapesplat.baselines.dummy_baselines import DUMMY_BASELINES, run_dummy_baseline, save_baseline_prediction
from shapesplat.baselines.evaluate_baseline import evaluate_baseline_prediction
from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.frontend.file_mask_loader import load_mask_file


def _print_rows(rows: list[dict]) -> None:
    keys = ["method", "AttrAcc", "AttrPurity_mean", "Leakage", "InstIoU_mean", "ForegroundRGBL1"]
    widths = {k: max(len(k), 12) for k in keys}
    print(" | ".join(k.ljust(widths[k]) for k in keys))
    print("-+-".join("-" * widths[k] for k in keys))
    for row in rows:
        vals = []
        for k in keys:
            v = row.get(k, "")
            vals.append((f"{v:.6f}" if isinstance(v, float) else str(v)).ljust(widths[k]))
        print(" | ".join(vals))


def run_dummy_baselines_for_image(
    image,
    masks,
    out_dir: str | Path,
    image_id: str,
    methods: list[str] | None = None,
) -> list[dict]:
    """运行协议 smoke-test dummy baselines，并保存统一 outputs。

    dummy baselines 只用于验证 baseline protocol，不代表论文正式对比方法。
    """

    out_dir = Path(out_dir)
    methods = methods or list(DUMMY_BASELINES.keys())
    predictions = {}
    rows = []
    for method in methods:
        pred = run_dummy_baseline(method, image, masks)
        metrics = evaluate_baseline_prediction(pred, masks, image=image)
        save_baseline_prediction(pred, out_dir / method, method, image_id, metrics=metrics)
        predictions[method] = pred
        rows.append({"method": method, **metrics})
    # 同时保存一个 comparison.{json,csv}，便于和外部 baseline 输出格式保持一致。
    compare_methods_for_image(image, masks, predictions, out_dir)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run protocol-only dummy baselines for one image.")
    parser.add_argument("--config", default="configs/same_mask.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-id", default="image")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image = load_image(args.input, size=int(cfg["image"]["size"]))
    mask_set = load_mask_file(args.mask, image_hw=image.shape[-2:], cfg=cfg)
    rows = run_dummy_baselines_for_image(image, mask_set.masks, args.out, args.image_id)
    _print_rows(rows)
    print(f"dummy baseline outputs saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.baselines.independent_gaussian import run_independent_gaussian_baseline
from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.evaluation.report import print_metrics
from shapesplat.frontend.file_mask_loader import load_mask_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent per-mask Gaussian baseline.")
    parser.add_argument("--config", default="configs/benchmark_baseline.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", default="outputs/independent_gaussian/image")
    parser.add_argument("--image-id", default="image")
    args = parser.parse_args()
    cfg = load_config(args.config)
    image = load_image(args.input, cfg["image"]["size"])
    masks = load_mask_file(args.mask, image_hw=image.shape[-2:], cfg=cfg).masks
    row = run_independent_gaussian_baseline(image, masks, cfg, args.out, image_id=args.image_id)
    print_metrics(row)
    print(f"independent gaussian baseline saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

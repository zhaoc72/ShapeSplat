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

from shapesplat.baselines.export_inputs import export_baseline_inputs
from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.frontend.file_mask_loader import load_mask_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Export shared same-mask inputs for external baselines.")
    parser.add_argument("--config", default="configs/same_mask.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-id", default="image")
    parser.add_argument("--crop-padding", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image = load_image(args.input, size=int(cfg["image"]["size"]))
    mask_set = load_mask_file(args.mask, image_hw=image.shape[-2:], cfg=cfg)
    spec = export_baseline_inputs(image, mask_set.masks, args.out, args.image_id, crop_padding=args.crop_padding)
    print(f"baseline inputs saved to: {Path(args.out).resolve()}")
    print(f"image_id: {spec.image_id}")
    print(f"num_objects: {spec.num_objects}")


if __name__ == "__main__":
    main()


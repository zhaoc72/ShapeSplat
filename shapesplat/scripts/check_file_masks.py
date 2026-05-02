from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image, save_tensor_image
from shapesplat.frontend.file_mask_loader import load_mask_file
from shapesplat.utils.visualization import save_input_with_mask_overlay, save_mask_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Check file masks for same-mask protocol.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--config", default="configs/minimal.yaml")
    parser.add_argument("--out", default="outputs/check_file_masks")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(args.image, size=int(cfg["image"]["size"]))
    masks = load_mask_file(args.mask, image.shape[-2:], cfg)

    area = masks.masks.flatten(1).sum(dim=1) / float(image.shape[-1] * image.shape[-2])
    print(f"num masks: {masks.masks.shape[0]}")
    print(f"masks shape: {tuple(masks.masks.shape)}")
    print(f"confidences: {masks.confidences.detach().cpu().tolist()}")
    print(f"boxes: {masks.boxes.detach().cpu().tolist()}")
    print(f"area ratios: {area.detach().cpu().tolist()}")

    save_tensor_image(image, out_dir / "input.png")
    save_mask_grid(masks.masks, out_dir / "masks.png")
    save_input_with_mask_overlay(image, masks.masks, out_dir / "input_mask_overlay.png")
    print("file mask check ok")


if __name__ == "__main__":
    main()

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
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.sam_backend import build_sam_backend
from shapesplat.utils.visualization import save_input_with_mask_overlay, save_mask_grid


def main() -> None:
    """检查当前配置下 SAM backend 是否能输出 retained visible masks。"""
    parser = argparse.ArgumentParser(description="Check ShapeSplat++ SAM backend.")
    parser.add_argument("--config", default="configs/minimal.yaml", help="配置文件路径")
    parser.add_argument("--input", default=None, help="可选输入图片")
    parser.add_argument("--backend", default=None, help="覆盖 frontend.sam_backend: stub/real/auto")
    parser.add_argument("--out", default="outputs/check_sam_backend", help="输出目录")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend is not None:
        cfg["frontend"]["sam_backend"] = args.backend
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input or cfg["image"].get("input_path")
    if input_path:
        image = load_image(input_path, size=int(cfg["image"]["size"]))
    else:
        image = make_synthetic_image(int(cfg["image"]["size"]))
    save_tensor_image(image, out_dir / "input.png")

    sam = build_sam_backend(cfg)
    mask_set = sam.predict_masks(image)
    print(f"backend type: {type(sam).__name__}")
    print(f"num masks: {mask_set.masks.shape[0]}")
    print(f"mask shapes: {tuple(mask_set.masks.shape)}")
    print(f"confidences: {[float(v) for v in mask_set.confidences.detach().cpu()]}")
    print(f"boxes: {mask_set.boxes.detach().cpu().tolist()}")
    h, w = mask_set.masks.shape[-2:]
    areas = mask_set.masks.flatten(1).sum(dim=1) / float(h * w)
    print(f"area ratios: {[float(v) for v in areas.detach().cpu()]}")

    save_mask_grid(mask_set.masks, out_dir / "masks.png")
    save_input_with_mask_overlay(image, mask_set.masks, out_dir / "input_mask_overlay.png")
    print(f"SAM backend check outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image, save_tensor_image
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.depth_backend import build_depth_backend
from shapesplat.frontend.depth_normalization import normalize_depth_to_canonical
from shapesplat.frontend.sam_backend import build_sam_backend
from shapesplat.utils.visualization import save_depth_map, save_mask_grid


def _stats(x: torch.Tensor) -> dict:
    finite = torch.isfinite(x)
    vals = x[finite]
    return {
        "min": float(vals.min()) if vals.numel() else 0.0,
        "max": float(vals.max()) if vals.numel() else 0.0,
        "mean": float(vals.mean()) if vals.numel() else 0.0,
        "finite_ratio": float(finite.float().mean()),
    }


def main() -> None:
    """检查 depth backend 的 raw depth 和 canonical normalized depth，不检查 reconstruction。"""
    parser = argparse.ArgumentParser(description="Check ShapeSplat++ depth backend.")
    parser.add_argument("--config", default="configs/minimal.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--out", default="outputs/check_depth_backend")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend is not None:
        cfg["frontend"]["depth_backend"] = args.backend
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input or cfg["image"].get("input_path")
    image = load_image(input_path, size=int(cfg["image"]["size"])) if input_path else make_synthetic_image(int(cfg["image"]["size"]))
    save_tensor_image(image, out_dir / "input.png")

    masks = build_sam_backend(cfg).predict_masks(image)
    save_mask_grid(masks.masks, out_dir / "masks.png")
    depth_model = build_depth_backend(cfg)
    raw = depth_model.predict_depth(image)
    norm = normalize_depth_to_canonical(raw, masks.masks, cfg, cfg["camera"]["z_near"], cfg["camera"]["z_far"])
    save_depth_map(raw, out_dir / "raw_depth.png", normalize=True)
    save_depth_map(norm, out_dir / "depth_normalized.png", normalize=True)

    raw_stats, norm_stats = _stats(raw), _stats(norm)
    stats = {"backend_type": type(depth_model).__name__, "image_shape": list(image.shape), "raw_depth_shape": list(raw.shape), "raw": raw_stats, "normalized": norm_stats}
    print(f"backend type: {stats['backend_type']}")
    print(f"image shape: {tuple(image.shape)}")
    print(f"raw depth shape: {tuple(raw.shape)}")
    print(f"raw depth min/max/mean: {raw_stats['min']:.6f} / {raw_stats['max']:.6f} / {raw_stats['mean']:.6f}")
    print(f"normalized depth min/max/mean: {norm_stats['min']:.6f} / {norm_stats['max']:.6f} / {norm_stats['mean']:.6f}")
    print(f"finite ratio: {norm_stats['finite_ratio']:.6f}")
    with open(out_dir / "depth_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Depth backend check outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

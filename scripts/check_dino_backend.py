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
from shapesplat.frontend.dino_backend import build_dino_backend
from shapesplat.frontend.sam_backend import build_sam_backend
from shapesplat.utils.visualization import save_mask_grid


def main() -> None:
    """检查 DINO backend 的 dense features 和 mask-guided descriptors。

    这个脚本只检查 feature/descriptor 接口，不检查 reconstruction 质量。
    """
    parser = argparse.ArgumentParser(description="Check ShapeSplat++ DINO backend.")
    parser.add_argument("--config", default="configs/minimal.yaml", help="配置文件路径")
    parser.add_argument("--input", default=None, help="可选输入图片")
    parser.add_argument("--backend", default=None, help="覆盖 frontend.dino_backend: stub/real/auto")
    parser.add_argument("--out", default="outputs/check_dino_backend", help="输出目录")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend is not None:
        cfg["frontend"]["dino_backend"] = args.backend
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input or cfg["image"].get("input_path")
    image = load_image(input_path, size=int(cfg["image"]["size"])) if input_path else make_synthetic_image(int(cfg["image"]["size"]))
    save_tensor_image(image, out_dir / "input.png")

    sam = build_sam_backend(cfg)
    masks = sam.predict_masks(image)
    save_mask_grid(masks.masks, out_dir / "masks.png")

    dino = build_dino_backend(cfg)
    features = dino.extract_dense_features(image)
    descriptors = dino.pool_descriptors(features, masks.masks)
    norms = descriptors.norm(dim=1) if descriptors.shape[0] else torch.zeros((0,))
    fmap_norm = features.norm(dim=0)
    if float(fmap_norm.max()) > 0:
        save_tensor_image((fmap_norm / fmap_norm.max()).clamp(0, 1), out_dir / "feature_norm.png")

    stats = {
        "backend_type": type(dino).__name__,
        "image_shape": list(image.shape),
        "features_shape": list(features.shape),
        "masks_shape": list(masks.masks.shape),
        "descriptors_shape": list(descriptors.shape),
        "descriptor_norm_min": float(norms.min()) if norms.numel() else 0.0,
        "descriptor_norm_max": float(norms.max()) if norms.numel() else 0.0,
        "descriptor_norm_mean": float(norms.mean()) if norms.numel() else 0.0,
    }
    print(f"backend type: {stats['backend_type']}")
    print(f"image shape: {tuple(image.shape)}")
    print(f"features shape: {tuple(features.shape)}")
    print(f"masks shape: {tuple(masks.masks.shape)}")
    print(f"descriptors shape: {tuple(descriptors.shape)}")
    print(
        "descriptor norm min / max / mean: "
        f"{stats['descriptor_norm_min']:.6f} / {stats['descriptor_norm_max']:.6f} / {stats['descriptor_norm_mean']:.6f}"
    )
    with open(out_dir / "descriptor_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"DINO backend check outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

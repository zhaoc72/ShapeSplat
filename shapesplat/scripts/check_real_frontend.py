from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.cache.frontend_cache import save_frontend_output
from shapesplat.config import load_config
from shapesplat.data.image_io import load_image, save_tensor_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.utils.visualization import save_depth_map, save_input_with_mask_overlay, save_mask_grid


def check_real_frontend(config: str, input_path: str, out: str, save_cache: bool, save_dino_features: bool) -> dict:
    """统一检查真实/auto/stub 前端输出。

    这里不会强制安装真实模型；auto backend 失败时应由各 backend factory fallback 到 stub。
    """
    cfg = load_config(config)
    image = load_image(input_path, cfg["image"]["size"])
    front = build_frontend(image, cfg)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    desc_norm = torch.linalg.norm(front.descriptors.detach().cpu(), dim=1)
    stats = {
        "mask_source": cfg["frontend"].get("mask_source"),
        "sam_backend": cfg["frontend"].get("sam_backend"),
        "dino_backend": cfg["frontend"].get("dino_backend"),
        "depth_backend": cfg["frontend"].get("depth_backend"),
        "masks_shape": list(front.masks.shape),
        "descriptors_shape": list(front.descriptors.shape),
        "descriptor_norm_min": float(desc_norm.min()) if desc_norm.numel() else None,
        "descriptor_norm_max": float(desc_norm.max()) if desc_norm.numel() else None,
        "descriptor_finite": bool(torch.isfinite(front.descriptors).all()),
        "depth_shape": list(front.depth.shape),
        "depth_finite": bool(torch.isfinite(front.depth).all()),
        "depth_min": float(front.depth.min()),
        "depth_max": float(front.depth.max()),
        "depth_mean": float(front.depth.mean()),
    }
    if front.masks.shape[0] < 1:
        raise RuntimeError("frontend produced no masks")
    if not stats["descriptor_finite"] or not stats["depth_finite"]:
        raise RuntimeError("frontend produced non-finite descriptor or depth")

    save_tensor_image(front.image, out_dir / "input.png")
    save_mask_grid(front.masks, out_dir / "masks.png")
    save_input_with_mask_overlay(front.image, front.masks, out_dir / "input_mask_overlay.png")
    save_depth_map(front.depth, out_dir / "depth.png")
    (out_dir / "frontend_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "descriptor_stats.json").write_text(
        json.dumps({"norms": [float(v) for v in desc_norm], "shape": list(front.descriptors.shape)}, indent=2),
        encoding="utf-8",
    )
    if save_cache:
        record = save_frontend_output(
            front,
            out_dir / "cache",
            image_id=Path(input_path).stem,
            save_dino_features=save_dino_features,
            save_visuals=True,
        )
        stats["cache_dir"] = record.cache_dir
    print(f"mask_source: {stats['mask_source']}")
    print(f"sam_backend: {stats['sam_backend']} dino_backend: {stats['dino_backend']} depth_backend: {stats['depth_backend']}")
    print(f"masks: {tuple(front.masks.shape)} descriptors: {tuple(front.descriptors.shape)} depth: {tuple(front.depth.shape)}")
    print(f"frontend check ok: {out_dir}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Check real/auto/stub frontend backend outputs.")
    parser.add_argument("--config", default="configs/local_real_frontend.yaml")
    parser.add_argument("--input", default="examples/test_image.png")
    parser.add_argument("--out", default="outputs/check_real_frontend")
    parser.add_argument("--save-cache", action="store_true")
    parser.add_argument("--save-dino-features", action="store_true")
    parser.add_argument("--force-stub-ok", action="store_true")
    args = parser.parse_args()
    check_real_frontend(args.config, args.input, args.out, args.save_cache, args.save_dino_features)


if __name__ == "__main__":
    main()

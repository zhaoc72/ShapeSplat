from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Windows/Anaconda 环境中 torch 与 matplotlib 可能加载两份 OpenMP runtime。
# 这里作为最小 demo 的兼容开关；真实实验环境建议从依赖层面统一 OpenMP runtime。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.data.image_io import load_image, save_tensor_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.optimization.trainer import Trainer
from shapesplat.utils.logging import save_json
from shapesplat.utils.seed import seed_everything
from shapesplat.utils.visualization import save_mask_grid, save_render_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ minimal pipeline.")
    parser.add_argument("--config", required=True, help="配置文件路径，例如 configs/minimal.yaml")
    parser.add_argument("--out", required=True, help="输出目录，例如 outputs/minimal")
    parser.add_argument("--input", default=None, help="可选输入 RGB 图像；为空时自动生成 synthetic 图")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 输入图像：用户未提供时生成 smoke-test synthetic 多物体图。
    input_path = args.input or cfg["image"].get("input_path")
    if input_path:
        image = load_image(input_path, int(cfg["image"]["size"]))
    else:
        image = make_synthetic_image(int(cfg["image"]["size"]))
    save_tensor_image(image, out_dir / "input.png")

    # 2. frozen front-end：SAM3 stub -> masks；DINOv3 stub -> descriptors；Depth stub -> weak depth。
    front = build_frontend(image, cfg)
    save_mask_grid(front.masks, out_dir / "masks.png")

    # 3. 初始化 visible-hidden Gaussian buffers，并按 visible/hidden/joint/edit schedule 优化。
    trainer = Trainer(front, cfg)
    loss_log = trainer.train()

    # 4. 最终渲染与输出：RGB/alpha/ownership/per-object alpha/log/checkpoint。
    render = trainer.render()
    save_render_outputs(render, out_dir)
    save_json(loss_log, out_dir / "loss_log.json")
    trainer.save_checkpoint(out_dir / "checkpoint_minimal.pt")
    print(f"ShapeSplat++ minimal outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

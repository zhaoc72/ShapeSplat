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
from shapesplat.data.image_io import load_image, save_tensor_image
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.optimization.trainer import Trainer
from shapesplat.utils.logging import save_json
from shapesplat.utils.seed import seed_everything
from shapesplat.utils.visualization import save_input_with_mask_overlay, save_mask_grid, save_render_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ minimal pipeline.")
    parser.add_argument("--config", required=True, help="配置文件路径，例如 configs/minimal.yaml")
    parser.add_argument("--out", required=True, help="输出目录，例如 outputs/minimal")
    parser.add_argument("--input", default=None, help="可选输入 RGB 图像；为空时使用 config 或 synthetic 图")
    return parser.parse_args()


def run_pipeline(config_path: str | Path, out: str | Path, input_path: str | Path | None = None) -> Path:
    """运行最小 ShapeSplat++ pipeline。

    输入优先级为：CLI --input > cfg['image']['input_path'] > synthetic image。
    该函数供 run_minimal.py 和 run_real_input_demo.py 复用，避免两套入口逻辑漂移。
    """
    cfg = load_config(config_path)
    seed_everything(int(cfg["seed"]))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_input = input_path or cfg["image"].get("input_path")
    if resolved_input:
        print(f"Using input image: {resolved_input}")
        image = load_image(resolved_input, size=int(cfg["image"]["size"]))
    else:
        print("No input image provided. Using synthetic image.")
        image = make_synthetic_image(int(cfg["image"]["size"]))
    save_tensor_image(image, out_dir / "input.png")

    # frozen front-end：SAM3 stub -> visible masks；DINOv3 stub -> descriptors；Depth stub -> weak depth。
    front = build_frontend(image, cfg)
    if front.masks.shape[0] == 0:
        raise RuntimeError("Front-end 没有检测到任何 mask；请检查输入图像或 Sam3Stub 前景启发式。")
    save_mask_grid(front.masks, out_dir / "masks.png")
    save_input_with_mask_overlay(front.image, front.masks, out_dir / "input_mask_overlay.png")

    # 初始化 visible-hidden Gaussian buffers，并执行分阶段优化。
    trainer = Trainer(front, cfg)
    loss_log = trainer.train()

    # 保存最终渲染、日志和 checkpoint。
    render = trainer.render()
    save_render_outputs(render, out_dir)
    save_json(loss_log, out_dir / "loss_log.json")
    trainer.save_checkpoint(out_dir / "checkpoint_minimal.pt")
    print(f"ShapeSplat++ minimal outputs saved to: {out_dir.resolve()}")
    return out_dir


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, args.out, args.input)


if __name__ == "__main__":
    main()

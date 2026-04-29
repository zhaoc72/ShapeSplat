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

import torch

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image, save_tensor_image
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.evaluation.edit_metrics import compute_edit_metrics
from shapesplat.evaluation.metrics import compute_basic_metrics
from shapesplat.evaluation.report import merge_metrics, print_metrics, save_metrics_json
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.optimization.trainer import Trainer
from shapesplat.utils.seed import seed_everything
from shapesplat.utils.visualization import save_input_with_mask_overlay, save_mask_grid, save_render_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ShapeSplat++ minimal outputs with 2D ownership/edit metrics.")
    parser.add_argument("--config", default="configs/minimal.yaml", help="配置文件路径")
    parser.add_argument("--input", default=None, help="可选输入 RGB 图像；为空时使用 config 或 synthetic 图")
    parser.add_argument("--out", default="outputs/eval_minimal", help="评估输出目录")
    parser.add_argument("--checkpoint", default=None, help="可选 checkpoint；结构不匹配时回退到重新训练结果")
    return parser.parse_args()


def maybe_load_checkpoint(trainer: Trainer, checkpoint: str | Path | None) -> None:
    """尝试加载 checkpoint 中的 scene state_dict。

    当前 scene 结构由 front-end mask 数量和采样数量决定，因此不同输入/配置可能不匹配；
    不匹配时给出 warning 并继续使用刚训练得到的 scene。
    """
    if checkpoint is None:
        return
    path = Path(checkpoint)
    if not path.exists():
        print(f"Warning: checkpoint not found: {path}. Using trained scene instead.")
        return
    try:
        ckpt = torch.load(path, map_location=trainer.front.image.device, weights_only=False)
        state = ckpt.get("scene") if isinstance(ckpt, dict) else None
        if state is None:
            print("Warning: checkpoint has no 'scene' key. Using trained scene instead.")
            return
        trainer.scene.load_state_dict(state)
        print(f"Loaded scene checkpoint: {path}")
    except Exception as exc:
        print(f"Warning: failed to load checkpoint ({exc}). Using trained scene instead.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input or cfg["image"].get("input_path")
    if input_path:
        print(f"Using input image: {input_path}")
        image = load_image(input_path, size=int(cfg["image"]["size"]))
    else:
        print("No input image provided. Using synthetic image.")
        image = make_synthetic_image(int(cfg["image"]["size"]))
    save_tensor_image(image, out_dir / "input.png")

    front = build_frontend(image, cfg)
    if front.masks.shape[0] == 0:
        raise RuntimeError("Front-end 没有检测到任何 mask，无法评估。")
    save_mask_grid(front.masks, out_dir / "masks.png")
    save_input_with_mask_overlay(front.image, front.masks, out_dir / "input_mask_overlay.png")

    trainer = Trainer(front, cfg)
    trainer.train()
    maybe_load_checkpoint(trainer, args.checkpoint)
    render = trainer.render()
    save_render_outputs(render, out_dir)

    metrics = merge_metrics(
        compute_basic_metrics(render, front.masks),
        compute_edit_metrics(trainer.scene, trainer.renderer, front, render, cfg, object_id=0),
    )
    save_metrics_json(metrics, out_dir / "metrics.json")
    print_metrics(metrics)
    print(f"Evaluation outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

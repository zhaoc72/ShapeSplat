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

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.evaluation.report import print_metrics
from shapesplat.experiments.single_image import run_single_image_experiment
from shapesplat.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ minimal pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--eval", action="store_true")
    return parser.parse_args()


def run_pipeline(config_path: str | Path, out: str | Path, input_path: str | Path | None = None, do_eval: bool = False) -> Path:
    """运行单图 minimal pipeline。

    输入优先级保持不变：CLI --input > cfg["image"]["input_path"] > synthetic image。
    实际单图训练/保存逻辑复用 experiments.single_image，避免 batch runner 和单图脚本重复。
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

    metrics = run_single_image_experiment(
        image=image,
        cfg=cfg,
        out_dir=out_dir,
        image_id="minimal",
        save_visuals=True,
        save_checkpoint=True,
        eval_metrics=do_eval,
    )
    if do_eval:
        print_metrics(metrics)
    print(f"ShapeSplat++ minimal outputs saved to: {out_dir.resolve()}")
    return out_dir


def main() -> None:
    args = parse_args()
    run_pipeline(args.config, args.out, args.input, do_eval=args.eval)


if __name__ == "__main__":
    main()

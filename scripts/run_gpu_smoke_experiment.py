from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.experiments.single_image import run_single_image_experiment
from shapesplat.runtime.cli import add_runtime_args, apply_runtime_cli_overrides, prepare_runtime_for_run, runtime_overrides_from_args
from shapesplat.runtime.memory import log_gpu_memory
from shapesplat.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny GPU-aware ShapeSplat++ smoke experiment.")
    parser.add_argument("--config", default="configs/local_windows_rtx5090.yaml")
    parser.add_argument("--out", default="outputs/gpu_smoke")
    parser.add_argument("--input", default=None)
    parser.add_argument("--iters", type=int, default=2)
    add_runtime_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, runtime_overrides_from_args(args))
    apply_runtime_cli_overrides(cfg, args)
    # 中文注释：smoke experiment 强制极少迭代，只验证 CUDA 端到端可运行。
    cfg["training"]["visible_warmup_iters"] = int(args.iters)
    cfg["training"]["hidden_prior_iters"] = 0
    cfg["training"]["joint_ownership_iters"] = 0
    cfg["training"]["edit_finetune_iters"] = 0
    seed_everything(int(cfg.get("seed", 0)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prepare_runtime_for_run(cfg, out, save_summary=True)
    log_gpu_memory("before ")
    image = load_image(args.input, size=int(cfg["image"]["size"])) if args.input else make_synthetic_image(int(cfg["image"]["size"]))
    row = run_single_image_experiment(image, cfg, out, image_id="gpu_smoke", save_checkpoint=False, eval_metrics=True)
    log_gpu_memory("after ")
    print(row)
    print(f"gpu smoke outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()

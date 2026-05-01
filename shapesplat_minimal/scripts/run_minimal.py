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
from shapesplat.reproducibility.finalize import finalize_run_outputs
from shapesplat.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ minimal pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--mask", default=None, help="可选 file mask，用于 same-mask 单图运行")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--no-run-metadata", action="store_true", help="不写入 run_info / registry 元数据")
    parser.add_argument("--registry", default="runs/run_registry.jsonl", help="全局 run registry 路径")
    return parser.parse_args()


def run_pipeline(
    config_path: str | Path,
    out: str | Path,
    input_path: str | Path | None = None,
    do_eval: bool = False,
    mask_path: str | Path | None = None,
) -> Path:
    """运行单图 minimal pipeline，并保持 CLI 旧输出结构不变。"""

    cfg = load_config(config_path)
    if mask_path is not None:
        cfg["frontend"]["mask_source"] = "file"
        cfg["frontend"]["mask_path"] = str(mask_path)
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
    print(f"Mask source: {cfg['frontend'].get('mask_source', 'sam')}")

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
    out_dir = run_pipeline(args.config, args.out, args.input, do_eval=args.eval, mask_path=args.mask)
    if not args.no_run_metadata:
        try:
            # 元数据写入是可复现实验追踪的辅助步骤，失败时不影响主实验结果。
            finalize_run_outputs(
                out_dir=out_dir,
                config_path=args.config,
                run_type="minimal",
                input_path=args.input,
                registry_path=args.registry,
            )
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")


if __name__ == "__main__":
    main()

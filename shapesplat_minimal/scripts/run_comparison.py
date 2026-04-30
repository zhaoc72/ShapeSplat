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
from shapesplat.datasets.image_dataset import build_dataset_from_manifest
from shapesplat.experiments.comparison_runner import run_comparison_dataset
from shapesplat.experiments.comparison_summary import save_comparison_summary
from shapesplat.evaluation.report import print_metrics
from shapesplat.utils.comparison_visualization import make_qualitative_index
from shapesplat.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run same-mask comparison: Ours + dummy baselines.")
    parser.add_argument("--config", default="configs/comparison_minimal.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/comparison_run")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-ours", action="store_true")
    parser.add_argument("--no-dummy-baselines", action="store_true")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # comparison runner 的默认协议是 same-mask；这里显式设置 file，避免 SAM proposal 干扰公平比较。
    cfg.setdefault("frontend", {})
    cfg["frontend"]["mask_source"] = "file"
    seed_everything(int(cfg["seed"]))

    dataset = build_dataset_from_manifest(args.manifest, image_size=int(cfg["image"]["size"]))
    rows = run_comparison_dataset(
        dataset,
        cfg,
        args.out,
        max_images=args.max_images,
        skip_existing=args.skip_existing,
        run_ours=not args.no_ours,
        run_dummy_baselines=not args.no_dummy_baselines,
        save_visuals=not args.no_visuals,
        save_checkpoint=args.save_checkpoint,
    )
    summary = save_comparison_summary(rows, args.out)
    make_qualitative_index(args.out, Path(args.out) / "qualitative_index.md")
    print_metrics(summary)
    print(f"comparison outputs saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()


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
from shapesplat.experiments.batch_runner import run_batch_experiment
from shapesplat.experiments.summary import save_batch_summary
from shapesplat.evaluation.report import print_metrics
from shapesplat.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ minimal pipeline on a manifest dataset.")
    parser.add_argument("--config", default="configs/minimal.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/dataset_run")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--mask-source", default=None, help="覆盖 frontend.mask_source: sam/file/auto")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mask_source is not None:
        cfg["frontend"]["mask_source"] = args.mask_source
    seed_everything(int(cfg["seed"]))
    dataset = build_dataset_from_manifest(args.manifest, image_size=int(cfg["image"]["size"]))
    rows = run_batch_experiment(
        dataset,
        cfg,
        args.out,
        max_images=args.max_images,
        skip_existing=args.skip_existing,
        save_visuals=not args.no_visuals,
        save_checkpoint=args.save_checkpoint,
        eval_metrics=True,
    )
    summary = save_batch_summary(rows, args.out)
    print_metrics(summary)
    print(f"Dataset run outputs saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

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
from shapesplat.evaluation.method_output_evaluator import evaluate_method_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one method output root on a benchmark manifest.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/final_ours.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = evaluate_method_dataset(args.method, args.outputs, args.manifest, cfg, args.out, max_images=args.max_images)
    print(f"evaluated rows: {len(rows)}")
    print(f"outputs saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

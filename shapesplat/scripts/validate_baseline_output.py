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

from shapesplat.baselines.validate_outputs import validate_baseline_output_dir
from shapesplat.evaluation.report import save_metrics_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a baseline output directory.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-objects", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    hw = (args.height, args.width) if args.height is not None and args.width is not None else None
    result = validate_baseline_output_dir(args.output, expected_num_objects=args.num_objects, image_hw=hw, strict=args.strict)
    save_metrics_json(result, Path(args.output) / "validation.json")
    print(f"valid: {result['valid']}")
    print(f"warnings: {result['warnings']}")
    print(f"errors: {result['errors']}")
    if args.strict and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


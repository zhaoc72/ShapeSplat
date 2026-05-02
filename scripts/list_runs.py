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

from shapesplat.reproducibility.registry import find_runs, print_run_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="List ShapeSplat++ run registry.")
    parser.add_argument("--registry", default="runs/run_registry.jsonl")
    parser.add_argument("--run-type", default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--max-rows", type=int, default=20)
    args = parser.parse_args()
    print_run_registry(find_runs(args.registry, run_type=args.run_type, status=args.status), max_rows=args.max_rows)


if __name__ == "__main__":
    main()


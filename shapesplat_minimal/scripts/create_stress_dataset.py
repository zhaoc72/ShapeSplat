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

from shapesplat.benchmarks.stress_generator import DEFAULT_SUBSETS, create_stress_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic object-centric stress dataset.")
    parser.add_argument("--out", default="examples/stress_dataset")
    parser.add_argument("--num-per-subset", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--subsets", default=None, help="comma-separated subset names")
    args = parser.parse_args()
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()] if args.subsets else DEFAULT_SUBSETS
    manifest = create_stress_dataset(args.out, args.num_per_subset, args.size, subsets=subsets, seed=args.seed)
    for subset in subsets:
        print(f"{subset}: {args.num_per_subset}")
    print(f"Stress dataset manifest saved to: {manifest.resolve()}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.evaluation.report import save_metrics_csv
from shapesplat.shape_prior.descriptor_precompute import precompute_shape_descriptors


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute minimal descriptors for a file shape bank.")
    parser.add_argument("--input", default="examples/shape_bank")
    parser.add_argument("--out", default="outputs/shape_bank_precomputed")
    parser.add_argument("--descriptor-dim", type=int, default=16)
    parser.add_argument("--mode", choices=["point_stats", "random"], default="point_stats")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    rows = precompute_shape_descriptors(args.input, args.out, args.descriptor_dim, args.mode, args.seed)
    out = Path(args.out)
    (out / "precompute_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_metrics_csv(rows, out / "precompute_summary.csv")
    print(f"precomputed descriptors: {len(rows)} -> {out}")


if __name__ == "__main__":
    main()

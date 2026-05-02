from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.datasets.benchmark.summary_v2 import save_benchmark_summary, summarize_benchmark_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark manifest v2.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/benchmark_summary")
    args = parser.parse_args()
    summary = summarize_benchmark_v2(args.manifest)
    save_benchmark_summary(summary, args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


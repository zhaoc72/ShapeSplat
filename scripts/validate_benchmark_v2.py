from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.datasets.benchmark.summary_v2 import save_benchmark_summary, summarize_benchmark_v2
from shapesplat.datasets.benchmark.validator_v2 import save_benchmark_v2_validation, validate_benchmark_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark manifest v2.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="outputs/benchmark_v2_validation")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--no-check-optional-gt", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else None
    report = validate_benchmark_v2(args.manifest, cfg=cfg, max_rows=args.max_rows, check_optional_gt=not args.no_check_optional_gt, check_cache=args.check_cache)
    save_benchmark_v2_validation(report, args.out)
    save_benchmark_summary(summarize_benchmark_v2(args.manifest), args.out)
    print(f"valid: {report['valid']} rows: {report['num_rows']} failed: {report['num_failed']} warnings: {report['num_warnings']}")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


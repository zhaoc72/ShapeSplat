from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.benchmarks.standard.validator import save_validation_report, validate_benchmark_manifest
from shapesplat.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a same-mask benchmark manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/same_mask.yaml")
    parser.add_argument("--out", default="outputs/benchmark_validation")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    report = validate_benchmark_manifest(args.manifest, load_config(args.config), args.max_rows)
    save_validation_report(report, args.out)
    print(f"valid: {report['valid']}")
    print(f"num_rows: {report['num_rows']} num_valid: {report['num_valid']} num_failed: {report['num_failed']}")
    if report.get("errors"):
        print(f"errors: {report['errors']}")
    print(f"validation saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

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

from shapesplat.reporting.io import load_json
from shapesplat.reporting.report import generate_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ShapeSplat++ experiment report.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="ShapeSplat++ Experiment Report")
    args = parser.parse_args()

    manifest = generate_experiment_report(args.root, args.out, args.title)
    sanity = load_json(manifest["metric_sanity"])
    print(f"report: {manifest['report']}")
    print(f"metric sanity bad rows: {sanity.get('num_bad_rows', 0)}")
    print(f"failure cases: {manifest.get('num_failure_cases', 0)}")


if __name__ == "__main__":
    main()


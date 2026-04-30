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

from shapesplat.reporting.diagnostics import (
    detect_failure_cases,
    metric_sanity_check,
    select_best_worst_cases,
    summarize_failures,
)
from shapesplat.reporting.io import load_json, save_json


def _bool(text: str) -> bool:
    return str(text).lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose metrics rows for sanity, best/worst, and failures.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", default="AttrAcc")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--higher-is-better", default="true")
    args = parser.parse_args()

    data = load_json(args.metrics)
    if isinstance(data, dict):
        print("Warning: input looks like summary dict; best/worst diagnostics are more useful on per-image rows.")
        rows = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
    else:
        rows = data
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sanity = metric_sanity_check(rows)
    best_worst = select_best_worst_cases(rows, args.metric, _bool(args.higher_is_better), top_k=args.top_k)
    failures = detect_failure_cases(rows)
    save_json(sanity, out / "metric_sanity.json")
    save_json(best_worst, out / "best_worst_cases.json")
    save_json(failures, out / "failure_cases.json")
    save_json(summarize_failures(failures), out / "failure_summary.json")
    print(f"rows: {sanity['num_rows']}")
    print(f"bad rows: {sanity['num_bad_rows']}")
    print(f"failure cases: {len(failures)}")


if __name__ == "__main__":
    main()


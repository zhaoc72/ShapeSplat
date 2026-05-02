from __future__ import annotations

import argparse
import json
from pathlib import Path


COLUMNS = ["op", "num_edits", "CollateralL1_mean", "AlphaCollateral_mean", "EditLocality_mean", "DeletionResidual_mean", "ObjectSupportIoU_mean"]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, int):
        return str(v)
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print object editing summary table.")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = list(data.values()) if isinstance(data, dict) else data
    widths = {c: len(c) for c in COLUMNS}
    for row in rows:
        for col in COLUMNS:
            widths[col] = max(widths[col], len(_fmt(row.get(col, ""))))
    print(" | ".join(c.ljust(widths[c]) for c in COLUMNS))
    print("-+-".join("-" * widths[c] for c in COLUMNS))
    for row in rows:
        print(" | ".join(_fmt(row.get(c, "")).ljust(widths[c]) for c in COLUMNS))


if __name__ == "__main__":
    main()


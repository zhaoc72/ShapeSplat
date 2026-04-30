from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = [
    "method",
    "num_success",
    "AttrAcc_mean",
    "AttrPurity_mean_mean",
    "Leakage_mean",
    "InstIoU_mean_mean",
    "IsoIoU_mean_mean",
    "ForegroundAlphaError_mean",
    "CollateralL1_mean",
    "EditLocality_mean",
    "DeletionResidual_mean",
]


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print per-method comparison summary.")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    path = Path(args.summary)
    if not path.exists():
        raise FileNotFoundError(f"comparison summary not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = list(data.values()) if isinstance(data, dict) else data
    widths = {k: len(k) for k in KEYS}
    for row in rows:
        for k in KEYS:
            widths[k] = max(widths[k], len(_fmt(row.get(k, ""))))
    print(" | ".join(k.ljust(widths[k]) for k in KEYS))
    print("-+-".join("-" * widths[k] for k in KEYS))
    for row in rows:
        print(" | ".join(_fmt(row.get(k, "")).ljust(widths[k]) for k in KEYS))


if __name__ == "__main__":
    main()


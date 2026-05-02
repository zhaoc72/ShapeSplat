from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a lightweight baseline summary table.")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    path = Path(args.summary)
    if not path.exists():
        raise FileNotFoundError(f"summary not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = rows.get("rows", [])

    keys = ["image_id", "method", "AttrAcc", "AttrPurity_mean", "Leakage", "InstIoU_mean", "ForegroundRGBL1"]
    widths = {k: max(len(k), 12) for k in keys}
    for row in rows:
        for key in keys:
            widths[key] = max(widths[key], len(_fmt(row.get(key, ""))))
    print(" | ".join(k.ljust(widths[k]) for k in keys))
    print("-+-".join("-" * widths[k] for k in keys))
    for row in rows:
        print(" | ".join(_fmt(row.get(k, "")).ljust(widths[k]) for k in keys))


if __name__ == "__main__":
    main()


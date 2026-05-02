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
from shapesplat.reporting.latex import save_latex_table
from shapesplat.reporting.tables import ABLATION_COLUMNS, COMPARISON_COLUMNS, flatten_summary, select_table_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a LaTeX table from experiment summary JSON.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--kind", choices=["comparison", "ablation", "generic"], default="generic")
    args = parser.parse_args()

    rows = flatten_summary(load_json(args.input))
    columns = COMPARISON_COLUMNS if args.kind == "comparison" else ABLATION_COLUMNS if args.kind == "ablation" else list(rows[0].keys())
    selected = select_table_columns(rows, columns)
    save_latex_table(selected, columns, args.caption, args.label, args.out)
    print(f"latex table saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()


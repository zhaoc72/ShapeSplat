from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.reporting.all_final_tables import export_all_final_tables


def main() -> None:
    # scripts 只做 CLI wrapper；表格导出逻辑在 shapesplat.reporting.all_final_tables。
    parser = argparse.ArgumentParser(description="Export all final paper tables.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--columns", default="configs/paper/table_columns.yaml")
    args = parser.parse_args()
    written = export_all_final_tables(args.root, args.out, args.columns)
    for k, v in written.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

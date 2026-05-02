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

from shapesplat.reporting.final_tables import export_final_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final comparison tables.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    written = export_final_tables(args.summary, args.out)
    for name, paths in written.items():
        print(f"{name}: {paths['csv']} | {paths['tex']}")


if __name__ == "__main__":
    main()

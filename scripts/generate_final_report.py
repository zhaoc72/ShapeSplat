from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.reporting.final_report import generate_final_report


def main() -> None:
    # 中文注释：从 final paper 输出目录汇总 Markdown 报告，便于最终人工检查。
    parser = argparse.ArgumentParser(description="Generate final paper report.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="ShapeSplat++ Final Paper Report")
    args = parser.parse_args()
    manifest = generate_final_report(args.root, args.out, args.title)
    print(f"final report: {manifest['report']}")


if __name__ == "__main__":
    main()

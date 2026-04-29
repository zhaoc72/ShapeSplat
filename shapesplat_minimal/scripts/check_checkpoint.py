from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.utils.checks import check_checkpoint


def main() -> None:
    """命令行入口：检查 checkpoint_minimal.pt 是否可重新加载。"""
    parser = argparse.ArgumentParser(description="Check ShapeSplat++ minimal checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/minimal/checkpoint_minimal.pt", help="checkpoint 路径")
    args = parser.parse_args()
    check_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()

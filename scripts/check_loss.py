from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.utils.checks import check_loss_log


def main() -> None:
    """命令行入口：检查 loss_log.json 是否可读、非空且没有 NaN/Inf。"""
    parser = argparse.ArgumentParser(description="Check ShapeSplat++ minimal loss log.")
    parser.add_argument("--log", default="outputs/minimal/loss_log.json", help="loss_log.json 路径")
    args = parser.parse_args()
    check_loss_log(args.log)


if __name__ == "__main__":
    main()

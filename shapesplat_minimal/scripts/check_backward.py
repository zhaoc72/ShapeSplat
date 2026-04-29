from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.utils.checks import check_backward


def main() -> None:
    """命令行入口：检查指定 stage 的 loss 是否能反向传播到 Gaussian 参数。"""
    parser = argparse.ArgumentParser(description="Check ShapeSplat++ backward graph.")
    parser.add_argument("--config", default="configs/minimal.yaml", help="配置文件路径")
    parser.add_argument("--stage", default="visible", help="loss stage，例如 visible/hidden/joint/edit")
    args = parser.parse_args()
    check_backward(load_config(args.config), stage=args.stage)


if __name__ == "__main__":
    main()

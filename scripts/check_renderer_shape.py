from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.utils.checks import check_renderer_shapes


def main() -> None:
    """命令行入口：重建最小 scene 并检查 renderer 输出张量形状。"""
    parser = argparse.ArgumentParser(description="Check ShapeSplat++ renderer output shapes.")
    parser.add_argument("--config", default="configs/minimal.yaml", help="配置文件路径")
    args = parser.parse_args()
    check_renderer_shapes(load_config(args.config))


if __name__ == "__main__":
    main()

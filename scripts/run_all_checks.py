from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.utils.checks import check_backward, check_checkpoint, check_loss_log, check_renderer_shapes


def main() -> None:
    """一键执行 loss/checkpoint/renderer shape/backward 四项 sanity checks。"""
    parser = argparse.ArgumentParser(description="Run all ShapeSplat++ minimal sanity checks.")
    parser.add_argument("--config", default="configs/minimal.yaml", help="配置文件路径")
    parser.add_argument("--out", default="outputs/minimal", help="最小 demo 输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out)
    loss_log = out_dir / "loss_log.json"
    checkpoint = out_dir / "checkpoint_minimal.pt"
    cfg = load_config(args.config)

    print("[1/4] Checking loss log...")
    check_loss_log(loss_log)
    print("[2/4] Checking checkpoint...")
    check_checkpoint(checkpoint)
    print("[3/4] Checking renderer shapes...")
    check_renderer_shapes(cfg)
    print("[4/4] Checking backward...")
    check_backward(cfg, stage="visible")
    print("all checks passed")


if __name__ == "__main__":
    main()

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

from shapesplat.renderer.real_renderer_check import check_real_renderer


def main() -> None:
    # scripts 只做本地 renderer smoke test 的命令行入口。
    parser = argparse.ArgumentParser(description="Check real 3DGS renderer adapter and soft fallback.")
    parser.add_argument("--config", default="configs/real_3dgs_renderer.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--out", default="outputs/check_real_renderer")
    parser.add_argument("--backend", default=None, choices=["auto", "real", "soft"])
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--save-visuals", action="store_true")
    args = parser.parse_args()
    check_real_renderer(args.config, args.out, backend=args.backend, input_path=args.input, allow_fallback=args.allow_fallback, save_visuals=args.save_visuals or True)


if __name__ == "__main__":
    main()

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

from shapesplat.datasets.example_dataset import create_example_dataset


def main() -> None:
    """CLI 入口。

    可复用逻辑位于 `shapesplat.datasets.example_dataset`，脚本只负责参数解析，
    避免 tests 直接 import scripts 目录。
    """

    parser = argparse.ArgumentParser(description="Create a tiny same-mask example dataset.")
    parser.add_argument("--out", default="examples/example_dataset")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    manifest = create_example_dataset(args.out, args.num_images, args.size)
    print(f"Example dataset saved to: {manifest.parent.resolve()}")


if __name__ == "__main__":
    main()

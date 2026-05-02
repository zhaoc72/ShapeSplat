from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.cache.same_mask_export import cache_to_same_mask_dataset


def main() -> None:
    # scripts 只解析命令行参数；cache 到 same-mask 的转换逻辑在 shapesplat.cache。
    parser = argparse.ArgumentParser(description="Convert frontend cache masks to a same-mask dataset.")
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = cache_to_same_mask_dataset(args.image_manifest, args.cache_manifest, args.out, args.copy_images, args.overwrite)
    print(f"cached same-mask manifest: {path}")


if __name__ == "__main__":
    main()

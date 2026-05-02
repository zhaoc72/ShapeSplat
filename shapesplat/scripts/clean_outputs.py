from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SAFE_TARGETS = ["outputs", "runs", ".pytest_cache"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean generated ShapeSplat++ outputs and caches.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-runs", action="store_true")
    args = parser.parse_args()

    # 清理工具只处理运行产物，不删除 src/configs/examples/docs/tests。
    targets = ["outputs", ".pytest_cache"]
    if args.include_runs:
        targets.append("runs")
    for name in targets:
        path = Path(name)
        if not path.exists():
            continue
        print(f"{'would remove' if args.dry_run else 'remove'}: {path}")
        if not args.dry_run:
            shutil.rmtree(path)


if __name__ == "__main__":
    main()

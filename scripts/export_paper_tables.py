from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.paper_runner import generate_paper_tables


def main() -> None:
    # 轻量表格导出工具：从 paper 输出目录整理 CSV/LaTeX，不依赖 pandas。
    parser = argparse.ArgumentParser(description="Export paper CSV/LaTeX tables from paper outputs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--kind", default="all")
    parser.add_argument("--out", default=None)
    parser.add_argument("--columns", default="configs/paper/table_columns.yaml")
    args = parser.parse_args()
    written = generate_paper_tables(args.root, args.columns)
    if args.out:
        dst = Path(args.out)
        dst.mkdir(parents=True, exist_ok=True)
        for pair in written.values():
            for p in pair.values():
                src = Path(p)
                target = dst / src.name
                # 当 --out 正好是 root/tables 时，避免把文件复制到自身。
                if src.resolve() != target.resolve():
                    shutil.copy2(src, target)
    for kind, pair in written.items():
        if args.kind in ("all", kind):
            print(f"{kind}: {pair}")


if __name__ == "__main__":
    main()

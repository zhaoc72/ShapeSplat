from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.integration.config_templates import create_local_backend_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local backend integration template.")
    parser.add_argument("--out", default="configs/local_backend_template.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    # 生成模板只写配置文件，不下载模型、不安装真实 backend。
    create_local_backend_template(path)
    print(f"template saved to: {path}")


if __name__ == "__main__":
    main()

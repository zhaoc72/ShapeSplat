from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.validation.project_health import check_project_health


def main() -> None:
    # 快速工程健康检查：只读文件和配置，不运行训练或真实 backend。
    result = check_project_health()
    print(f"healthy: {result['healthy']}")
    print(f"warnings: {len(result['warnings'])}")
    print(f"errors: {len(result['errors'])}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["healthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.runtime.env import collect_runtime_environment


def main() -> None:
    # 中文注释：只打印环境信息，不运行 CUDA kernel，适合快速排查 conda/PyTorch/GPU。
    env = collect_runtime_environment()
    print(json.dumps(env, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

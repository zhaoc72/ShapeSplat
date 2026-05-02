from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.frontend.dinov3_dependency_check import check_dinov3_dependencies


def main() -> None:
    # 只检查依赖，不自动安装，避免破坏当前 CUDA PyTorch。
    report = check_dinov3_dependencies()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        print("Suggested install command:")
        print(report["install_command"])
        print("DINOv3 real backend is not ready until these packages are installed.")


if __name__ == "__main__":
    main()

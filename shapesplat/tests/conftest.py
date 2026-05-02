from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    # 测试直接使用源码树，避免依赖用户是否已经 pip install -e .
    sys.path.insert(0, str(SRC))

# 不覆盖 TMP/TEMP：让 pytest 使用系统临时目录，避免项目内历史临时目录在 Windows 上残留权限锁。

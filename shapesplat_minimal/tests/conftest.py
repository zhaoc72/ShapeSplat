from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    # 测试直接使用源码树，避免依赖用户是否已经 pip install -e .
    sys.path.insert(0, str(SRC))

TEST_TMP = ROOT / ".pytest_tmp"
TEST_TMP.mkdir(exist_ok=True)
os.environ.setdefault("TMP", str(TEST_TMP))
os.environ.setdefault("TEMP", str(TEST_TMP))
tempfile.tempdir = str(TEST_TMP)

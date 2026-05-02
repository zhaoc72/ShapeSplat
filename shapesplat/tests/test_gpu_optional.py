from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_matmul_optional():
    # 中文注释：可选本地 GPU 测试，不在无 GPU CI 中强制通过。
    x = torch.randn(64, 64, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    assert y.is_cuda

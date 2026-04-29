import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，便于 smoke test 复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

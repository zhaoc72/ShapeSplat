from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_cache_dir(path: str | Path) -> Path:
    """创建缓存目录。

    前端缓存主要用于真实 SAM3 / DINOv3 / Depth 推理结果复用，默认 stub 流程也可以使用。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_numpy(path: str | Path, array: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, array)


def load_numpy(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def save_torch(path: str | Path, tensor: torch.Tensor) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), p)


def load_torch(path: str | Path, map_location: str | torch.device = "cpu") -> torch.Tensor:
    return torch.load(Path(path), map_location=map_location)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

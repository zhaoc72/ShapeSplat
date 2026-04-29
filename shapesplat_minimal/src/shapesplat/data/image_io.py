from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
from PIL import Image


def ensure_dir(path: str | Path) -> None:
    """创建目录；如果传入文件路径，调用方应传 parent。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image(path: str | Path, size: int) -> torch.Tensor:
    """加载 RGB 图片并转为 [3,H,W] tensor。

    本项目统一使用 channel-first tensor layout：[C,H,W]。
    真实训练时可在这里接入更完整的数据增强和颜色管理。
    """
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    data = torch.from_numpy(np.array(img)).float()
    return data.permute(2, 0, 1).contiguous() / 255.0


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    """保存 [3,H,W] RGB 或 [H,W] grayscale tensor 到图片。

    输入会被 clamp 到 [0,1]；这让训练中间结果即使略越界也能可视化。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = tensor.detach().cpu().float().clamp(0, 1)
    if t.ndim == 3 and t.shape[0] == 3:
        arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(arr, mode="RGB").save(path)
    elif t.ndim == 2:
        arr = (t.numpy() * 255).astype("uint8")
        Image.fromarray(arr, mode="L").save(path)
    else:
        raise ValueError(f"不支持的 tensor 图像形状: {tuple(t.shape)}")

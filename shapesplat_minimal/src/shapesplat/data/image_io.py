from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def ensure_dir(path: str | Path) -> None:
    """自动创建目录。

    这里的参数语义是“目录路径”；如果调用方传入文件路径，应传入它的 parent。
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image(path: str | Path, size: int | None = None) -> torch.Tensor:
    """加载真实 RGB 图片，输出 [3,H,W]、float32、范围 [0,1] 的 tensor。

    支持 PNG / JPG / JPEG 等 PIL 能读取的常见格式，并自动转为 RGB。
    当前最小版本为了让 camera、renderer 和训练配置简单稳定，默认会在
    run_minimal.py 中统一 resize 到 config 的 image.size；后续真实实验可以在这里
    扩展为保持长宽比、padding 或多尺度输入。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"输入图像不存在: {p.resolve()}")
    if not p.is_file():
        raise FileNotFoundError(f"输入路径不是文件: {p.resolve()}")
    img = Image.open(p).convert("RGB")
    if size is not None:
        img = img.resize((int(size), int(size)), Image.BILINEAR)
    data = torch.from_numpy(np.array(img)).float()
    return data.permute(2, 0, 1).contiguous() / 255.0


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    """保存 tensor 图像，自动 clamp 到 [0,1] 并创建父目录。

    本项目统一使用 channel-first tensor layout：
    RGB 为 [3,H,W]，灰度图为 [H,W]。
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

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def ensure_dir(path: str | Path) -> None:
    """自动创建目录。

    这里的参数语义是“目录路径”；如果调用方传入文件路径，应传入它的 parent。
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def _resize_pil_keep_aspect(img: Image.Image, long_side: int) -> Image.Image:
    """按长边等比例缩放 PIL 图像。

    中文注释：CO3Dv2 原图通常接近 640x479，不能沿用 minimal 的 128x128
    square resize，否则 file mask、renderer 和 ownership 都会显得粗糙。
    """
    w, h = img.size
    if max(w, h) == int(long_side):
        return img
    scale = float(long_side) / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR)


def load_image(
    path: str | Path,
    size: int | None = None,
    resize_mode: str | None = None,
    long_side: int | None = None,
) -> torch.Tensor:
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
    mode = resize_mode or ("square" if size is not None else "none")
    if mode == "square" and size is not None:
        img = img.resize((int(size), int(size)), Image.BILINEAR)
    elif mode == "keep_aspect":
        img = _resize_pil_keep_aspect(img, int(long_side or size or max(img.size)))
    elif mode in ("none", None):
        pass
    else:
        raise ValueError(f"Unsupported image resize_mode: {mode}")
    data = torch.from_numpy(np.array(img)).float()
    return data.permute(2, 0, 1).contiguous() / 255.0


def image_resize_kwargs_from_cfg(cfg: dict) -> dict:
    """从 cfg.image 生成 load_image 参数，默认保持旧 minimal square 行为。"""
    image_cfg = cfg.get("image", cfg)
    mode = image_cfg.get("resize_mode", "square")
    return {
        "size": image_cfg.get("size"),
        "resize_mode": mode,
        "long_side": image_cfg.get("long_side", image_cfg.get("size")),
    }


def resize_mask_nearest(mask: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    """用 nearest resize binary/instance mask，并保持 0/1 边界。

    中文注释：file masks 是 retained visible masks，禁止使用 bilinear/bicubic，
    否则 CO3Dv2 边界会产生灰边并污染 same-mask protocol。
    """
    h, w = int(image_hw[0]), int(image_hw[1])
    if mask.shape[-2:] == (h, w):
        return mask.float()
    squeeze = False
    x = mask.float()
    if x.ndim == 2:
        x = x[None]
        squeeze = True
    y = F.interpolate(x[:, None], size=(h, w), mode="nearest")[:, 0]
    y = (y > 0.5).float()
    return y[0] if squeeze else y


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

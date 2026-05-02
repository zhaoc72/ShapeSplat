from __future__ import annotations

import torch
import numpy as np
from PIL import Image, ImageDraw


def make_synthetic_image(size: int) -> torch.Tensor:
    """生成一张简单多前景物体 RGB 图像。

    这只是用于 smoke test 的合成输入：背景接近白色，前景包含圆形、矩形和多边形。
    它不代表真实数据分布，目的只是让 SAM3/DINO/depth/renderer/optimizer 的完整链路可运行。

    Returns:
        torch.Tensor: [3, H, W]，float32，数值范围 [0, 1]。
    """
    img = Image.new("RGB", (size, size), (245, 245, 242))
    draw = ImageDraw.Draw(img)
    s = size
    draw.ellipse((int(0.08 * s), int(0.15 * s), int(0.38 * s), int(0.48 * s)), fill=(220, 55, 55))
    draw.rectangle((int(0.55 * s), int(0.12 * s), int(0.86 * s), int(0.42 * s)), fill=(45, 135, 225))
    pts = [
        (int(0.22 * s), int(0.72 * s)),
        (int(0.42 * s), int(0.55 * s)),
        (int(0.57 * s), int(0.83 * s)),
        (int(0.30 * s), int(0.90 * s)),
    ]
    draw.polygon(pts, fill=(60, 175, 95))
    arr = torch.from_numpy(np.array(img)).float()
    return arr.permute(2, 0, 1).contiguous() / 255.0

from __future__ import annotations

import torch


class DepthStub:
    """最小版本占位：生成 canonical monocular depth。

    后续可替换为 Depth Anything / VGGT / MapAnything 等 weak monocular depth wrapper。
    注意这里的 depth 只是初始化和 layout cue，不是 oracle geometry。
    """

    def __init__(self, z_near: float = 1.0, z_far: float = 3.2):
        self.z_near = z_near
        self.z_far = z_far

    def predict_depth(self, image: torch.Tensor) -> torch.Tensor:
        """输出 [H,W]，上方较远、下方较近。"""
        _, h, w = image.shape
        y = torch.linspace(0, 1, h, device=image.device).view(h, 1).expand(h, w)
        return self.z_far * (1 - y) + self.z_near * y

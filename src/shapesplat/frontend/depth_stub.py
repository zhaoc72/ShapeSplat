from __future__ import annotations

import torch


class DepthStub:
    """最小版本占位：生成 canonical monocular depth plane。

    该 stub 只用于 smoke test。真实版本可替换为 Depth Anything / VGGT /
    MapAnything 等 wrapper；depth 在本项目中只是 weak initialization/layout cue，
    不是 oracle geometry。
    """

    def __init__(self, cfg_or_z_near=1.0, z_far: float | None = None):
        if isinstance(cfg_or_z_near, dict):
            cfg = cfg_or_z_near
            self.z_near = float(cfg["camera"]["z_near"])
            self.z_far = float(cfg["camera"]["z_far"])
        else:
            self.z_near = float(cfg_or_z_near)
            self.z_far = float(3.2 if z_far is None else z_far)

    def predict_depth(self, image: torch.Tensor) -> torch.Tensor:
        """输出 [H,W]，范围已经是 canonical [z_near,z_far]。"""
        _, h, w = image.shape
        y = torch.linspace(0, 1, h, device=image.device).view(h, 1).expand(h, w)
        return self.z_far * (1 - y) + self.z_near * y

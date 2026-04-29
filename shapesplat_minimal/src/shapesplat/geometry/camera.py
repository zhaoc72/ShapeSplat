from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Camera:
    """最小 pinhole camera。

    单图重建通常没有真实内参；最小版本使用 canonical camera 作为尺度约定。
    后续若数据集提供 COLMAP/EXIF/标定内参，只需替换构造 camera 的入口。
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    device: torch.device

    @staticmethod
    def canonical(width: int, height: int, focal_scale: float, device: torch.device) -> "Camera":
        f = focal_scale * max(width, height)
        return Camera(width, height, f, f, (width - 1) / 2.0, (height - 1) / 2.0, device)

    def project(self, xyz: torch.Tensor) -> torch.Tensor:
        """把 [K,3] 3D 点投影到 [K,2] 像素坐标 uv。"""
        z = xyz[:, 2].clamp_min(1e-4)
        u = self.fx * xyz[:, 0] / z + self.cx
        v = self.fy * xyz[:, 1] / z + self.cy
        return torch.stack([u, v], dim=-1)

    def unproject(self, uv: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """把像素坐标 uv [K,2] 和深度 [K] 反投影到 [K,3]。"""
        x = (uv[:, 0] - self.cx) / self.fx * depth
        y = (uv[:, 1] - self.cy) / self.fy * depth
        return torch.stack([x, y, depth], dim=-1)

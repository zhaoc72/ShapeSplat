from __future__ import annotations

from typing import List

import torch

from .types import ShapeAsset


class SoftSupportField:
    """retrieved shapes 构成的 soft hidden support prior。

    这个 field 对任意 3D 查询点 x 输出 Phi(x)，表示它离检索 shape point cloud
    的 soft proximity。它只是 hidden branch 的弱先验，不是 hard template fitting；
    后续替换为真实 GSO/Objaverse shape bank 时仍然可以保持 support(x) 接口。
    """

    def __init__(self, shapes: List[ShapeAsset], weights: torch.Tensor, sigma: float, chunk_points: int = 2048):
        self.shapes = shapes
        self.weights = weights.detach()
        self.sigma = float(sigma)
        self.chunk_points = int(chunk_points)

    def _nearest_dist2(self, x: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """分块计算最近点距离，避免真实 shape point cloud 稍大时一次性 cdist 爆内存。"""
        points = points.to(x.device, x.dtype)
        if points.numel() == 0:
            return torch.full((x.shape[0],), float("inf"), device=x.device, dtype=x.dtype)
        best = torch.full((x.shape[0],), float("inf"), device=x.device, dtype=x.dtype)
        for start in range(0, points.shape[0], self.chunk_points):
            chunk = points[start : start + self.chunk_points]
            dist2 = torch.cdist(x, chunk).square().amin(dim=1)
            best = torch.minimum(best, dist2)
        return best

    def support(self, x: torch.Tensor) -> torch.Tensor:
        """查询 soft support。

        Args:
            x: [Q,3] hidden Gaussian centers 或任意 3D 查询点。

        Returns:
            [Q]，范围大致在 [0,1]。空 shape bank 时返回 0，不让训练崩溃。
        """
        if x.numel() == 0 or not self.shapes:
            return torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)

        fields = []
        sigma2 = max(self.sigma * self.sigma, 1e-12)
        for shape, w in zip(self.shapes, self.weights.to(x.device, x.dtype)):
            dist2 = self._nearest_dist2(x, shape.points)
            kappa = torch.exp(-dist2 / (2.0 * sigma2))
            fields.append(torch.log(kappa.clamp_min(1e-8)) + torch.log(w.clamp_min(1e-8)))
        return torch.logsumexp(torch.stack(fields, dim=0), dim=0).exp().clamp(0, 1)

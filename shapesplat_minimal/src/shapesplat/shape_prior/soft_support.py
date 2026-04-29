from __future__ import annotations

from typing import List

import torch

from .toy_shape_bank import ToyShape


class SoftSupportField:
    """retrieved shapes 诱导的 soft support prior。

    它是 soft prior，不是 hard template：hidden Gaussian 靠近检索形状会被鼓励，
    但不会强制每个点严格贴在某个模板上。
    """

    def __init__(self, shapes: List[ToyShape], weights: torch.Tensor, sigma: float):
        self.shapes = shapes
        self.weights = weights.detach()
        self.sigma = sigma

    def support(self, x: torch.Tensor) -> torch.Tensor:
        """计算查询点 x [Q,3] 的 support Phi(x)，输出 [Q]。"""
        if x.numel() == 0 or not self.shapes:
            return torch.zeros((x.shape[0],), device=x.device)
        fields = []
        for shape, w in zip(self.shapes, self.weights.to(x.device)):
            pts = shape.points.to(x.device)
            dist2 = torch.cdist(x, pts).square().amin(dim=1)
            kappa = torch.exp(-dist2 / (2 * self.sigma * self.sigma))
            fields.append(torch.log(kappa.clamp_min(1e-8)) + torch.log(w.clamp_min(1e-8)))
        return torch.logsumexp(torch.stack(fields, dim=0), dim=0).exp().clamp(0, 1)

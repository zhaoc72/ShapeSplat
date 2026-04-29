from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from .types import ShapeAsset


class ToyShapeBank:
    """最小 toy shape bank：sphere / box / cylinder。

    它只是 smoke test fallback，不是正式 shape prior。正式论文实验应替换为
    GSO / Objaverse / custom shape bank，并保证 train/test instance-disjoint。
    """

    def __init__(self, descriptor_dim: int, device: torch.device, points_per_shape: int = 96):
        self.descriptor_dim = descriptor_dim
        self.device = device
        self._assets: List[ShapeAsset] = [
            ShapeAsset("sphere", self._sphere(points_per_shape), self._desc(descriptor_dim, 0), category="toy"),
            ShapeAsset("box", self._box(points_per_shape), self._desc(descriptor_dim, 1), category="toy"),
            ShapeAsset("cylinder", self._cylinder(points_per_shape), self._desc(descriptor_dim, 2), category="toy"),
        ]

    @property
    def assets(self) -> list[ShapeAsset]:
        return self._assets

    @property
    def shapes(self) -> list[ShapeAsset]:
        """兼容旧代码命名。"""
        return self._assets

    def __len__(self) -> int:
        return len(self._assets)

    @property
    def descriptors(self) -> torch.Tensor:
        return torch.stack([a.descriptor for a in self._assets if a.descriptor is not None], dim=0)

    def _desc(self, d: int, idx: int) -> torch.Tensor:
        v = torch.zeros(d, device=self.device)
        v[idx % d] = 1.0
        v[(idx + 3) % d] = 0.5
        return F.normalize(v, dim=0)

    def _sphere(self, p: int) -> torch.Tensor:
        t = torch.linspace(0, 1, p, device=self.device)
        theta = 2 * torch.pi * t
        z = 2 * t - 1
        r = (1 - z.square()).clamp_min(0).sqrt()
        return torch.stack([0.5 * r * torch.cos(theta), 0.5 * r * torch.sin(theta), 0.5 * z], dim=-1)

    def _box(self, p: int) -> torch.Tensor:
        g = torch.Generator(device=self.device).manual_seed(11)
        x = torch.rand(p, 3, device=self.device, generator=g) - 0.5
        face = torch.randint(0, 3, (p,), device=self.device, generator=g)
        sign = torch.randint(0, 2, (p,), device=self.device, generator=g).float() * 2 - 1
        x[torch.arange(p, device=self.device), face] = 0.5 * sign
        return x

    def _cylinder(self, p: int) -> torch.Tensor:
        t = torch.linspace(0, 1, p, device=self.device)
        theta = 2 * torch.pi * t
        z = torch.linspace(-0.5, 0.5, p, device=self.device)
        return torch.stack([0.35 * torch.cos(theta), 0.35 * torch.sin(theta), z], dim=-1)

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class ToyShape:
    name: str
    points: torch.Tensor
    descriptor: torch.Tensor


class ToyShapeBank:
    """最小 toy shape bank：sphere / box / cylinder。

    真实版本应替换为 GSO / Objaverse / custom shape bank，并确保 train/test instance disjoint。
    本类只提供 normalized point cloud 和 toy descriptor，用于验证 hidden support prior 可运行。
    """

    def __init__(self, descriptor_dim: int, device: torch.device, points_per_shape: int = 96):
        self.descriptor_dim = descriptor_dim
        self.device = device
        self.shapes: List[ToyShape] = [
            ToyShape("sphere", self._sphere(points_per_shape), self._desc(descriptor_dim, 0)),
            ToyShape("box", self._box(points_per_shape), self._desc(descriptor_dim, 1)),
            ToyShape("cylinder", self._cylinder(points_per_shape), self._desc(descriptor_dim, 2)),
        ]

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
        x = torch.rand(p, 3, device=self.device) - 0.5
        face = torch.randint(0, 3, (p,), device=self.device)
        sign = torch.randint(0, 2, (p,), device=self.device).float() * 2 - 1
        x[torch.arange(p, device=self.device), face] = 0.5 * sign
        return x

    def _cylinder(self, p: int) -> torch.Tensor:
        t = torch.linspace(0, 1, p, device=self.device)
        theta = 2 * torch.pi * t
        z = torch.linspace(-0.5, 0.5, p, device=self.device)
        return torch.stack([0.35 * torch.cos(theta), 0.35 * torch.sin(theta), z], dim=-1)

    @property
    def descriptors(self) -> torch.Tensor:
        return torch.stack([s.descriptor for s in self.shapes], dim=0)

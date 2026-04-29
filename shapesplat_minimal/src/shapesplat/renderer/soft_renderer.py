from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from shapesplat.gaussian.scene import ObjectGaussianScene
from shapesplat.geometry.camera import Camera


@dataclass
class RenderOutput:
    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    contributions: torch.Tensor
    ownership: torch.Tensor
    bg_ownership: torch.Tensor


class SoftGaussianRenderer(nn.Module):
    """最小 PyTorch 可微分 Gaussian splatting 近似。

    真实版本应替换为 CUDA 3D Gaussian renderer，但保留输出字段，
    尤其是 per-object contribution maps / ownership，供 object-level optimization 和 edit loss 使用。
    """

    def __init__(self, camera: Camera, beta_depth: float = 1.5, min_sigma_px: float = 1.0, max_sigma_px: float = 4.0):
        super().__init__()
        self.camera = camera
        self.beta_depth = beta_depth
        self.min_sigma_px = min_sigma_px
        self.max_sigma_px = max_sigma_px

    def forward(self, scene: ObjectGaussianScene) -> RenderOutput:
        p = scene.all_parameters()
        means, colors, opacities, obj_ids = p["means"], p["colors"], p["opacities"][:, 0], p["object_ids"]
        uv = self.camera.project(means)
        h, w = self.camera.height, self.camera.width
        yy, xx = torch.meshgrid(torch.arange(h, device=means.device), torch.arange(w, device=means.device), indexing="ij")
        dx2 = (xx[None] - uv[:, 0, None, None]).square()
        dy2 = (yy[None] - uv[:, 1, None, None]).square()
        sigma = p["log_scales"].exp().mean(dim=1).mul(max(h, w) * 8).clamp(self.min_sigma_px, self.max_sigma_px)
        kernel = torch.exp(-(dx2 + dy2) / (2 * sigma[:, None, None].square()))
        alpha_i = (opacities[:, None, None] * kernel).clamp(0, 0.95)
        z = means[:, 2].clamp_min(1e-3)
        weighted = alpha_i * torch.exp(-self.beta_depth * z[:, None, None])
        denom = weighted.sum(dim=0).clamp_min(1e-6)
        rgb = (weighted[:, None] * colors[:, :, None, None]).sum(dim=0) / denom[None]
        alpha = 1.0 - torch.prod(1.0 - alpha_i.clamp(0, 0.95), dim=0)
        depth = (weighted * z[:, None, None]).sum(dim=0) / denom

        n_obj = len(scene.objects)
        contributions = []
        for n in range(n_obj):
            m = obj_ids == n
            contributions.append(weighted[m].sum(dim=0) if bool(m.any()) else torch.zeros((h, w), device=means.device))
        contributions = torch.stack(contributions, dim=0)
        bg = (1.0 - alpha).clamp(0, 1)
        own_denom = (bg + contributions.sum(dim=0)).clamp_min(1e-6)
        ownership = contributions / own_denom[None]
        bg_ownership = bg / own_denom
        rgb = rgb * alpha[None] + (1 - alpha[None])  # 白背景合成，便于和 synthetic 输入对齐。
        return RenderOutput(rgb.clamp(0, 1), alpha.clamp(0, 1), depth, contributions, ownership, bg_ownership)

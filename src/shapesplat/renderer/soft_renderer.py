from __future__ import annotations

from typing import Any

import torch
from torch import nn

from shapesplat.gaussian.scene import ObjectGaussianScene
from shapesplat.geometry.camera import Camera
from shapesplat.renderer.types import RenderOutput


class SoftGaussianRenderer(nn.Module):
    """最小 PyTorch differentiable Gaussian splatting approximation。

    这不是 CUDA 3DGS renderer，只是用 2D Gaussian kernel 和 depth-aware soft
    visibility 近似 alpha compositing。它的价值是稳定反向传播，并固定未来真实
    renderer 必须遵守的 `RenderOutput` 接口。
    """

    def __init__(
        self,
        camera: Camera,
        cfg_or_beta_depth: dict[str, Any] | float = 1.5,
        beta_depth: float | None = None,
        min_sigma_px: float = 1.0,
        max_sigma_px: float = 4.0,
    ):
        super().__init__()
        self.camera = camera
        if isinstance(cfg_or_beta_depth, dict):
            rcfg = cfg_or_beta_depth.get("renderer", cfg_or_beta_depth)
            self.beta_depth = float(rcfg.get("beta_depth", 1.5))
            self.min_sigma_px = float(rcfg.get("min_sigma_px", 1.0))
            self.max_sigma_px = float(rcfg.get("max_sigma_px", 4.0))
            self.ownership_eps = float(rcfg.get("ownership_eps", 1.0e-6))
        else:
            self.beta_depth = float(cfg_or_beta_depth if beta_depth is None else beta_depth)
            self.min_sigma_px = float(min_sigma_px)
            self.max_sigma_px = float(max_sigma_px)
            self.ownership_eps = 1.0e-6

    def forward(self, scene: ObjectGaussianScene) -> RenderOutput:
        if len(scene.objects) == 0:
            raise RuntimeError("SoftGaussianRenderer received an empty ObjectGaussianScene.")

        p = scene.all_parameters()
        means = p["means"]
        if means.numel() == 0:
            raise RuntimeError("SoftGaussianRenderer received zero Gaussians.")
        colors = p["colors"]
        opacities = p["opacities"][:, 0]
        obj_ids = p["object_ids"]

        uv = self.camera.project(means)
        h, w = self.camera.height, self.camera.width
        yy, xx = torch.meshgrid(
            torch.arange(h, device=means.device),
            torch.arange(w, device=means.device),
            indexing="ij",
        )
        dx2 = (xx[None] - uv[:, 0, None, None]).square()
        dy2 = (yy[None] - uv[:, 1, None, None]).square()
        sigma = p["log_scales"].exp().mean(dim=1).mul(max(h, w) * 8).clamp(self.min_sigma_px, self.max_sigma_px)
        kernel = torch.exp(-(dx2 + dy2) / (2 * sigma[:, None, None].square()))
        alpha_i = (opacities[:, None, None] * kernel).clamp(0, 0.95)

        # depth-aware visibility approximation：近处 Gaussian 获得更高 weighted alpha。
        z = means[:, 2].clamp_min(1e-3)
        weighted = alpha_i * torch.exp(-self.beta_depth * z[:, None, None])
        denom = weighted.sum(dim=0).clamp_min(1e-6)
        rgb = (weighted[:, None] * colors[:, :, None, None]).sum(dim=0) / denom[None]
        alpha = 1.0 - torch.prod(1.0 - alpha_i.clamp(0, 0.95), dim=0)
        depth = (weighted * z[:, None, None]).sum(dim=0) / denom

        n_obj = len(scene.objects)
        contributions = []
        for n in range(n_obj):
            mask = obj_ids == n
            if bool(mask.any()):
                contributions.append(weighted[mask].sum(dim=0))
            else:
                contributions.append(torch.zeros((h, w), device=means.device, dtype=means.dtype))
        contributions = torch.stack(contributions, dim=0)
        if contributions.ndim != 3 or contributions.shape[0] != n_obj:
            raise RuntimeError(f"Invalid contributions shape: {tuple(contributions.shape)}")

        bg = (1.0 - alpha).clamp(0, 1)
        own_denom = (bg + contributions.sum(dim=0)).clamp_min(self.ownership_eps)
        ownership = contributions / own_denom[None]
        bg_ownership = bg / own_denom

        # 白底合成，方便 synthetic/real-input smoke test 的可视化和 RGB loss。
        rgb = rgb * alpha[None] + (1 - alpha[None])
        return RenderOutput(
            rgb=rgb.clamp(0, 1),
            alpha=alpha.clamp(0, 1),
            depth=depth,
            contributions=contributions,
            ownership=ownership,
            bg_ownership=bg_ownership,
        )

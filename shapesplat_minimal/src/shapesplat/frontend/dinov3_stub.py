from __future__ import annotations

import torch
import torch.nn.functional as F

from shapesplat.frontend.dino_pooling import pool_mask_descriptors


class DinoV3Stub:
    """最小版本占位：用 RGB、坐标和正弦位置编码构造 dense features。

    这不是真实 DINOv3。DINOv3 在本项目中是 frozen descriptor extractor，
    只提供 object "what" 表征，不参与训练；真实替换时保持接口不变。
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def extract_dense_features(self, image: torch.Tensor) -> torch.Tensor:
        """输出 [D,H,W] dense feature map。"""
        _, h, w = image.shape
        ys = torch.linspace(-1, 1, h, device=image.device).view(1, h, 1).expand(1, h, w)
        xs = torch.linspace(-1, 1, w, device=image.device).view(1, 1, w).expand(1, h, w)
        pe = torch.cat(
            [torch.sin(3.1416 * xs), torch.cos(3.1416 * xs), torch.sin(3.1416 * ys), torch.cos(3.1416 * ys)],
            dim=0,
        )
        feats = torch.cat([image, xs, ys, pe], dim=0)
        if self.cfg.get("frontend", self.cfg).get("dino_l2_normalize", True):
            feats = F.normalize(feats, dim=0)
        return feats

    def pool_descriptors(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """使用统一 pooling 逻辑输出 [N,D] descriptor。"""
        return pool_mask_descriptors(features, masks, self.cfg)

from __future__ import annotations

import torch
import torch.nn.functional as F


class DinoV3Stub:
    """最小版本占位：用 RGB、坐标和正弦位置编码构造 dense features。

    DINOv3 在技术路线中是 frozen descriptor extractor，只负责提供 object "what" 表征，
    不参与训练；真实替换时保持 extract_dense_features / pool_descriptors 接口不变。
    """

    def extract_dense_features(self, image: torch.Tensor) -> torch.Tensor:
        """输出 [D,H,W] dense feature map。"""
        _, h, w = image.shape
        ys = torch.linspace(-1, 1, h, device=image.device).view(1, h, 1).expand(1, h, w)
        xs = torch.linspace(-1, 1, w, device=image.device).view(1, 1, w).expand(1, h, w)
        pe = torch.cat([torch.sin(3.1416 * xs), torch.cos(3.1416 * xs), torch.sin(3.1416 * ys), torch.cos(3.1416 * ys)], dim=0)
        feats = torch.cat([image, xs, ys, pe], dim=0)
        return F.normalize(feats, dim=0)

    def pool_descriptors(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """对每个 visible mask 做 mask pooling，输出 L2-normalized [N,D] descriptor。"""
        if masks.shape[0] == 0:
            return torch.zeros((0, features.shape[0]), device=features.device)
        weights = masks.float()
        denom = weights.flatten(1).sum(dim=1).clamp_min(1e-6)
        desc = torch.einsum("dhw,nhw->nd", features, weights) / denom[:, None]
        return F.normalize(desc, dim=-1)

from __future__ import annotations

import torch
from torch import nn


class GaussianObject(nn.Module):
    """单个物体的 visible-hidden Gaussian buffer。

    visible branch 主要由可见 mask/RGB/depth 强监督；
    hidden branch 只受 soft support prior 等弱约束，用于 plausible completion，
    当前最小版本不声称恢复真实不可见几何。
    """

    def __init__(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        opacity_logits: torch.Tensor,
        colors: torch.Tensor,
        branch_ids: torch.Tensor,
        object_id: int,
    ):
        super().__init__()
        self.means = nn.Parameter(means.float())
        self.log_scales = nn.Parameter(log_scales.float())
        self.opacity_logits = nn.Parameter(opacity_logits.float())
        self.colors = nn.Parameter(colors.float())
        self.register_buffer("branch_ids", branch_ids.long())
        self.object_id = object_id

    @property
    def visible_mask(self) -> torch.Tensor:
        return self.branch_ids == 0

    @property
    def hidden_mask(self) -> torch.Tensor:
        return self.branch_ids == 1

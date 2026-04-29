from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from shapesplat.shape_prior.soft_support import SoftSupportField
from .object_buffer import GaussianObject


@dataclass
class ObjectMeta:
    object_id: int
    mask: torch.Tensor
    descriptor: torch.Tensor
    retrieval_confidence: torch.Tensor
    support_field: SoftSupportField | None
    k_visible: int
    k_hidden: int


class ObjectGaussianScene(nn.Module):
    """多物体 Gaussian scene。

    每个 object buffer 独立存储，便于编辑和身份约束；
    渲染时把所有 object 联合 compositional rendering，得到 scene-coupled ownership。
    """

    def __init__(self, objects: List[GaussianObject], metas: List[ObjectMeta]):
        super().__init__()
        self.objects = nn.ModuleList(objects)
        self.metas = metas

    def all_parameters(self):
        """拼接所有 Gaussian 参数，renderer 使用该接口避免关心对象内部结构。"""
        means, colors, log_scales, opacities, obj_ids, branch_ids = [], [], [], [], [], []
        for obj in self.objects:
            k = obj.means.shape[0]
            means.append(obj.means)
            colors.append(obj.colors.sigmoid())
            log_scales.append(obj.log_scales)
            opacities.append(obj.opacity_logits.sigmoid())
            obj_ids.append(torch.full((k,), obj.object_id, device=obj.means.device, dtype=torch.long))
            branch_ids.append(obj.branch_ids)
        return {
            "means": torch.cat(means, dim=0),
            "colors": torch.cat(colors, dim=0),
            "log_scales": torch.cat(log_scales, dim=0),
            "opacities": torch.cat(opacities, dim=0),
            "object_ids": torch.cat(obj_ids, dim=0),
            "branch_ids": torch.cat(branch_ids, dim=0),
        }

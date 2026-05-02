from __future__ import annotations

import torch

from shapesplat.geometry.masks import dilate_mask


def get_object_support_mask(mask_n: torch.Tensor, ownership_n: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
    """编辑允许区域的基础 support：file/SAM mask 与 renderer ownership 的并集。"""

    return ((mask_n.float() > 0.5) | (ownership_n.float() > float(threshold))).float()


def make_edit_region(mask_n: torch.Tensor, ownership_n: torch.Tensor, radius: int = 3, threshold: float = 0.2) -> torch.Tensor:
    """编辑区域允许目标 object 发生变化，使用 dilation 给边界留出安全余量。"""

    return (dilate_mask(get_object_support_mask(mask_n, ownership_n, threshold), int(radius)) > 0.5).float()


def make_non_edit_region(mask_n: torch.Tensor, ownership_n: torch.Tensor, radius: int = 3, threshold: float = 0.2) -> torch.Tensor:
    """非编辑区域用于计算 collateral change，应尽量保持不变。"""

    return 1.0 - make_edit_region(mask_n, ownership_n, radius=radius, threshold=threshold)


def make_multi_object_non_edit_region(masks: torch.Tensor, ownership: torch.Tensor, object_id: int, radius: int = 3) -> torch.Tensor:
    """编辑 object_id 时，把目标 support 外的区域作为 non-edit region。"""

    if masks.shape[0] == 0 or ownership.shape[0] == 0:
        return torch.ones(masks.shape[-2:], device=masks.device)
    oid = min(int(object_id), masks.shape[0] - 1, ownership.shape[0] - 1)
    return make_non_edit_region(masks[oid], ownership[oid], radius=radius)


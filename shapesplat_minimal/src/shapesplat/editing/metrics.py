from __future__ import annotations

import torch

from shapesplat.editing.masks import make_edit_region, make_multi_object_non_edit_region


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.float().sum().clamp_min(1.0)
    return (x.float() * mask.float()).sum() / denom


def collateral_l1(base_rgb: torch.Tensor, edited_rgb: torch.Tensor, non_edit_mask: torch.Tensor) -> float:
    """非编辑区域 RGB L1；越低表示 collateral change 越小。"""

    return float(_masked_mean((base_rgb - edited_rgb).abs().mean(dim=0), non_edit_mask).detach().cpu())


def alpha_collateral(base_alpha: torch.Tensor, edited_alpha: torch.Tensor, non_edit_mask: torch.Tensor) -> float:
    """非编辑区域 alpha L1。"""

    return float(_masked_mean((base_alpha - edited_alpha).abs(), non_edit_mask).detach().cpu())


def edit_locality(base_rgb: torch.Tensor, edited_rgb: torch.Tensor, non_edit_mask: torch.Tensor) -> float:
    """编辑局部性，1 - CollateralL1 并 clamp 到 [0,1]。"""

    return max(0.0, min(1.0, 1.0 - collateral_l1(base_rgb, edited_rgb, non_edit_mask)))


def deletion_residual(edited_alpha: torch.Tensor, object_mask: torch.Tensor) -> float:
    """删除后原 object mask 内的残留 alpha。"""

    return float(_masked_mean(edited_alpha, object_mask > 0.5).detach().cpu())


def object_support_iou(before_ownership_n: torch.Tensor, after_ownership_n: torch.Tensor, mask_n: torch.Tensor, threshold: float = 0.2) -> float:
    """编辑后 object support 与原 visible mask 的 IoU，轻量检查 object support 是否合理。"""

    after = after_ownership_n.float() > float(threshold)
    target = mask_n.float() > 0.5
    inter = (after & target).float().sum()
    union = (after | target).float().sum().clamp_min(1.0)
    return float((inter / union).detach().cpu())


def compute_edit_metrics(
    base_render,
    edited_render,
    masks: torch.Tensor,
    object_id: int,
    op: str,
    radius: int = 3,
) -> dict:
    """计算 minimal editing diagnostics；这些指标不是 LPIPS/Chamfer。"""

    oid = min(int(object_id), masks.shape[0] - 1, base_render.ownership.shape[0] - 1)
    masks = masks.to(base_render.rgb.device).float()
    non_edit = make_multi_object_non_edit_region(masks, base_render.ownership.detach(), oid, radius=radius)
    obj_mask = masks[oid]
    out = {
        "EditOp": op,
        "ObjectID": int(object_id),
        "CollateralL1": collateral_l1(base_render.rgb.detach(), edited_render.rgb, non_edit),
        "AlphaCollateral": alpha_collateral(base_render.alpha.detach(), edited_render.alpha, non_edit),
        "EditLocality": edit_locality(base_render.rgb.detach(), edited_render.rgb, non_edit),
        "DeletionResidual": None,
        "ObjectSupportIoU": None,
    }
    if op == "remove":
        out["DeletionResidual"] = deletion_residual(edited_render.alpha, obj_mask)
    if oid < edited_render.ownership.shape[0]:
        out["ObjectSupportIoU"] = object_support_iou(base_render.ownership[oid].detach(), edited_render.ownership[oid], obj_mask)
    return out


from __future__ import annotations

import torch


def _bbox_extent(points: torch.Tensor) -> torch.Tensor:
    return (points.max(dim=0).values - points.min(dim=0).values).clamp_min(1e-8)


def center_align(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """中心对齐。alignment 会影响 geometry metric，正式实验必须固定协议。"""
    return pred - pred.mean(dim=0, keepdim=True), gt - gt.mean(dim=0, keepdim=True)


def scale_align_unit_bbox(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """分别缩放到 unit bbox，不做 ICP 或旋转。"""
    pred_c, gt_c = center_align(pred, gt)
    return pred_c / _bbox_extent(pred_c).max(), gt_c / _bbox_extent(gt_c).max()


def similarity_align_pred_to_gt(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """简化相似对齐：中心对齐，并按 RMS radius 缩放 pred 到 gt。"""
    pred_c, gt_c = center_align(pred, gt)
    pred_r = torch.sqrt(pred_c.square().sum(dim=1).mean()).clamp_min(1e-8)
    gt_r = torch.sqrt(gt_c.square().sum(dim=1).mean()).clamp_min(1e-8)
    return pred_c * (gt_r / pred_r), gt_c


def apply_alignment(pred: torch.Tensor, gt: torch.Tensor, mode: str = "none") -> tuple[torch.Tensor, torch.Tensor]:
    """应用轻量 alignment；不实现 ICP，不依赖外部几何库。"""
    if mode in (None, "none"):
        return pred, gt
    if mode == "center":
        return center_align(pred, gt)
    if mode in {"unit_bbox", "center_unit_bbox"}:
        return scale_align_unit_bbox(pred, gt)
    if mode == "similarity_scale":
        return similarity_align_pred_to_gt(pred, gt)
    raise ValueError(f"unknown alignment mode: {mode}")

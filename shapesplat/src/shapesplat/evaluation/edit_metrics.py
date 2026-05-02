from __future__ import annotations

from typing import Any

import torch

from shapesplat.geometry.masks import dilate_mask
from shapesplat.optimization.edit_ops import edited_scene


def edit_support_mask(mask: torch.Tensor, ownership_n: torch.Tensor, radius: int = 3, threshold: float = 0.2) -> torch.Tensor:
    """构造编辑区域 support。

    support 同时包含 retained mask 和 renderer 当前认为属于该 object 的区域，再膨胀一圈，
    用于把 edit 区域和 collateral 区域分开。
    """
    support = ((mask > 0.5) | (ownership_n > threshold)).float()
    return dilate_mask(support, radius).float()


def collateral_l1(base_rgb: torch.Tensor, edited_rgb: torch.Tensor, non_edit_mask: torch.Tensor) -> torch.Tensor:
    """在非编辑区域计算 RGB L1，作为 Collateral LPIPS 的轻量替代。"""
    denom = non_edit_mask.sum().clamp_min(1.0) * 3.0
    return ((base_rgb - edited_rgb).abs() * non_edit_mask[None]).sum() / denom


def edit_locality(base_rgb: torch.Tensor, edited_rgb: torch.Tensor, non_edit_mask: torch.Tensor) -> torch.Tensor:
    """计算 1 - normalized collateral L1，越高表示编辑越局部。"""
    return (1.0 - collateral_l1(base_rgb, edited_rgb, non_edit_mask)).clamp(0.0, 1.0)


def deletion_residual(edited_alpha: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
    """删除 object 后，在该 object 原 visible mask 区域内残留 alpha 的平均值，越低越好。"""
    m = (object_mask > 0.5).float()
    return (edited_alpha * m).sum() / m.sum().clamp_min(1.0)


def compute_edit_metrics(scene: Any, renderer: Any, front: Any, render: Any, cfg: dict, object_id: int = 0) -> dict:
    """计算 minimal edit metrics。

    当前仅做 remove edit，不依赖 LPIPS；后续实验版可把 CollateralL1 替换为 Collateral LPIPS，
    或加入更多 edit operation 的 batched 评估。
    """
    if front.masks.shape[0] == 0:
        raise ValueError("compute_edit_metrics 需要至少一个 mask。")
    object_id = int(max(0, min(object_id, front.masks.shape[0] - 1)))
    edited = edited_scene(scene, object_id, "remove")
    edited_render = renderer(edited)
    radius = int(cfg.get("edit", {}).get("dilate_radius", 3))
    support = edit_support_mask(front.masks[object_id], render.ownership[object_id].detach(), radius=radius)
    non_edit = 1.0 - support
    l1 = collateral_l1(render.rgb.detach(), edited_render.rgb, non_edit)
    locality = edit_locality(render.rgb.detach(), edited_render.rgb, non_edit)
    residual = deletion_residual(edited_render.alpha, front.masks[object_id])
    return {
        "CollateralL1": float(l1.detach().cpu()),
        "EditLocality": float(locality.detach().cpu()),
        "DeletionResidual": float(residual.detach().cpu()),
    }


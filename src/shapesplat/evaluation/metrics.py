from __future__ import annotations

from typing import Any

import torch


def safe_divide(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """安全除法，避免 mask 为空时出现除零。"""
    return num / den.clamp_min(eps)


def foreground_mask(masks: torch.Tensor) -> torch.Tensor:
    """计算 retained visible masks 的 union foreground mask。

    Args:
        masks: [N,H,W] instance masks。
    Returns:
        [H,W] float mask，1 表示前景。
    """
    if masks.numel() == 0 or masks.shape[0] == 0:
        return torch.zeros(masks.shape[-2:], device=masks.device, dtype=torch.float32)
    return (masks.float().amax(dim=0) > 0.5).float()


def inst_iou(ownership: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """计算每个 object 的 Inst-IoU。

    Inst-IoU 用于评估每个 object buffer 的 isolated / ownership support 是否与对应
    retained visible mask 一致。当前 minimal 版本用 ownership threshold 得到预测 support。
    """
    pred = ownership > threshold
    gt = masks > 0.5
    inter = (pred & gt).flatten(1).sum(dim=1).float()
    union = (pred | gt).flatten(1).sum(dim=1).float()
    return safe_divide(inter, union)


def iso_iou(ownership: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """minimal approximation 的 Iso-IoU。

    当前没有单独 render each object 的真实 isolated alpha，因此先用 ownership[n]
    近似 object alpha support。真实版本可替换为 isolated object rendering 的 alpha IoU。
    """
    return inst_iou(ownership, masks, threshold=threshold)


def attribution_accuracy(ownership: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """计算 foreground 像素上的 attribution accuracy。

    AttrAcc 衡量每个可见前景像素是否由正确 object buffer 解释，是 object ownership 的核心指标。
    """
    fg = foreground_mask(masks) > 0.5
    if not bool(fg.any()):
        return torch.tensor(0.0, device=ownership.device)
    pred_label = ownership.argmax(dim=0)
    gt_label = masks.argmax(dim=0)
    return (pred_label[fg] == gt_label[fg]).float().mean()


def attribution_purity(ownership: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """计算每个 object 的 ownership purity。

    如果 object n 的 contribution 大量落到其他 object 或背景，purity 会下降。
    """
    fg = foreground_mask(masks)
    vals = []
    for n in range(ownership.shape[0]):
        inside = (ownership[n] * (masks[n] > 0.5).float()).sum()
        total_fg = (ownership[n] * fg).sum()
        vals.append(safe_divide(inside, total_fg))
    return torch.stack(vals, dim=0) if vals else torch.zeros((0,), device=ownership.device)


def leakage(alpha: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """计算背景区域 alpha 泄漏。

    Leakage 衡量 foreground Gaussian alpha 是否泄漏到背景区域，越低越好。
    """
    bg = 1.0 - foreground_mask(masks)
    return safe_divide((alpha * bg).sum(), bg.sum())


def foreground_alpha_consistency(alpha: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """计算前景 alpha 与 1 的 L1 error，用于检查前景是否被渲染覆盖。"""
    fg = foreground_mask(masks) > 0.5
    if not bool(fg.any()):
        return torch.tensor(0.0, device=alpha.device)
    return (alpha[fg] - 1.0).abs().mean()


def _to_float(x: torch.Tensor | float) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def _to_list(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().cpu().flatten()]


def compute_basic_metrics(render: Any, masks: torch.Tensor) -> dict:
    """基于 RenderOutput 和 retained masks 计算 minimal 2D / ownership metrics。"""
    iou = inst_iou(render.ownership, masks)
    iso = iso_iou(render.ownership, masks)
    purity = attribution_purity(render.ownership, masks)
    metrics = {
        "InstIoU_mean": _to_float(iou.mean()) if iou.numel() else 0.0,
        "InstIoU_per_object": _to_list(iou),
        "IsoIoU_mean": _to_float(iso.mean()) if iso.numel() else 0.0,
        "AttrAcc": _to_float(attribution_accuracy(render.ownership, masks)),
        "AttrPurity_mean": _to_float(purity.mean()) if purity.numel() else 0.0,
        "AttrPurity_per_object": _to_list(purity),
        "Leakage": _to_float(leakage(render.alpha, masks)),
        "ForegroundAlphaError": _to_float(foreground_alpha_consistency(render.alpha, masks)),
    }
    return metrics


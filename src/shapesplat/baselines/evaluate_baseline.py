from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from shapesplat.evaluation.metrics import compute_basic_metrics, foreground_mask


@dataclass
class BaselineRenderLike:
    """给 baseline prediction 适配 compute_basic_metrics 的轻量包装。"""

    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    contributions: torch.Tensor
    ownership: torch.Tensor
    bg_ownership: torch.Tensor
    extras: dict = field(default_factory=dict)


def evaluate_baseline_prediction(
    prediction: dict,
    masks: torch.Tensor,
    image: Optional[torch.Tensor] = None,
) -> dict:
    """评估 baseline 输出。

    当前只计算 2D/ownership 指标和可选 ForegroundRGBL1，不包含 Chamfer、
    F-score 或 LPIPS；这些需要 GT mesh 或额外依赖，留给后续正式实验版本。
    """

    ownership = prediction["ownership"].float()
    alpha = prediction.get("alpha", ownership.sum(dim=0).clamp(0, 1)).float()
    rgb = prediction.get("rgb")
    if rgb is None:
        rgb = torch.ones(3, *alpha.shape, device=alpha.device)
    rgb = rgb.float()
    depth = prediction.get("depth", torch.zeros_like(alpha)).float()
    bg = prediction.get("bg_ownership", (1.0 - alpha).clamp(0, 1)).float()
    render = BaselineRenderLike(
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        contributions=ownership,
        ownership=ownership,
        bg_ownership=bg,
    )
    metrics = compute_basic_metrics(render, masks.float())
    if image is not None:
        fg = foreground_mask(masks) > 0.5
        if bool(fg.any()):
            metrics["ForegroundRGBL1"] = float((rgb[:, fg] - image.float()[:, fg]).abs().mean().detach().cpu())
        else:
            metrics["ForegroundRGBL1"] = 0.0
    return metrics


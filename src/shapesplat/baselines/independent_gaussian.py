from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from shapesplat.baselines.dummy_baselines import save_baseline_prediction
from shapesplat.baselines.evaluate_baseline import evaluate_baseline_prediction


def _smooth_mask(mask: torch.Tensor, steps: int = 2) -> torch.Tensor:
    x = mask.float()[None, None]
    for _ in range(max(1, steps)):
        x = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
    return x[0, 0].clamp(0, 1)


def run_independent_gaussian_baseline(
    image: torch.Tensor,
    masks: torch.Tensor,
    cfg: dict,
    out_dir: str | Path,
    image_id: str = "image",
    save_visuals: bool = True,
) -> dict:
    """运行 minimal independent per-mask Gaussian baseline。

    该 baseline 保留 object buffer separation，但每个 object 独立拟合自己的 mask/color，
    没有 scene-coupled ownership competition，因此可作为 Ours full 的快速对照。
    """
    masks = (masks.float() > 0.5).float()
    n, h, w = masks.shape
    device = image.device
    alpha_parts = []
    rgb_parts = []
    for i in range(n):
        alpha = _smooth_mask(masks[i]).to(device)
        support = masks[i] > 0.5
        color = image[:, support].mean(dim=1) if bool(support.any()) else torch.ones(3, device=device)
        rgb_parts.append(color.view(3, 1, 1).expand(3, h, w))
        alpha_parts.append(alpha)
    alpha_stack = torch.stack(alpha_parts, dim=0).clamp(0, 1)
    denom = alpha_stack.sum(dim=0, keepdim=True).clamp_min(1e-6)
    ownership = alpha_stack / denom
    alpha = alpha_stack.sum(dim=0).clamp(0, 1)
    rgb_num = torch.zeros_like(image.float())
    for i in range(n):
        rgb_num = rgb_num + rgb_parts[i] * alpha_stack[i][None]
    rgb = rgb_num / alpha_stack.sum(dim=0, keepdim=True).clamp_min(1e-6)
    rgb = rgb * alpha[None] + (1.0 - alpha[None])
    pred = {"rgb": rgb.clamp(0, 1), "alpha": alpha, "ownership": ownership, "bg_ownership": 1.0 - alpha}
    metrics = evaluate_baseline_prediction(pred, masks, image=image)
    save_baseline_prediction(pred, out_dir, "independent_gaussian", image_id, metrics=metrics)
    row = {"image_id": image_id, "method": "independent_gaussian", "status": "success", "output_dir": str(out_dir)}
    row.update(metrics)
    return row

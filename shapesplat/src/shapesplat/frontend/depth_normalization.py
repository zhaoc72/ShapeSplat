from __future__ import annotations

import torch


def make_fallback_depth(height: int, width: int, z_near: float, z_far: float, device: torch.device) -> torch.Tensor:
    """生成平滑 canonical depth plane，用于真实 depth 异常时兜底。"""
    y = torch.linspace(0, 1, height, device=device).view(height, 1).expand(height, width)
    return z_far * (1.0 - y) + z_near * y


def ensure_valid_depth(depth: torch.Tensor, z_near: float, z_far: float) -> torch.Tensor:
    """清理 NaN/Inf/非正值，并 clamp 到合理 canonical 范围。"""
    d = depth.float()
    finite = torch.isfinite(d) & (d > 0)
    if not bool(finite.any()):
        return make_fallback_depth(d.shape[0], d.shape[1], z_near, z_far, d.device)
    fill = d[finite].median()
    d = torch.where(finite, d, fill)
    return d.clamp(float(z_near), float(z_far))


def _foreground_for_stats(masks: torch.Tensor | None, depth: torch.Tensor, cfg: dict) -> torch.Tensor:
    if masks is None or masks.numel() == 0 or not cfg["frontend"].get("depth_normalize_on_foreground", True):
        return torch.ones_like(depth, dtype=torch.bool)
    fg = masks.to(depth.device).float().amax(dim=0) > 0.5
    if int(fg.sum()) < 8:
        return torch.ones_like(depth, dtype=torch.bool)
    return fg


def normalize_depth_to_canonical(
    depth: torch.Tensor,
    masks: torch.Tensor | None,
    cfg: dict,
    z_near: float,
    z_far: float,
) -> torch.Tensor:
    """把任意 monocular relative depth 归一化到 canonical camera range [z_near,z_far]。

    单目深度通常只有相对尺度，甚至不同模型的深度方向也可能不同；因此进入
    Gaussian 初始化和 weak depth loss 前必须做 canonical normalization。
    """
    d = depth.float()
    if d.ndim != 2:
        raise ValueError(f"normalize_depth_to_canonical 需要 [H,W]，实际: {tuple(d.shape)}")
    if not cfg["frontend"].get("depth_normalize", True):
        return ensure_valid_depth(d, z_near, z_far)

    finite = torch.isfinite(d) & (d > float(cfg["frontend"].get("depth_min_valid", 1e-6)))
    if not bool(finite.any()):
        return make_fallback_depth(d.shape[0], d.shape[1], z_near, z_far, d.device)
    d = torch.where(finite, d, d[finite].median())
    stats_mask = _foreground_for_stats(masks, d, cfg) & torch.isfinite(d)
    vals = d[stats_mask]
    if vals.numel() < 8:
        vals = d[torch.isfinite(d)]
    if vals.numel() < 2:
        return make_fallback_depth(d.shape[0], d.shape[1], z_near, z_far, d.device)

    lo_p = float(cfg["frontend"].get("depth_percentile_low", 2.0)) / 100.0
    hi_p = float(cfg["frontend"].get("depth_percentile_high", 98.0)) / 100.0
    lo = torch.quantile(vals, lo_p)
    hi = torch.quantile(vals, hi_p)
    if not torch.isfinite(lo) or not torch.isfinite(hi) or float((hi - lo).abs()) < 1e-6:
        return make_fallback_depth(d.shape[0], d.shape[1], z_near, z_far, d.device)
    x = d.clamp(lo, hi)
    x = (x - lo) / (hi - lo).clamp_min(1e-6)
    out = float(z_near) + x * (float(z_far) - float(z_near))
    return ensure_valid_depth(out, z_near, z_far)

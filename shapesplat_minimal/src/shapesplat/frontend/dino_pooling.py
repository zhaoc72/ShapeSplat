from __future__ import annotations

import torch
import torch.nn.functional as F

from shapesplat.geometry.masks import erode_mask, mask_to_box


def _frontend_cfg(cfg: dict | None) -> dict:
    if cfg is None:
        return {}
    return cfg.get("frontend", cfg)


def _pooling_cfg(cfg: dict | None) -> dict:
    fcfg = _frontend_cfg(cfg)
    return fcfg.get("dino_pooling", {})


def bbox_pool_descriptor(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """小物体 fallback：在 mask bbox 范围内平均 feature。

    小 mask 的像素太少，直接 mask 平均容易被单个边界像素支配；bbox pooling 更稳但更粗。
    """
    box = mask_to_box(mask)
    x0, y0, x1, y1 = [int(v) for v in box.detach().cpu()]
    h, w = mask.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 < x0 or y1 < y0:
        return features.flatten(1).mean(dim=1)
    crop = features[:, y0 : y1 + 1, x0 : x1 + 1]
    return crop.flatten(1).mean(dim=1)


def make_pooling_weights(masks: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    """根据 masks 构造 descriptor pooling 权重。

    直接平均 mask 内 feature 容易被分割边界污染；这里可选腐蚀 mask，
    并用多次腐蚀的近似距离权重让内部像素权重更高。保持纯 torch，避免 scipy 依赖。
    """
    if masks.shape[0] == 0:
        return masks.float()
    pcfg = _pooling_cfg(cfg)
    weights = masks.float().clone()
    if pcfg.get("use_eroded_mask", False):
        radius = max(0, int(pcfg.get("erosion_kernel", 5)) // 2)
        eroded = torch.stack([erode_mask(m, radius) for m in weights], dim=0)
        # 腐蚀后如果为空，回退原 mask，避免小物体被完全抹掉。
        non_empty = eroded.flatten(1).sum(dim=1) > 0
        weights = torch.where(non_empty[:, None, None], eroded, weights)
    if pcfg.get("use_distance_weight", False):
        # 轻量距离近似：原 mask + 多级腐蚀，越靠内部累计权重越大。
        acc = weights.clone()
        for r in (1, 2, 3):
            er = torch.stack([erode_mask(m, r) for m in masks.float()], dim=0)
            acc = acc + er * (1.0 / (r + 1))
        weights = acc * masks.float()
    return weights.float()


def pool_mask_descriptors(features: torch.Tensor, masks: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    """对 [D,H,W] dense features 做 mask-guided weighted pooling，输出 [N,D]。"""
    if masks.shape[0] == 0:
        return torch.zeros((0, features.shape[0]), device=features.device)
    fcfg = _frontend_cfg(cfg)
    pcfg = _pooling_cfg(cfg)
    masks = masks.to(features.device).float()
    weights = make_pooling_weights(masks, cfg).to(features.device)
    h, w = masks.shape[-2:]
    descs = []
    for n in range(masks.shape[0]):
        area_ratio = float((masks[n] > 0.5).float().mean().detach().cpu())
        if (
            pcfg.get("fallback_to_bbox_crop_for_small_masks", False)
            and area_ratio < float(pcfg.get("small_mask_area_ratio", 0.005))
        ):
            desc = bbox_pool_descriptor(features, masks[n])
        else:
            denom = weights[n].sum().clamp_min(1e-6)
            desc = (features * weights[n][None]).flatten(1).sum(dim=1) / denom
        descs.append(desc)
    desc = torch.stack(descs, dim=0)
    if fcfg.get("dino_l2_normalize", True):
        desc = F.normalize(desc, dim=-1)
    return desc

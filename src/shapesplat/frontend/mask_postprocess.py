from __future__ import annotations

import torch

from shapesplat.frontend.types import MaskSet
from shapesplat.geometry.masks import mask_to_box, stable_sort_masks


def _empty_like(masks: torch.Tensor, confidences: torch.Tensor, boxes: torch.Tensor) -> MaskSet:
    return MaskSet(masks, confidences, boxes)


def filter_masks_by_area(
    masks: torch.Tensor,
    confidences: torch.Tensor,
    boxes: torch.Tensor,
    min_area_ratio: float,
    max_area_ratio: float | None = None,
) -> MaskSet:
    """按面积过滤 retained visible masks，去掉碎片和异常大区域。"""
    if masks.shape[0] == 0:
        return _empty_like(masks, confidences, boxes)
    h, w = masks.shape[-2:]
    area = masks.flatten(1).sum(dim=1)
    keep = area >= float(min_area_ratio) * h * w
    if max_area_ratio is not None:
        keep = keep & (area <= float(max_area_ratio) * h * w)
    return MaskSet(masks[keep].float(), confidences[keep].float(), boxes[keep].float())


def filter_masks_by_score(masks: torch.Tensor, confidences: torch.Tensor, boxes: torch.Tensor, threshold: float) -> MaskSet:
    """按 score/confidence threshold 过滤 masks。"""
    if masks.shape[0] == 0:
        return _empty_like(masks, confidences, boxes)
    keep = confidences.float() >= float(threshold)
    return MaskSet(masks[keep].float(), confidences[keep].float(), boxes[keep].float())


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = ((a > 0.5) & (b > 0.5)).float().sum()
    union = ((a > 0.5) | (b > 0.5)).float().sum().clamp_min(1.0)
    return inter / union


def remove_duplicate_masks(masks: torch.Tensor, confidences: torch.Tensor, boxes: torch.Tensor, iou_threshold: float) -> MaskSet:
    """去除高度重复的 masks，优先保留 confidence 高的结果。"""
    if masks.shape[0] <= 1:
        return _empty_like(masks.float(), confidences.float(), boxes.float())
    order = torch.argsort(confidences.float(), descending=True)
    kept = []
    for idx in order.tolist():
        m = masks[idx]
        duplicate = any(float(_mask_iou(m, masks[j])) > iou_threshold for j in kept)
        if not duplicate:
            kept.append(idx)
    keep = torch.tensor(kept, device=masks.device, dtype=torch.long)
    return MaskSet(masks[keep].float(), confidences[keep].float(), boxes[keep].float())


def limit_num_masks(masks: torch.Tensor, confidences: torch.Tensor, boxes: torch.Tensor, max_num: int) -> MaskSet:
    """限制 mask 数量，先按 confidence 保留 top-K，再交给 stable sort 固定 object ID。"""
    if masks.shape[0] == 0 or max_num is None or max_num <= 0:
        return _empty_like(masks.float(), confidences.float(), boxes.float())
    if masks.shape[0] <= max_num:
        return _empty_like(masks.float(), confidences.float(), boxes.float())
    order = torch.argsort(confidences.float(), descending=True)[: int(max_num)]
    return MaskSet(masks[order].float(), confidences[order].float(), boxes[order].float())


def _fallback_center_mask(height: int, width: int, device: torch.device) -> MaskSet:
    """过滤后无 mask 时生成中心 fallback，避免 minimal pipeline 静默断掉。"""
    print("Warning: mask postprocess produced no masks; using a center fallback mask.")
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    ry, rx = height * 0.22, width * 0.22
    mask = ((((yy - cy) / ry).square() + ((xx - cx) / rx).square()) <= 1.0).float()
    masks = mask[None].float()
    conf = torch.tensor([0.5], device=device, dtype=torch.float32)
    boxes = torch.stack([mask_to_box(mask)], dim=0).float()
    return MaskSet(masks, conf, boxes)


def postprocess_masks(mask_set: MaskSet, cfg: dict, image_shape: tuple[int, int]) -> MaskSet:
    """统一后处理 retained visible masks。

    这些规则服务于 visible mask construction：score/area 过滤、重复 mask 去除、
    数量限制和稳定排序。真实 SAM3 与 Sam3Stub 都通过这里输出统一 MaskSet。
    """
    fcfg = cfg["frontend"] if "frontend" in cfg else cfg
    h, w = image_shape
    ms = MaskSet(mask_set.masks.float(), mask_set.confidences.float(), mask_set.boxes.float())
    ms = filter_masks_by_score(ms.masks, ms.confidences, ms.boxes, float(fcfg.get("mask_conf_threshold", 0.0)))
    ms = filter_masks_by_area(ms.masks, ms.confidences, ms.boxes, float(fcfg.get("min_area_ratio", 0.0)))
    ms = remove_duplicate_masks(ms.masks, ms.confidences, ms.boxes, float(fcfg.get("duplicate_iou_threshold", 0.92)))
    max_num = int(fcfg.get("max_num_objects", fcfg.get("sam3_max_masks", 8)))
    ms = limit_num_masks(ms.masks, ms.confidences, ms.boxes, max_num)
    if ms.masks.shape[0] == 0:
        return _fallback_center_mask(h, w, mask_set.masks.device)
    boxes = torch.stack([mask_to_box(m) for m in ms.masks], dim=0).float()
    masks, conf, boxes = stable_sort_masks(ms.masks.float(), ms.confidences.float(), boxes)
    return MaskSet(masks.float(), conf.float(), boxes.float())

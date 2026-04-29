from __future__ import annotations

import torch
import torch.nn.functional as F


def union_mask(masks: torch.Tensor) -> torch.Tensor:
    """返回所有 instance visible masks 的并集 [H,W]。"""
    if masks.numel() == 0:
        return torch.zeros(0, 0, device=masks.device)
    return (masks.float().amax(dim=0) > 0.5).float()


def mask_to_box(mask: torch.Tensor) -> torch.Tensor:
    """从单个 [H,W] mask 计算 xyxy bbox。"""
    ys, xs = torch.where(mask > 0.5)
    if xs.numel() == 0:
        return torch.tensor([0, 0, 0, 0], device=mask.device, dtype=torch.float32)
    return torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], device=mask.device, dtype=torch.float32)


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """用 max-pooling 做二值膨胀，避免依赖 OpenCV/scipy。"""
    if radius <= 0:
        return mask.float()
    x = mask.float()[None, None]
    y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[0, 0]


def erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """用 1 - dilate(1-mask) 做二值腐蚀。"""
    if radius <= 0:
        return mask.float()
    return 1.0 - dilate_mask(1.0 - mask.float(), radius)


def stable_sort_masks(masks: torch.Tensor, confidences: torch.Tensor, boxes: torch.Tensor):
    """按 bbox 中心 y、再中心 x 稳定排序。

    object ID 必须稳定，否则 mask、descriptor 和 Gaussian buffer 的对应关系会混乱；
    真实 SAM3 输出顺序可能随 prompt/阈值变化，因此这里显式固定排序规则。
    """
    if masks.shape[0] == 0:
        return masks, confidences, boxes
    centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
    centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
    order = sorted(range(masks.shape[0]), key=lambda i: (float(centers_y[i]), float(centers_x[i])))
    idx = torch.tensor(order, device=masks.device, dtype=torch.long)
    return masks[idx], confidences[idx], boxes[idx]

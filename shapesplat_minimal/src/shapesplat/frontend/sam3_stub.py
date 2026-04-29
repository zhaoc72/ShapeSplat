from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from shapesplat.geometry.masks import mask_to_box, stable_sort_masks


@dataclass
class MaskSet:
    """SAM3 stub 的输出结构。

    masks 是 retained visible instance masks，不是 amodal masks；hidden 部分由后续 shape prior 弱补全。
    """

    masks: torch.Tensor
    confidences: torch.Tensor
    boxes: torch.Tensor


class Sam3Stub:
    """最小版本占位：用颜色阈值和 connected components 模拟 SAM3。

    后续替换真实模型时，实现 RealSAM3Wrapper.predict_masks(image) 并返回同样的 MaskSet 即可。
    """

    def __init__(self, max_num_objects: int = 4, min_area_ratio: float = 0.002, conf_threshold: float = 0.2):
        self.max_num_objects = max_num_objects
        self.min_area_ratio = min_area_ratio
        self.conf_threshold = conf_threshold

    def _connected_components(self, fg: torch.Tensor) -> List[torch.Tensor]:
        h, w = fg.shape
        visited = torch.zeros_like(fg, dtype=torch.bool)
        comps: List[torch.Tensor] = []
        ys, xs = torch.where(fg > 0.5)
        coords: List[Tuple[int, int]] = [(int(y), int(x)) for y, x in zip(ys, xs)]
        for sy, sx in coords:
            if visited[sy, sx]:
                continue
            stack = [(sy, sx)]
            visited[sy, sx] = True
            pix = []
            while stack:
                y, x = stack.pop()
                pix.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and (not visited[ny, nx]) and fg[ny, nx] > 0.5:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            m = torch.zeros_like(fg)
            if pix:
                yy = torch.tensor([p[0] for p in pix], device=fg.device)
                xx = torch.tensor([p[1] for p in pix], device=fg.device)
                m[yy, xx] = 1.0
                comps.append(m)
        return comps

    def predict_masks(self, image: torch.Tensor) -> MaskSet:
        """从 [3,H,W] 图像预测 visible instance masks。"""
        _, h, w = image.shape
        # synthetic 背景接近白色；真实 SAM3 应使用 prompt / automatic mask generator。
        fg = (image.mean(dim=0) < 0.90).float()
        min_area = self.min_area_ratio * h * w
        comps = [m for m in self._connected_components(fg) if float(m.sum()) >= min_area]
        comps = sorted(comps, key=lambda m: float(m.sum()), reverse=True)[: self.max_num_objects]
        if not comps:
            masks = torch.zeros((0, h, w), device=image.device)
            conf = torch.zeros((0,), device=image.device)
            boxes = torch.zeros((0, 4), device=image.device)
            return MaskSet(masks, conf, boxes)
        masks = torch.stack(comps, dim=0).float()
        confidences = torch.full((masks.shape[0],), 0.9, device=image.device)
        boxes = torch.stack([mask_to_box(m) for m in masks], dim=0)
        keep = confidences >= self.conf_threshold
        masks, confidences, boxes = masks[keep], confidences[keep], boxes[keep]
        masks, confidences, boxes = stable_sort_masks(masks, confidences, boxes)
        return MaskSet(masks, confidences, boxes)

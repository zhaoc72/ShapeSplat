from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from shapesplat.frontend.mask_postprocess import postprocess_masks
from shapesplat.frontend.types import MaskSet
from shapesplat.geometry.masks import mask_to_box


class Sam3Stub:
    """最小版本占位：用前景启发式和 connected components 模拟 SAM3。

    这不是语义分割模型，只是为了让 synthetic 和简单真实 RGB 图像都能走通 pipeline。
    后续替换真实 SAM3 时，保持 predict_masks(image)->MaskSet 的输入输出协议即可。
    """

    def __init__(self, max_num_objects: int = 4, min_area_ratio: float = 0.002, conf_threshold: float = 0.2):
        self.max_num_objects = max_num_objects
        self.min_area_ratio = min_area_ratio
        self.conf_threshold = conf_threshold

    def _foreground_heuristic(self, image: torch.Tensor) -> torch.Tensor:
        """结合饱和度、边界背景颜色距离和亮背景距离估计前景区域。"""
        _, h, w = image.shape
        rgb = image.clamp(0, 1)
        maxc = rgb.max(dim=0).values
        minc = rgb.min(dim=0).values
        saturation = maxc - minc

        border = torch.cat([rgb[:, 0, :], rgb[:, -1, :], rgb[:, :, 0], rgb[:, :, -1]], dim=1)
        bg_color = border.median(dim=1).values.view(3, 1, 1)
        dist_border_bg = torch.linalg.vector_norm(rgb - bg_color, dim=0)
        dist_light_bg = torch.linalg.vector_norm(rgb - 0.96, dim=0)
        gray = rgb.mean(dim=0)

        # 这些阈值刻意偏宽松：stub 更关注“有可用 mask”，不追求真实 SAM3 质量。
        fg = (saturation > 0.10) | (dist_border_bg > 0.16) | ((gray < 0.86) & (dist_light_bg > 0.12))
        fg = fg.float()
        # 简单形态学闭运算，减少真实图片边缘上的小孔洞和断裂。
        fg = F.max_pool2d(fg[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        fg = 1.0 - F.max_pool2d((1.0 - fg)[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        return fg[:h, :w]

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
            if pix:
                m = torch.zeros_like(fg)
                yy = torch.tensor([p[0] for p in pix], device=fg.device)
                xx = torch.tensor([p[1] for p in pix], device=fg.device)
                m[yy, xx] = 1.0
                comps.append(m)
        return comps

    def _fallback_center_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """真实图片启发式失败时的中心 fallback mask，避免 pipeline 静默断掉。"""
        print("Warning: Sam3Stub did not find connected components; using a center fallback mask.")
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        ry, rx = h * 0.22, w * 0.22
        return (((yy - cy) / ry).square() + ((xx - cx) / rx).square() <= 1.0).float()

    def predict_masks(self, image: torch.Tensor) -> MaskSet:
        """从 [3,H,W] 图像预测 visible instance masks。"""
        _, h, w = image.shape
        fg = self._foreground_heuristic(image)
        min_area = self.min_area_ratio * h * w
        comps = [m for m in self._connected_components(fg) if float(m.sum()) >= min_area]
        if not comps:
            comps = [self._fallback_center_mask(h, w, image.device)]

        # 先按面积保留 top-K；随后统一后处理会按位置稳定排序，避免 object ID 漂移。
        comps = sorted(comps, key=lambda m: float(m.sum()), reverse=True)[: self.max_num_objects]
        masks = torch.stack(comps, dim=0).float()
        confidences = torch.full((masks.shape[0],), 0.9, device=image.device)
        boxes = torch.stack([mask_to_box(m) for m in masks], dim=0)
        cfg = {
            "frontend": {
                "max_num_objects": self.max_num_objects,
                "min_area_ratio": self.min_area_ratio,
                "mask_conf_threshold": self.conf_threshold,
            }
        }
        return postprocess_masks(MaskSet(masks.float(), confidences.float(), boxes.float()), cfg, (h, w))

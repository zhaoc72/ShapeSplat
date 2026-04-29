from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MaskSet:
    """统一的 SAM backend 输出结构。

    masks: [N,H,W] float32，retained visible instance masks，不是 amodal masks。
    confidences: [N] float32，每个 mask 的置信度或 stub score。
    boxes: [N,4] float32，xyxy 像素坐标。
    """

    masks: torch.Tensor
    confidences: torch.Tensor
    boxes: torch.Tensor

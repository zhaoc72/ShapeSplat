from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ShapeAsset:
    """统一的 shape bank 资产结构。

    shape_id: 形状唯一 ID，通常来自文件名或 toy 名称。
    points: [P,3] normalized point cloud，用于 hidden soft support。
    descriptor: [D] 全局 descriptor，可用于 cosine retrieval。
    descriptors: [V,D] 可选多视角 descriptor，用于 max-over-view retrieval。
    category: 可选类别名。
    metadata: 文件路径、原始点数等附加信息。
    """

    shape_id: str
    points: torch.Tensor
    descriptor: Optional[torch.Tensor] = None
    descriptors: Optional[torch.Tensor] = None
    category: Optional[str] = None
    metadata: dict = field(default_factory=dict)

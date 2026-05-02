from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RenderOutput:
    """统一 renderer 输出契约。

    字段约定：
    - rgb: [3,H,W]，最终合成 RGB，用于 RGB reconstruction loss 和可视化。
    - alpha: [H,W]，前景不透明度，用于 foreground/background 约束。
    - depth: [H,W]，渲染深度，用于 weak depth consistency。
    - contributions: [N,H,W]，每个 object 的 visibility/transmittance-like contribution。
    - ownership: [N,H,W]，object contribution 归一化后的 ownership map。
    - bg_ownership: [H,W]，背景 ownership。
    - extras: 预留给真实 CUDA renderer 的额外调试信息。

    contributions / ownership 是 ShapeSplat++ object-centric reconstruction 的核心输出。
    后续真实 3DGS renderer 不能只返回 RGB，必须保留 per-object contribution maps。
    """

    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    contributions: torch.Tensor
    ownership: torch.Tensor
    bg_ownership: torch.Tensor
    extras: dict = field(default_factory=dict)

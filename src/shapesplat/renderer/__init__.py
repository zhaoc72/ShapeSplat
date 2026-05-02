"""Renderer backend 接口。

默认使用 SoftGaussianRenderer；后续真实 CUDA 3DGS renderer 应通过 backend factory 接入。
"""

from .backend import build_renderer
from .soft_renderer import SoftGaussianRenderer
from .types import RenderOutput

__all__ = ["build_renderer", "SoftGaussianRenderer", "RenderOutput"]

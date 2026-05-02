from __future__ import annotations

from shapesplat.renderer.types import RenderOutput


class RealRendererPlaceholder:
    """真实 CUDA 3DGS renderer 的占位类。

    这个类不会做任何真实渲染，只用于明确接口边界，避免误以为 minimal 版本已经
    集成 CUDA renderer。后续真实实现应替换为自己的 class，并在配置里填写
    real_renderer_module / real_renderer_class。
    """

    def __init__(self, camera, cfg: dict):
        self.camera = camera
        self.cfg = cfg

    def __call__(self, scene) -> RenderOutput:
        raise NotImplementedError(
            "Real CUDA 3DGS renderer is not implemented in ShapeSplat++ minimal. "
            "A real renderer must return RenderOutput with rgb [3,H,W], alpha [H,W], "
            "depth [H,W], contributions [N,H,W], ownership [N,H,W], and bg_ownership [H,W]. "
            "Per-object contributions/ownership are required by scene-coupled ownership optimization."
        )

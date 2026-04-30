from __future__ import annotations

import importlib

from shapesplat.renderer.soft_renderer import SoftGaussianRenderer


def build_real_renderer(camera, cfg: dict):
    """动态构建真实 renderer。

    这是后续接入 CUDA 3D Gaussian Splatting renderer 的唯一入口。真实 renderer
    class 应接受 `(camera, cfg)`，并在调用时返回 `RenderOutput`，尤其需要
    `contributions [N,H,W]` 和 `ownership [N,H,W]`。
    """
    rcfg = cfg.get("renderer", {})
    module_name = rcfg.get("real_renderer_module")
    class_name = rcfg.get("real_renderer_class")
    if not module_name or not class_name:
        raise ValueError(
            "renderer.backend=real requires renderer.real_renderer_module and "
            "renderer.real_renderer_class. CUDA 3DGS renderer is not bundled in minimal version."
        )
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(camera, cfg)


def build_renderer(camera, cfg: dict):
    """构建 renderer backend。

    默认 soft backend 使用当前 PyTorch differentiable approximation。real backend
    只做动态导入，不强制引入任何 CUDA 3DGS 依赖；auto backend 失败后可 fallback soft。
    """
    rcfg = cfg.get("renderer", {})
    backend = str(rcfg.get("backend", "soft")).lower()

    if backend == "soft":
        return SoftGaussianRenderer(camera, cfg)

    if backend == "real":
        try:
            return build_real_renderer(camera, cfg)
        except Exception as exc:
            raise RuntimeError(f"Failed to build real renderer backend: {exc}") from exc

    if backend == "auto":
        try:
            return build_real_renderer(camera, cfg)
        except Exception as exc:
            if rcfg.get("fallback_to_soft", True):
                print(f"[Renderer warning] real renderer unavailable ({exc}); fallback to SoftGaussianRenderer.")
                return SoftGaussianRenderer(camera, cfg)
            raise RuntimeError(f"Failed to build real renderer backend and fallback_to_soft=false: {exc}") from exc

    raise ValueError(f"Unknown renderer.backend={backend!r}; expected soft/real/auto.")

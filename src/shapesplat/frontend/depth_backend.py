from __future__ import annotations

from shapesplat.frontend.depth_stub import DepthStub


def build_depth_backend(cfg: dict) -> object:
    """构建 depth backend。

    使用 factory 可以让默认 stub 模式完全不受真实 depth 依赖影响；
    pipeline 只依赖 predict_depth(image)->[H,W] 统一接口。
    """
    backend = cfg["frontend"].get("depth_backend", "stub").lower()
    if backend == "stub":
        return DepthStub(cfg)
    if backend in ("real", "auto"):
        try:
            from shapesplat.frontend.depth_real import RealDepthWrapper

            return RealDepthWrapper(cfg)
        except Exception as exc:
            if backend == "auto":
                print(f"Warning: RealDepthWrapper unavailable ({exc}); falling back to DepthStub.")
                return DepthStub(cfg)
            raise RuntimeError(f"无法构建 real depth backend: {exc}") from exc
    raise ValueError(f"未知 frontend.depth_backend: {backend}. 可选: stub / real / auto")

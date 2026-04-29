from __future__ import annotations

from shapesplat.frontend.dinov3_stub import DinoV3Stub


def build_dino_backend(cfg: dict) -> object:
    """构建 DINO backend。

    使用 factory 可以让默认 stub 模式完全不受真实 DINOv3 依赖影响；
    pipeline 只依赖 extract_dense_features / pool_descriptors 统一接口。
    """
    backend = cfg["frontend"].get("dino_backend", "stub").lower()
    if backend == "stub":
        return DinoV3Stub(cfg)
    if backend in ("real", "auto"):
        try:
            from shapesplat.frontend.dinov3_real import RealDINOv3Wrapper

            return RealDINOv3Wrapper(cfg)
        except Exception as exc:
            if backend == "auto":
                print(f"Warning: RealDINOv3Wrapper unavailable ({exc}); falling back to DinoV3Stub.")
                return DinoV3Stub(cfg)
            raise RuntimeError(f"无法构建 real DINOv3 backend: {exc}") from exc
    raise ValueError(f"未知 frontend.dino_backend: {backend}. 可选: stub / real / auto")

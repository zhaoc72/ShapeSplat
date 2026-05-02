from __future__ import annotations

from shapesplat.frontend.sam3_stub import Sam3Stub


def _build_stub(cfg: dict) -> Sam3Stub:
    fcfg = cfg["frontend"]
    return Sam3Stub(
        max_num_objects=fcfg["max_num_objects"],
        min_area_ratio=fcfg["min_area_ratio"],
        conf_threshold=fcfg["mask_conf_threshold"],
    )


def build_sam_backend(cfg: dict) -> object:
    """构建 SAM backend。

    使用 factory 的原因是让默认 stub 模式完全不受真实 SAM3 依赖影响；
    pipeline 只依赖 predict_masks(image)->MaskSet 统一接口，不关心具体实现。
    """
    backend = cfg["frontend"].get("sam_backend", "stub").lower()
    if backend == "stub":
        return _build_stub(cfg)
    if backend in ("real", "auto"):
        try:
            from shapesplat.frontend.sam3_real import RealSAM3Wrapper

            return RealSAM3Wrapper(cfg)
        except Exception as exc:
            if backend == "auto":
                print(f"Warning: RealSAM3Wrapper unavailable ({exc}); falling back to Sam3Stub.")
                return _build_stub(cfg)
            raise RuntimeError(f"无法构建 real SAM3 backend: {exc}") from exc
    raise ValueError(f"未知 frontend.sam_backend: {backend}. 可选: stub / real / auto")

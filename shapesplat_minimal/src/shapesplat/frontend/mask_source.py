from __future__ import annotations

from pathlib import Path

import torch

from shapesplat.frontend.file_mask_loader import load_mask_file
from shapesplat.frontend.sam_backend import build_sam_backend
from shapesplat.frontend.types import MaskSet


def _record_mask_path(record) -> str | None:
    if record is None:
        return None
    meta = getattr(record, "metadata", {}) or {}
    return meta.get("mask_path")


def _resolve_mask_path(cfg: dict, record=None) -> Path | None:
    """解析 file mask 路径。

    mask_source 控制 masks 从哪里来；sam_backend 只在 mask_source=sam 时生效。
    dataset record.metadata 可以携带 mask_path，用于 batch same-mask setting。
    """
    fcfg = cfg.get("frontend", {})
    raw = _record_mask_path(record) or fcfg.get("mask_path")
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    mask_dir = fcfg.get("mask_dir")
    if mask_dir:
        return Path(mask_dir) / path
    return path


def get_masks_for_image(image: torch.Tensor, cfg: dict, record=None) -> MaskSet:
    """根据 cfg.frontend.mask_source 获取当前图像的 visible masks。

    - sam: 使用现有 SAM backend。
    - file: 必须提供 mask_path。
    - auto: 有可用 mask_path 时用 file，否则 fallback 到 SAM。
    """
    source = str(cfg.get("frontend", {}).get("mask_source", "sam")).lower()
    mask_path = _resolve_mask_path(cfg, record)
    image_hw = tuple(image.shape[-2:])

    if source == "sam":
        return build_sam_backend(cfg).predict_masks(image)
    if source == "file":
        if mask_path is None:
            raise FileNotFoundError("frontend.mask_source=file but no mask_path was provided.")
        return load_mask_file(mask_path, image_hw, cfg)
    if source == "auto":
        if mask_path is not None and mask_path.exists():
            return load_mask_file(mask_path, image_hw, cfg)
        print("Warning: mask_source=auto but no valid mask file was found; fallback to SAM backend.")
        return build_sam_backend(cfg).predict_masks(image)
    raise ValueError(f"Unknown frontend.mask_source={source!r}; expected sam/file/auto.")

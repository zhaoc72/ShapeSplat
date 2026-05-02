from __future__ import annotations

import torch

from .file_shape_bank import FileShapeBank
from .toy_shape_bank import ToyShapeBank


def build_shape_bank(cfg: dict, descriptor_dim: int, device: torch.device):
    """构建 shape bank backend。

    backend factory 让主 pipeline 只依赖统一的 `.assets` 接口。默认 toy backend
    不需要任何外部数据；file/auto backend 用于接入本地真实点云 shape bank。
    """
    bank_cfg = cfg.get("shape_bank", {})
    backend = str(bank_cfg.get("backend", "toy")).lower()
    points_per_shape = int(bank_cfg.get("num_points") or 512)

    if backend == "toy":
        return ToyShapeBank(descriptor_dim=descriptor_dim, device=device, points_per_shape=points_per_shape)

    if backend == "file":
        return FileShapeBank(cfg=cfg, descriptor_dim=descriptor_dim, device=device)

    if backend == "auto":
        try:
            return FileShapeBank(cfg=cfg, descriptor_dim=descriptor_dim, device=device)
        except Exception as exc:
            if bank_cfg.get("fallback_to_toy", True):
                print(f"[ShapeBank warning] file backend failed ({exc}); fallback to ToyShapeBank.")
                return ToyShapeBank(descriptor_dim=descriptor_dim, device=device, points_per_shape=points_per_shape)
            raise

    raise ValueError(f"Unknown shape_bank.backend={backend!r}; expected toy/file/auto.")

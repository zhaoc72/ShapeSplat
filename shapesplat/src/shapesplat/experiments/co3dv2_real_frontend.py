from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def check_checkpoint_path(path: str | Path | None, allow_missing: bool = False) -> dict:
    """检查本地权重路径；测试可以用 allow_missing=True 避免依赖真实文件。"""
    if path is None:
        report = {"path": None, "exists": False, "status": "missing", "error": "checkpoint path is null"}
    else:
        p = Path(path)
        report = {"path": str(p), "exists": p.exists(), "status": "ok" if p.exists() else "missing", "error": None if p.exists() else f"checkpoint not found: {p}"}
    if report["status"] == "missing" and not allow_missing:
        raise FileNotFoundError(report["error"])
    return report


def apply_dinov3_cli_overrides(cfg: dict, checkpoint: str | None = None, model_name: str | None = None, device: str | None = None) -> dict:
    """应用 DINOv3 检查/缓存脚本的 CLI 覆盖，不触发真实模型加载。"""
    fcfg = cfg.setdefault("frontend", {})
    if checkpoint:
        fcfg["dino_checkpoint"] = checkpoint
    if model_name:
        fcfg["dino_model_name"] = model_name
    if device:
        fcfg["dino_device"] = device
        cfg.setdefault("runtime", {})["device"] = device
        cfg["device"] = device
    return cfg


def descriptor_stats(desc: torch.Tensor) -> dict:
    """生成 descriptor 统计，便于真实 DINOv3 cache readiness 报告。"""
    d = desc.detach().float().cpu()
    norms = torch.linalg.norm(d, dim=-1) if d.ndim >= 2 else torch.linalg.norm(d.reshape(1, -1), dim=-1)
    return {
        "shape": list(d.shape),
        "norm_min": float(norms.min()) if norms.numel() else 0.0,
        "norm_max": float(norms.max()) if norms.numel() else 0.0,
        "norm_mean": float(norms.mean()) if norms.numel() else 0.0,
        "finite": bool(torch.isfinite(d).all()),
    }


def best_iou_and_coverage(co3d_masks, sam_masks) -> dict:
    """比较 SAM3 automatic masks 与 CO3Dv2 file mask。

    CO3Dv2 主实验仍使用 file masks；这里的 IoU/coverage 只用于 automatic-mask diagnostic。
    """
    co3d = torch.as_tensor(co3d_masks).float()
    sam = torch.as_tensor(sam_masks).float()
    if co3d.ndim == 2:
        co3d = co3d[None]
    if sam.ndim == 2:
        sam = sam[None]
    if sam.shape[0] == 0 or co3d.shape[0] == 0:
        return {"best_iou": 0.0, "coverage": 0.0, "num_sam_masks": int(sam.shape[0]), "co3d_area": float((co3d > 0.5).sum()), "sam_area": 0.0}
    target = (co3d[0] > 0.5)
    ious = []
    for m in sam:
        pred = m > 0.5
        inter = (target & pred).sum().float()
        union = (target | pred).sum().float().clamp_min(1.0)
        ious.append(float((inter / union).item()))
    sam_union = (sam > 0.5).any(dim=0)
    coverage = float(((target & sam_union).sum().float() / target.sum().float().clamp_min(1.0)).item())
    return {
        "best_iou": max(ious),
        "coverage": coverage,
        "num_sam_masks": int(sam.shape[0]),
        "co3d_area": float(target.sum().item()),
        "sam_area": float(sam_union.sum().item()),
        "over_segmentation_warning": bool(sam.shape[0] > 3),
        "under_segmentation_warning": bool(coverage < 0.5),
    }


def save_missing_report(out_dir: str | Path, name: str, report: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def make_center_mask(height: int, width: int, device: torch.device | None = None) -> torch.Tensor:
    """为 DINOv3 权重检查构造中心矩形 mask，不依赖真实 CO3D 数据。"""
    mask = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    y0, y1 = height // 4, height * 3 // 4
    x0, x1 = width // 4, width * 3 // 4
    mask[:, y0:y1, x0:x1] = 1.0
    return mask


def save_feature_norm_image(features: torch.Tensor, path: str | Path) -> None:
    """保存 dense feature norm 可视化，快速确认 DINOv3 输出不是空图。"""
    from PIL import Image

    norm = torch.linalg.norm(features.detach().float().cpu(), dim=0)
    norm = (norm - norm.min()) / (norm.max() - norm.min()).clamp_min(1e-6)
    arr = (norm.numpy() * 255).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)

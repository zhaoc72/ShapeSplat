from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def chamfer_l2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """轻量 Chamfer L2。

    geometry metrics 是 optional；正式论文中还需要严格定义尺度、坐标系和 alignment protocol。
    """
    pred = pred.float()
    gt = gt.float()
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] != 3 or gt.shape[1] != 3:
        raise ValueError("pred/gt must be [P,3] and [Q,3]")
    d = torch.cdist(pred, gt, p=2) ** 2
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def fscore(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """基于最近邻距离的 point-cloud F-score。"""
    d = torch.cdist(pred.float(), gt.float(), p=2)
    pred_nn = d.min(dim=1).values
    gt_nn = d.min(dim=0).values
    precision = (pred_nn < threshold).float().mean()
    recall = (gt_nn < threshold).float().mean()
    return 2 * precision * recall / (precision + recall).clamp_min(1e-8)


def load_pointcloud(path: str | Path) -> torch.Tensor:
    """读取可选点云 GT/预测；没有 open3d/trimesh 依赖。"""
    p = Path(path)
    if p.suffix.lower() == ".npz":
        data = np.load(p, allow_pickle=True)
        key = "points" if "points" in data else "pointcloud"
        if key not in data:
            raise ValueError(f"npz pointcloud missing points/pointcloud: {p}")
        arr = data[key]
    else:
        arr = np.load(p, allow_pickle=False)
    pts = torch.as_tensor(arr, dtype=torch.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pointcloud must be [P,3], got {tuple(pts.shape)}")
    return pts


def compute_geometry_metrics(pred_path, gt_path, threshold: float = 0.01) -> dict:
    """在点云文件存在时计算轻量几何指标，否则返回 unavailable。

    optional_geometry 不属于默认 minimal/stub 实验的必需指标。
    """
    pred_p = Path(pred_path) if pred_path else None
    gt_p = Path(gt_path) if gt_path else None
    if pred_p is None or gt_p is None or not pred_p.exists() or not gt_p.exists():
        return {"available": False, "reason": "pred_path or gt_path does not exist"}
    pred = load_pointcloud(pred_p)
    gt = load_pointcloud(gt_p)
    return {"available": True, "ChamferL2": float(chamfer_l2(pred, gt)), "FScore": float(fscore(pred, gt, threshold))}

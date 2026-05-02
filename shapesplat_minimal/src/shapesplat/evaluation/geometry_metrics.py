from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from shapesplat.evaluation.alignment import apply_alignment


def _check_points(points: torch.Tensor, name: str) -> torch.Tensor:
    pts = torch.as_tensor(points, dtype=torch.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError(f"{name} pointcloud must be [P,3], got {tuple(pts.shape)}")
    if not bool(torch.isfinite(pts).all()):
        raise ValueError(f"{name} pointcloud contains NaN or Inf")
    return pts


def _min_distances(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    vals = []
    for start in range(0, a.shape[0], int(chunk_size)):
        chunk = a[start : start + int(chunk_size)]
        vals.append(torch.cdist(chunk, b, p=2).min(dim=1).values)
    return torch.cat(vals, dim=0)


def load_pointcloud(path: str | Path) -> torch.Tensor:
    """读取 .npy/.npz/.pt 点云，不依赖 open3d/trimesh。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"pointcloud not found: {p.resolve()}")
    if p.suffix.lower() == ".pt":
        data = torch.load(p, map_location="cpu")
        if isinstance(data, dict):
            key = next((k for k in ("points", "pointcloud", "xyz") if k in data), None)
            if key is None:
                raise ValueError(f"pt pointcloud missing points/pointcloud/xyz: {p}")
            data = data[key]
    elif p.suffix.lower() == ".npz":
        npz = np.load(p, allow_pickle=True)
        key = next((k for k in ("points", "pointcloud", "xyz") if k in npz), None)
        if key is None:
            raise ValueError(f"npz pointcloud missing points/pointcloud/xyz: {p}")
        data = npz[key]
    else:
        data = np.load(p, allow_pickle=False)
    return _check_points(torch.as_tensor(data, dtype=torch.float32), "pointcloud")


def save_pointcloud(points, path: str | Path) -> None:
    """保存 .npy 点云。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, _check_points(torch.as_tensor(points), "points").detach().cpu().numpy().astype("float32"))


def sample_pointcloud(points: torch.Tensor, num_points: int, seed: int = 123) -> torch.Tensor:
    """点数太多时 deterministic 下采样；点数不足则原样返回。"""
    pts = _check_points(points, "points")
    if pts.shape[0] <= int(num_points):
        return pts
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx = torch.randperm(pts.shape[0], generator=g)[: int(num_points)]
    return pts.cpu()[idx].to(pts.device)


def normalize_pointcloud_for_metric(points: torch.Tensor, mode: str = "none") -> torch.Tensor:
    """单点云归一化工具；正式实验应在 protocol 中固定该选择。"""
    pts = _check_points(points, "points")
    if mode in (None, "none"):
        return pts
    centered = pts - pts.mean(dim=0, keepdim=True)
    if mode == "center":
        return centered
    if mode in {"unit_bbox", "center_unit_bbox"}:
        extent = (centered.max(dim=0).values - centered.min(dim=0).values).max().clamp_min(1e-8)
        return centered / extent
    raise ValueError(f"unknown pointcloud normalize mode: {mode}")


def chamfer_l2(pred: torch.Tensor, gt: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """轻量 Chamfer L2；点数较大时分块 cdist，避免一次性占太多内存。"""
    pred = _check_points(pred.float(), "pred")
    gt = _check_points(gt.float(), "gt")
    pred_nn = _min_distances(pred, gt, chunk_size).square()
    gt_nn = _min_distances(gt, pred, chunk_size).square()
    return pred_nn.mean() + gt_nn.mean()


def fscore(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.01, chunk_size: int = 4096) -> torch.Tensor:
    """基于最近邻距离的 F-score。"""
    pred = _check_points(pred.float(), "pred")
    gt = _check_points(gt.float(), "gt")
    pred_nn = _min_distances(pred, gt, chunk_size)
    gt_nn = _min_distances(gt, pred, chunk_size)
    precision = (pred_nn < threshold).float().mean()
    recall = (gt_nn < threshold).float().mean()
    return 2 * precision * recall / (precision + recall).clamp_min(1e-8)


def compute_geometry_metrics(
    pred_points,
    gt_points,
    threshold: float = 0.01,
    normalize: str = "none",
    num_sample_points: int | None = None,
) -> dict:
    """计算 optional pointcloud geometry metrics。

    这是 lightweight evaluator；正式论文需要明确 alignment / scale protocol。
    real-image diagnostics 没有 GT 时不应报告 geometry。
    """
    if isinstance(pred_points, (str, Path)) or isinstance(gt_points, (str, Path)):
        return compute_geometry_metrics_from_paths(
            pred_points,
            gt_points,
            threshold=threshold,
            normalize=normalize,
            num_sample_points=num_sample_points,
        )
    pred = _check_points(torch.as_tensor(pred_points), "pred")
    gt = _check_points(torch.as_tensor(gt_points), "gt")
    if num_sample_points is not None:
        pred = sample_pointcloud(pred, int(num_sample_points), seed=123)
        gt = sample_pointcloud(gt, int(num_sample_points), seed=456)
    pred, gt = apply_alignment(pred, gt, normalize)
    pred_nn = _min_distances(pred, gt)
    gt_nn = _min_distances(gt, pred)
    precision = (pred_nn < threshold).float().mean()
    recall = (gt_nn < threshold).float().mean()
    fs = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "available": True,
        "ChamferL2": float(pred_nn.square().mean() + gt_nn.square().mean()),
        "FScore": float(fs),
        "Precision": float(precision),
        "Recall": float(recall),
        "NumPredPoints": int(pred.shape[0]),
        "NumGTPoints": int(gt.shape[0]),
    }


def compute_geometry_metrics_from_paths(
    pred_path,
    gt_path,
    threshold: float = 0.01,
    normalize: str = "none",
    num_sample_points: int | None = None,
) -> dict:
    """从路径计算 geometry；文件缺失时返回 available=false，不影响主评估。"""
    pred_p = Path(pred_path) if pred_path else None
    gt_p = Path(gt_path) if gt_path else None
    if pred_p is None or gt_p is None or not pred_p.exists() or not gt_p.exists():
        return {"available": False, "reason": "missing pred or gt"}
    return compute_geometry_metrics(
        load_pointcloud(pred_p),
        load_pointcloud(gt_p),
        threshold=threshold,
        normalize=normalize,
        num_sample_points=num_sample_points,
    )


def compute_geometry_metrics_legacy(pred_path, gt_path, threshold: float = 0.01) -> dict:
    return compute_geometry_metrics_from_paths(pred_path, gt_path, threshold=threshold)

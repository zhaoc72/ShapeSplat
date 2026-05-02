from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def point_statistics_descriptor(points: torch.Tensor, dim: int) -> torch.Tensor:
    """用点云统计量生成确定性 descriptor。

    这是 minimal 预计算方案；正式实验可替换成多视角渲染后的 DINO descriptor。
    """
    pts = points.detach().cpu().float()
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be [P,3], got {tuple(pts.shape)}")
    center = pts.mean(dim=0)
    std = pts.std(dim=0, unbiased=False)
    pmin = pts.amin(dim=0)
    pmax = pts.amax(dim=0)
    extent = pmax - pmin
    radius = torch.linalg.norm(pts - center, dim=1)
    stats = torch.cat(
        [
            center,
            std,
            pmin,
            pmax,
            extent,
            torch.tensor([radius.mean(), radius.std(unbiased=False), radius.min(), radius.max()]),
        ]
    )
    if stats.numel() < dim:
        repeats = (dim + stats.numel() - 1) // stats.numel()
        stats = stats.repeat(repeats)
    desc = stats[:dim].float()
    return F.normalize(desc, dim=0)


def make_random_descriptor(shape_id: str, dim: int, seed: int = 123) -> torch.Tensor:
    """根据 shape_id 生成稳定 random descriptor，保证不同机器测试一致。"""
    digest = hashlib.sha256(f"{seed}:{shape_id}".encode("utf-8")).hexdigest()
    local_seed = int(digest[:8], 16)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(local_seed)
    return F.normalize(torch.randn(dim, generator=gen), dim=0)


def _load_shape(path: Path) -> tuple[np.ndarray, str | None]:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        if "points" not in data:
            raise ValueError(f"{path} missing points")
        category = str(data["category"].item()) if "category" in data and data["category"].shape == () else None
        return data["points"], category
    return np.load(path, allow_pickle=False), None


def _normalize_points(points: torch.Tensor) -> torch.Tensor:
    pts = points.float().clone()
    pmin, pmax = pts.amin(dim=0), pts.amax(dim=0)
    pts = pts - (pmin + pmax) * 0.5
    scale = (pts.amax(dim=0) - pts.amin(dim=0)).amax().clamp_min(1e-6)
    return pts / scale


def precompute_shape_descriptors(
    shape_bank_root: str | Path,
    out_root: str | Path,
    descriptor_dim: int,
    mode: str = "point_stats",
    seed: int = 123,
) -> list[dict]:
    """为 .npy/.npz 点云写入 prepared descriptor bank。"""
    in_root = Path(shape_bank_root)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    if not in_root.exists():
        raise FileNotFoundError(f"shape bank input does not exist: {in_root}")
    rows: list[dict] = []
    for path in sorted([p for p in in_root.iterdir() if p.suffix.lower() in {".npy", ".npz"}]):
        points_np, category = _load_shape(path)
        points = torch.as_tensor(points_np, dtype=torch.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{path} points must be [P,3]")
        points = _normalize_points(points)
        if mode == "point_stats":
            descriptor = point_statistics_descriptor(points, descriptor_dim)
        elif mode == "random":
            descriptor = make_random_descriptor(path.stem, descriptor_dim, seed)
        else:
            raise ValueError(f"unknown descriptor mode: {mode}")
        out_path = out / f"{path.stem}.npz"
        np.savez(
            out_path,
            points=points.numpy().astype("float32"),
            descriptor=descriptor.numpy().astype("float32"),
            category=np.array(category or "prepared"),
        )
        rows.append(
            {
                "shape_id": path.stem,
                "input_path": str(path),
                "output_path": str(out_path),
                "num_points": int(points.shape[0]),
                "descriptor_dim": int(descriptor_dim),
                "mode": mode,
            }
        )
    return rows

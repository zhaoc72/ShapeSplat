from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .types import ShapeAsset


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value.tolist())
    return str(value)


class FileShapeBank:
    """从本地 .npz/.npy 点云目录读取 shape bank。

    这是 v0.8 的最小真实 shape bank 接口，只依赖 numpy/torch，不引入 open3d
    或 trimesh。正式论文实验时需要保证 shape bank 与测试实例 train/test
    instance-disjoint，避免检索泄漏；本类只负责读取和规范化本地点云资产。
    """

    def __init__(self, cfg: dict, descriptor_dim: int, device: torch.device):
        self.cfg = cfg
        self.bank_cfg = cfg.get("shape_bank", {})
        self.descriptor_dim = int(self.bank_cfg.get("descriptor_dim") or descriptor_dim)
        self.device = device
        self._assets: list[ShapeAsset] = []
        self._load()

    @property
    def assets(self) -> list[ShapeAsset]:
        return self._assets

    def __len__(self) -> int:
        return len(self._assets)

    def _load(self) -> None:
        root_value = self.bank_cfg.get("root")
        if not root_value:
            raise FileNotFoundError("shape_bank.root is null; file shape bank needs a local directory.")
        root = Path(root_value)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Shape bank root does not exist or is not a directory: {root}")

        exts = set(self.bank_cfg.get("file_extensions") or [".npz", ".npy"])
        files = [p for p in sorted(root.iterdir()) if p.suffix.lower() in exts]
        max_shapes = self.bank_cfg.get("max_shapes")
        if max_shapes is not None:
            files = files[: int(max_shapes)]

        for idx, path in enumerate(files):
            try:
                asset = self._load_one(path, idx)
            except Exception as exc:
                print(f"[FileShapeBank warning] skip {path}: {exc}")
                continue
            if asset is not None:
                self._assets.append(asset)

        if not self._assets:
            raise RuntimeError(f"No valid shape assets found in {root}")

    def _load_one(self, path: Path, index: int) -> ShapeAsset | None:
        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=True)
            if "points" not in data:
                raise ValueError("npz must contain points [P,3]")
            points_np = data["points"]
            descriptor_np = data["descriptor"] if "descriptor" in data else None
            descriptors_np = data["descriptors"] if "descriptors" in data else None
            category = _as_str(data["category"]) if "category" in data else None
        else:
            points_np = np.load(path, allow_pickle=False)
            descriptor_np = None
            descriptors_np = None
            category = None

        points = torch.as_tensor(points_np, dtype=torch.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be [P,3], got {tuple(points.shape)}")
        if points.shape[0] < 4:
            raise ValueError("point cloud has fewer than 4 points")
        if not bool(torch.isfinite(points).all()):
            raise ValueError("points contain NaN or Inf")

        original_points = int(points.shape[0])
        points = self._normalize_points(points)
        points = self._sample_points(points, index)

        descriptor = self._load_descriptor(descriptor_np, descriptors_np, index)
        descriptors = self._load_descriptors(descriptors_np) if descriptors_np is not None else None
        if descriptor is None and descriptors is not None:
            descriptor = F.normalize(descriptors.mean(dim=0), dim=0)
        if descriptor is None:
            descriptor = self._fallback_descriptor(index)

        metadata = {"path": str(path), "original_points": original_points}
        return ShapeAsset(
            shape_id=path.stem,
            points=points.to(self.device),
            descriptor=descriptor.to(self.device) if descriptor is not None else None,
            descriptors=descriptors.to(self.device) if descriptors is not None else None,
            category=category,
            metadata=metadata,
        )

    def _normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        if not self.bank_cfg.get("normalize_points", True):
            return points
        pts = points.clone()
        pmin, pmax = pts.amin(dim=0), pts.amax(dim=0)
        if self.bank_cfg.get("center_points", True):
            pts = pts - (pmin + pmax) * 0.5
            pmin, pmax = pts.amin(dim=0), pts.amax(dim=0)
        if self.bank_cfg.get("scale_to_unit", True):
            scale = (pmax - pmin).amax().clamp_min(1e-6)
            pts = pts / scale
        return pts

    def _sample_points(self, points: torch.Tensor, index: int) -> torch.Tensor:
        num_points = int(self.bank_cfg.get("num_points") or points.shape[0])
        if points.shape[0] == num_points:
            return points
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.bank_cfg.get("descriptor_seed", 123)) + index)
        if points.shape[0] > num_points:
            idx = torch.randperm(points.shape[0], generator=gen)[:num_points]
        else:
            extra = torch.randint(0, points.shape[0], (num_points - points.shape[0],), generator=gen)
            idx = torch.cat([torch.arange(points.shape[0]), extra], dim=0)
        return points[idx]

    def _check_dim(self, tensor: torch.Tensor, name: str) -> None:
        dim = tensor.shape[-1]
        if dim != self.descriptor_dim:
            raise ValueError(f"{name} dimension mismatch: expected {self.descriptor_dim}, got {dim}")

    def _load_descriptor(self, descriptor_np: Any, descriptors_np: Any, index: int) -> torch.Tensor | None:
        if descriptor_np is None:
            return None
        descriptor = torch.as_tensor(descriptor_np, dtype=torch.float32).flatten()
        self._check_dim(descriptor, "descriptor")
        if not bool(torch.isfinite(descriptor).all()):
            raise ValueError("descriptor contains NaN or Inf")
        return F.normalize(descriptor, dim=0)

    def _load_descriptors(self, descriptors_np: Any) -> torch.Tensor:
        descriptors = torch.as_tensor(descriptors_np, dtype=torch.float32)
        if descriptors.ndim != 2:
            raise ValueError(f"descriptors must be [V,D], got {tuple(descriptors.shape)}")
        self._check_dim(descriptors, "descriptors")
        if not bool(torch.isfinite(descriptors).all()):
            raise ValueError("descriptors contain NaN or Inf")
        return F.normalize(descriptors, dim=1)

    def _fallback_descriptor(self, index: int) -> torch.Tensor | None:
        if not self.bank_cfg.get("random_descriptor_if_missing", True):
            return None
        # 缺少预计算 DINO shape descriptor 时，生成 deterministic random descriptor。
        # 这只用于 smoke test；正式版本应替换为多视角 DINOv3 descriptor 预计算。
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.bank_cfg.get("descriptor_seed", 123)) + 1009 * (index + 1))
        desc = torch.randn(self.descriptor_dim, generator=gen)
        return F.normalize(desc, dim=0)

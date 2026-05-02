from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from shapesplat.cache.cache_io import ensure_cache_dir, load_json, load_numpy, load_torch, save_json, save_numpy, save_torch
from shapesplat.data.image_io import save_tensor_image
from shapesplat.frontend.pipeline import FrontEndOutput
from shapesplat.geometry.camera import Camera
from shapesplat.utils.visualization import save_depth_map, save_input_with_mask_overlay, save_mask_grid


@dataclass
class FrontendCacheRecord:
    image_id: str
    cache_dir: str
    masks_path: str
    confidences_path: str
    boxes_path: str
    descriptors_path: str
    depth_path: str
    dino_features_path: Optional[str]
    meta_path: str
    visualization_paths: dict


@dataclass
class FrontendCacheManifestRecord:
    """frontend cache manifest 的一行记录。

    cache manifest 让 dataset/comparison/stress/editing runner 可以按 image_id
    快速找到 cached front-end outputs；cache 中 masks 是 retained visible masks。
    """

    image_id: str
    image_path: str
    cache_dir: str
    masks_path: str
    descriptors_path: str
    depth_path: str
    meta_path: str
    status: str = "valid"
    num_masks: int | None = None
    descriptor_dim: int | None = None
    warnings: list[str] = field(default_factory=list)


def frontend_cache_exists(cache_dir: str | Path) -> bool:
    """检查最小可用 cache 是否存在。

    cache 中的 masks 是 retained visible masks，不是 amodal masks。
    """
    root = Path(cache_dir)
    required = ["masks.npy", "descriptors.npy", "depth.npy", "frontend_meta.json"]
    return all((root / name).exists() for name in required)


def write_frontend_cache_manifest(records: list[FrontendCacheManifestRecord], path: str | Path) -> None:
    """写出 frontend cache manifest CSV。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id",
        "image_path",
        "cache_dir",
        "masks_path",
        "descriptors_path",
        "depth_path",
        "meta_path",
        "status",
        "num_masks",
        "descriptor_dim",
        "warnings",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["warnings"] = "|".join(record.warnings)
            writer.writerow(row)


def load_frontend_cache_manifest(path: str | Path) -> dict[str, FrontendCacheManifestRecord]:
    """读取 frontend cache manifest，并按 image_id 建索引。"""

    out: dict[str, FrontendCacheManifestRecord] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            warnings = [x for x in (row.get("warnings") or "").split("|") if x]
            record = FrontendCacheManifestRecord(
                image_id=row["image_id"],
                image_path=row.get("image_path", ""),
                cache_dir=row["cache_dir"],
                masks_path=row["masks_path"],
                descriptors_path=row["descriptors_path"],
                depth_path=row["depth_path"],
                meta_path=row["meta_path"],
                status=row.get("status", "valid"),
                num_masks=int(row["num_masks"]) if row.get("num_masks") not in (None, "") else None,
                descriptor_dim=int(row["descriptor_dim"]) if row.get("descriptor_dim") not in (None, "") else None,
                warnings=warnings,
            )
            out[record.image_id] = record
    return out


def build_cache_manifest_from_root(cache_root: str | Path, image_manifest: str | Path | None = None) -> list[FrontendCacheManifestRecord]:
    """从 cache root 自动构建 cache manifest records。"""

    image_paths: dict[str, str] = {}
    if image_manifest is not None:
        from shapesplat.datasets.manifest import load_manifest

        image_paths = {r.image_id: r.image_path for r in load_manifest(image_manifest)}
    records: list[FrontendCacheManifestRecord] = []
    for cache_dir in sorted(Path(cache_root).iterdir() if Path(cache_root).exists() else []):
        if not cache_dir.is_dir():
            continue
        image_id = cache_dir.name
        status = "valid" if frontend_cache_exists(cache_dir) else "invalid"
        warnings = [] if status == "valid" else ["incomplete cache"]
        num_masks = descriptor_dim = None
        if status == "valid":
            try:
                masks = load_numpy(cache_dir / "masks.npy")
                desc = load_numpy(cache_dir / "descriptors.npy")
                num_masks = int(masks.shape[0])
                descriptor_dim = int(desc.shape[1])
            except Exception as exc:
                status = "invalid"
                warnings.append(str(exc))
        records.append(
            FrontendCacheManifestRecord(
                image_id=image_id,
                image_path=image_paths.get(image_id, ""),
                cache_dir=str(cache_dir),
                masks_path=str(cache_dir / "masks.npy"),
                descriptors_path=str(cache_dir / "descriptors.npy"),
                depth_path=str(cache_dir / "depth.npy"),
                meta_path=str(cache_dir / "frontend_meta.json"),
                status=status,
                num_masks=num_masks,
                descriptor_dim=descriptor_dim,
                warnings=warnings,
            )
        )
    return records


def save_frontend_output(
    front: FrontEndOutput,
    cache_dir: str | Path,
    image_id: str,
    save_dino_features: bool = False,
    save_visuals: bool = True,
) -> FrontendCacheRecord:
    """保存 FrontEndOutput。

    真实 backend 很慢时，缓存 masks / descriptors / depth 是批量实验稳定复现的关键。
    """
    root = ensure_cache_dir(cache_dir)
    masks_path = root / "masks.npy"
    confidences_path = root / "mask_confidences.npy"
    boxes_path = root / "boxes.npy"
    descriptors_path = root / "descriptors.npy"
    depth_path = root / "depth.npy"
    features_path = root / "dino_features.pt"
    meta_path = root / "frontend_meta.json"

    save_numpy(masks_path, front.masks.detach().cpu().numpy().astype("float32"))
    save_numpy(confidences_path, front.mask_confidences.detach().cpu().numpy().astype("float32"))
    save_numpy(boxes_path, front.boxes.detach().cpu().numpy().astype("float32"))
    save_numpy(descriptors_path, front.descriptors.detach().cpu().numpy().astype("float32"))
    save_numpy(depth_path, front.depth.detach().cpu().numpy().astype("float32"))
    dino_features_path = None
    if save_dino_features:
        save_torch(features_path, front.dino_features)
        dino_features_path = str(features_path)

    visual_paths: dict[str, str] = {}
    if save_visuals:
        save_tensor_image(front.image, root / "input.png")
        save_mask_grid(front.masks, root / "masks.png")
        save_depth_map(front.depth, root / "depth.png")
        save_input_with_mask_overlay(front.image, front.masks, root / "input_mask_overlay.png")
        visual_paths = {
            "input": str(root / "input.png"),
            "masks": str(root / "masks.png"),
            "depth": str(root / "depth.png"),
            "overlay": str(root / "input_mask_overlay.png"),
        }

    meta = {
        "image_id": image_id,
        "image_shape": list(front.image.shape),
        "masks_shape": list(front.masks.shape),
        "descriptors_shape": list(front.descriptors.shape),
        "depth_shape": list(front.depth.shape),
        "dino_features_saved": bool(save_dino_features),
        "config_path": front.metadata.get("config_path"),
        "image_path": front.metadata.get("image_path"),
        "original_image_shape": front.metadata.get("original_image_shape", list(front.image.shape)),
        "working_image_shape": front.metadata.get("working_image_shape", list(front.image.shape)),
        "original_mask_shape": front.metadata.get("original_mask_shape"),
        "working_mask_shape": front.metadata.get("working_mask_shape", list(front.masks.shape[-2:])),
        "dino_input_size": front.metadata.get("dino_input_size"),
        "dino_feature_shape": front.metadata.get("dino_feature_shape", list(front.dino_features.shape)),
        "mask_resize_applied": front.metadata.get("mask_resize_applied"),
        "mask_resize_mode": front.metadata.get("mask_resize_mode", "nearest"),
        "cache_resolution_tag": front.metadata.get(
            "cache_resolution_tag",
            f"highres_long{front.image.shape[-1] if max(front.image.shape[-2:]) > 160 else 'low'}_dino{front.metadata.get('dino_input_size')}"
            if max(front.image.shape[-2:]) > 160
            else "lowres_possible_minimal",
        ),
    }
    save_json(meta_path, meta)
    record = FrontendCacheRecord(
        image_id=image_id,
        cache_dir=str(root),
        masks_path=str(masks_path),
        confidences_path=str(confidences_path),
        boxes_path=str(boxes_path),
        descriptors_path=str(descriptors_path),
        depth_path=str(depth_path),
        dino_features_path=dino_features_path,
        meta_path=str(meta_path),
        visualization_paths=visual_paths,
    )
    save_json(root / "frontend_cache_record.json", asdict(record))
    return record


def load_frontend_output(cache_dir: str | Path, image: torch.Tensor, camera=None) -> FrontEndOutput:
    """从 cache 还原 FrontEndOutput。

    如果 dino_features.pt 不存在，使用零占位；后续流程只依赖 descriptors 时仍可正常运行。
    """
    root = Path(cache_dir)
    if not frontend_cache_exists(root):
        raise FileNotFoundError(f"frontend cache is incomplete: {root}")
    device = image.device
    masks = torch.from_numpy(load_numpy(root / "masks.npy")).float().to(device)
    confidences = torch.from_numpy(load_numpy(root / "mask_confidences.npy")).float().to(device)
    boxes = torch.from_numpy(load_numpy(root / "boxes.npy")).float().to(device)
    descriptors = torch.from_numpy(load_numpy(root / "descriptors.npy")).float().to(device)
    depth = torch.from_numpy(load_numpy(root / "depth.npy")).float().to(device)
    if (root / "dino_features.pt").exists():
        dino_features = load_torch(root / "dino_features.pt", map_location=device).float()
    else:
        dino_features = torch.zeros(descriptors.shape[1], image.shape[1], image.shape[2], device=device)
    meta = load_json(root / "frontend_meta.json")
    if camera is None:
        _, h, w = image.shape
        camera = Camera.canonical(w, h, 1.2, device)
    meta.update({"frontend_cache_used": True, "frontend_cache_dir": str(root)})
    return FrontEndOutput(image, masks, confidences, boxes, dino_features, descriptors, depth, camera, meta)

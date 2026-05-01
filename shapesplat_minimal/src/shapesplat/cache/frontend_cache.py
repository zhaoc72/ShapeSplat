from __future__ import annotations

from dataclasses import asdict, dataclass
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


def frontend_cache_exists(cache_dir: str | Path) -> bool:
    """检查最小可用 cache 是否存在。

    cache 中的 masks 是 retained visible masks，不是 amodal masks。
    """
    root = Path(cache_dir)
    required = ["masks.npy", "descriptors.npy", "depth.npy", "frontend_meta.json"]
    return all((root / name).exists() for name in required)


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
    if camera is None:
        _, h, w = image.shape
        camera = Camera.canonical(w, h, 1.2, device)
    return FrontEndOutput(image, masks, confidences, boxes, dino_features, descriptors, depth, camera)

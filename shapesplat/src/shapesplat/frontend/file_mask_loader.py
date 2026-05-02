from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from shapesplat.frontend.types import MaskSet
from shapesplat.geometry.masks import mask_to_box, stable_sort_masks


def masks_from_label_map(label_map: torch.Tensor, keep_ids: Iterable[int] | None = None) -> torch.Tensor:
    """从 [H,W] instance id map 构造 [N,H,W] visible masks。

    label 0 默认视为背景。file masks 表示 retained visible masks，不是 amodal masks。
    """
    labels = label_map.long()
    ids = sorted(int(v) for v in torch.unique(labels).tolist() if int(v) != 0)
    if keep_ids is not None:
        keep = {int(v) for v in keep_ids}
        ids = [v for v in ids if v in keep]
    if not ids:
        return torch.zeros((0, *labels.shape), dtype=torch.float32)
    return torch.stack([(labels == v).float() for v in ids], dim=0)


def masks_from_rgb_instance_png(rgb: np.ndarray) -> torch.Tensor:
    """从 RGB instance PNG 中把每种非黑色颜色视为一个 instance。"""
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    colors = sorted({tuple(int(c) for c in row) for row in flat if tuple(int(c) for c in row) != (0, 0, 0)})
    if not colors:
        return torch.zeros((0, h, w), dtype=torch.float32)
    masks = []
    for color in colors:
        m = np.all(rgb == np.array(color, dtype=np.uint8)[None, None, :], axis=-1)
        masks.append(torch.from_numpy(m.astype("float32")))
    return torch.stack(masks, dim=0)


def compute_boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """为 [N,H,W] masks 生成 xyxy boxes。"""
    if masks.shape[0] == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=masks.device)
    return torch.stack([mask_to_box(m) for m in masks], dim=0).float()


def _resize_masks_nearest(masks: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    if masks.shape[-2:] == (h, w):
        return masks.float()
    x = masks.float()[:, None]
    y = F.interpolate(x, size=(h, w), mode="nearest")
    return y[:, 0].float()


def _mask_resize_metadata(original_hw: tuple[int, int], image_hw: tuple[int, int], cfg: dict) -> dict:
    return {
        "original_mask_shape": [int(original_hw[0]), int(original_hw[1])],
        "working_mask_shape": [int(image_hw[0]), int(image_hw[1])],
        "mask_resize_applied": tuple(original_hw) != tuple(image_hw),
        "mask_resize_mode": cfg.get("frontend", cfg).get("mask_resize_mode", "nearest"),
    }


def _parse_keep_ids(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


def standardize_file_masks(
    masks: torch.Tensor,
    image_hw: tuple[int, int],
    cfg: dict,
    confidences: torch.Tensor | None = None,
    boxes: torch.Tensor | None = None,
) -> MaskSet:
    """统一 file mask 到 MaskSet，并做面积过滤、数量限制和稳定排序。

    该逻辑用于 same-mask setting：所有方法共享同一组 retained visible masks，
    从而避免 proposal quality 干扰 reconstruction / ownership / editing 评估。
    """
    fcfg = cfg.get("frontend", cfg)
    device = torch.device(cfg.get("device", "cpu")) if isinstance(cfg, dict) else torch.device("cpu")
    masks = masks.float().to(device)
    original_hw = tuple(int(v) for v in masks.shape[-2:])
    masks = _resize_masks_nearest(masks, image_hw)
    masks = (masks > 0.5).float()
    if confidences is None:
        confidences = torch.ones((masks.shape[0],), dtype=torch.float32, device=device)
    else:
        confidences = confidences.float().to(device).flatten()[: masks.shape[0]]
    if boxes is None or boxes.shape[0] != masks.shape[0]:
        boxes = compute_boxes_from_masks(masks)
    else:
        boxes = boxes.float().to(device)

    h, w = image_hw
    area = masks.flatten(1).sum(dim=1) if masks.shape[0] else torch.zeros((0,), device=device)
    min_ratio = float(fcfg.get("mask_min_area_ratio", fcfg.get("min_area_ratio", 0.0)))
    keep = area >= min_ratio * h * w
    if masks.shape[0]:
        masks, confidences, boxes = masks[keep], confidences[keep], boxes[keep]

    max_num = fcfg.get("mask_max_num_objects")
    if max_num is None:
        max_num = fcfg.get("max_num_objects")
    if max_num is not None and masks.shape[0] > int(max_num):
        masks, confidences, boxes = masks[: int(max_num)], confidences[: int(max_num)], boxes[: int(max_num)]

    if masks.shape[0] == 0:
        raise ValueError("File mask loader found no valid masks after filtering.")
    boxes = compute_boxes_from_masks(masks)
    masks, confidences, boxes = stable_sort_masks(masks, confidences, boxes)
    return MaskSet(masks.float(), confidences.float(), boxes.float(), _mask_resize_metadata(original_hw, image_hw, cfg))


def _load_np(path: Path, image_hw: tuple[int, int], cfg: dict) -> MaskSet:
    fcfg = cfg.get("frontend", cfg)
    keep_ids = _parse_keep_ids(fcfg.get("mask_keep_ids"))
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        if "masks" in data:
            arr = data["masks"]
            masks = torch.from_numpy(arr.astype("float32"))
        else:
            key = next((k for k in ("mask", "labels", "instance_map") if k in data), None)
            if key is None:
                raise ValueError(f"npz mask file must contain masks/mask/labels/instance_map: {path}")
            masks = masks_from_label_map(torch.from_numpy(data[key]), keep_ids=keep_ids)
        conf = torch.from_numpy(data["confidences"].astype("float32")) if "confidences" in data else None
        if conf is None and "scores" in data:
            conf = torch.from_numpy(data["scores"].astype("float32"))
        boxes = torch.from_numpy(data["boxes"].astype("float32")) if "boxes" in data else None
        return standardize_file_masks(masks, image_hw, cfg, conf, boxes)

    arr = np.load(path, allow_pickle=False)
    x = torch.from_numpy(arr)
    if x.ndim == 3:
        masks = x.float()
    elif x.ndim == 2:
        masks = masks_from_label_map(x, keep_ids=keep_ids)
    else:
        raise ValueError(f"Unsupported npy mask shape {arr.shape}; expected [N,H,W] or [H,W].")
    return standardize_file_masks(masks, image_hw, cfg)


def _load_png(path: Path, image_hw: tuple[int, int], cfg: dict) -> MaskSet:
    img = Image.open(path)
    if img.mode in ("RGB", "RGBA"):
        arr = np.array(img.convert("RGB"))
        unique_colors = np.unique(arr.reshape(-1, 3), axis=0)
        if len(unique_colors) > 2:
            masks = masks_from_rgb_instance_png(arr)
        else:
            gray = np.array(img.convert("L"))
            masks = masks_from_label_map(torch.from_numpy(gray))
    else:
        gray = np.array(img.convert("L"))
        masks = masks_from_label_map(torch.from_numpy(gray))
    return standardize_file_masks(masks, image_hw, cfg)


def load_mask_file(path: str | Path, image_hw: tuple[int, int], cfg: dict) -> MaskSet:
    """读取用户提供的 retained visible instance masks。

    支持 .npy/.npz/.png。file masks 用于 same-mask protocol：真实论文主实验中
    所有方法应共享同一组 visible masks；hidden branch 只负责 plausible completion。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mask file not found: {p.resolve()}")
    fmt = str(cfg.get("frontend", cfg).get("mask_format", "auto")).lower()
    suffix = p.suffix.lower()
    if fmt == "auto":
        fmt = suffix.lstrip(".")
    if fmt in ("npy", "npz") or suffix in (".npy", ".npz"):
        return _load_np(p, image_hw, cfg)
    if fmt == "png" or suffix == ".png":
        return _load_png(p, image_hw, cfg)
    raise ValueError(f"Unsupported mask format {fmt!r} for file: {p}")

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from shapesplat.baselines.protocol import BaselineInputSpec, write_baseline_input_spec
from shapesplat.data.image_io import save_tensor_image
from shapesplat.frontend.file_mask_loader import compute_boxes_from_masks
from shapesplat.utils.visualization import save_input_with_mask_overlay, save_mask_grid


def _clip_box(box: list[int], height: int, width: int) -> list[int]:
    x0, y0, x1, y1 = box
    return [max(0, x0), max(0, y0), min(width - 1, x1), min(height - 1, y1)]


def _pad_box(box: torch.Tensor, padding: int, height: int, width: int) -> list[int]:
    """根据 mask bbox 加 padding，并裁剪到图像范围内。"""

    x0, y0, x1, y1 = [int(v) for v in box.tolist()]
    return _clip_box([x0 - padding, y0 - padding, x1 + padding, y1 + padding], height, width)


def export_baseline_inputs(
    image: torch.Tensor,
    masks: torch.Tensor,
    out_dir: str | Path,
    image_id: str,
    crop_padding: int = 8,
) -> BaselineInputSpec:
    """导出 baseline 共享输入。

    per-object baseline 可以读取 crops/object_xxx_rgb.png 与 mask；
    scene-level baseline 可以读取 image.png；所有方法共享 masks.npy，确保
    same-mask setting 下 proposal 完全一致。
    """

    out_dir = Path(out_dir)
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    image = image.detach().cpu().float().clamp(0, 1)
    masks = (masks.detach().cpu().float() > 0.5).float()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"image must be [3,H,W], got {tuple(image.shape)}")
    if masks.ndim != 3:
        raise ValueError(f"masks must be [N,H,W], got {tuple(masks.shape)}")

    _, height, width = image.shape
    boxes = compute_boxes_from_masks(masks).cpu()

    image_path = out_dir / "image.png"
    masks_path = out_dir / "masks.npy"
    masks_vis_path = out_dir / "masks.png"
    overlay_path = out_dir / "overlay.png"
    metadata_path = out_dir / "metadata.json"
    spec_path = out_dir / "input_spec.json"

    save_tensor_image(image, image_path)
    np.save(masks_path, masks.numpy().astype("uint8"))
    save_mask_grid(masks, masks_vis_path)
    save_input_with_mask_overlay(image, masks, overlay_path)

    object_meta = []
    for object_id in range(masks.shape[0]):
        crop_box = _pad_box(boxes[object_id], crop_padding, height, width)
        x0, y0, x1, y1 = crop_box
        rgb_crop = image[:, y0 : y1 + 1, x0 : x1 + 1]
        mask_crop = masks[object_id, y0 : y1 + 1, x0 : x1 + 1]
        rgba_crop = torch.cat([rgb_crop, mask_crop[None]], dim=0)

        rgb_path = crop_dir / f"object_{object_id:03d}_rgb.png"
        mask_path = crop_dir / f"object_{object_id:03d}_mask.png"
        rgba_path = crop_dir / f"object_{object_id:03d}_rgba.png"
        meta_path = crop_dir / f"object_{object_id:03d}_meta.json"

        save_tensor_image(rgb_crop, rgb_path)
        save_tensor_image(mask_crop, mask_path)
        # PIL RGBA 保存需要手动处理，因为通用 save_tensor_image 只负责 RGB/灰度。
        rgba_arr = (rgba_crop.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype("uint8")
        from PIL import Image

        Image.fromarray(rgba_arr, mode="RGBA").save(rgba_path)

        item = {
            "object_id": object_id,
            "bbox_xyxy": [float(v) for v in boxes[object_id].tolist()],
            "crop_bbox_xyxy": crop_box,
            "crop_rgb_path": str(rgb_path),
            "crop_mask_path": str(mask_path),
            "crop_rgba_path": str(rgba_path),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2, ensure_ascii=False)
        object_meta.append(item)

    metadata = {
        "image_id": image_id,
        "image_size": [height, width],
        "num_objects": int(masks.shape[0]),
        "boxes": [[float(v) for v in box.tolist()] for box in boxes],
        "crop_padding": int(crop_padding),
        "masks_are_visible": True,
        "mask_format": "stack_npy",
        "objects": object_meta,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    spec = BaselineInputSpec(
        image_id=image_id,
        image_path=str(image_path),
        masks_path=str(masks_path),
        output_dir=str(out_dir),
        num_objects=int(masks.shape[0]),
        crop_dir=str(crop_dir),
        metadata_path=str(metadata_path),
    )
    write_baseline_input_spec(spec, spec_path)
    return spec


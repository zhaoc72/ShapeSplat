from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from shapesplat.frontend.depth_backend import build_depth_backend
from shapesplat.frontend.depth_normalization import normalize_depth_to_canonical
from shapesplat.frontend.dino_backend import build_dino_backend
from shapesplat.frontend.mask_source import get_masks_for_image
from shapesplat.geometry.camera import Camera


@dataclass
class FrontEndOutput:
    image: torch.Tensor
    masks: torch.Tensor
    mask_confidences: torch.Tensor
    boxes: torch.Tensor
    dino_features: torch.Tensor
    descriptors: torch.Tensor
    depth: torch.Tensor
    camera: Camera


def build_frontend(
    image: torch.Tensor,
    cfg: Dict[str, Any],
    record=None,
    cache_dir=None,
    use_cache: bool = False,
    save_cache: bool = False,
) -> FrontEndOutput:
    """构建 frozen front-end 输出。

    SAM/file mask source 负责 where：retained visible instance masks；
    DINO backend 负责 what：dense features 和 mask-guided descriptors；
    Depth backend 只提供 weak initialization/layout cue。

    file mask 模式用于 same-mask protocol，所有方法共享同一组 visible masks。
    默认 mask_source=sam 时行为与旧版本一致。
    """
    device = torch.device(cfg["device"])
    image = image.to(device)
    if use_cache and cache_dir is not None:
        from shapesplat.cache.frontend_cache import frontend_cache_exists, load_frontend_output

        # 真实 backend 批量实验时优先读取缓存，避免重复运行 SAM / DINO / Depth。
        if frontend_cache_exists(cache_dir):
            return load_frontend_output(cache_dir, image)

    mask_set = get_masks_for_image(image, cfg, record=record)

    dino = build_dino_backend(cfg)
    feats = dino.extract_dense_features(image)
    desc = dino.pool_descriptors(feats, mask_set.masks)

    depth_model = build_depth_backend(cfg)
    raw_depth = depth_model.predict_depth(image)
    depth = normalize_depth_to_canonical(
        raw_depth,
        mask_set.masks,
        cfg,
        cfg["camera"]["z_near"],
        cfg["camera"]["z_far"],
    )
    _, h, w = image.shape
    camera = Camera.canonical(w, h, cfg["camera"]["focal_scale"], device)
    front = FrontEndOutput(image, mask_set.masks, mask_set.confidences, mask_set.boxes, feats, desc, depth, camera)
    if save_cache and cache_dir is not None:
        from shapesplat.cache.frontend_cache import save_frontend_output

        image_id = getattr(record, "image_id", None) or "image"
        save_frontend_output(
            front,
            cache_dir,
            image_id=image_id,
            save_dino_features=bool(cfg.get("frontend_cache", {}).get("save_dino_features", False)),
            save_visuals=True,
        )
    return front

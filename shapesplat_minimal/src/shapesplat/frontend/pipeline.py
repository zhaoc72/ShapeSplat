from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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

    cache 模式只读取/保存 masks、descriptors 和 depth 等 front-end outputs；
    后续 Gaussian optimization 仍然正常运行。cache 中的 masks 是 retained
    visible masks，不是 amodal masks。
    """

    device = torch.device(cfg["device"])
    image = image.to(device)
    cache_cfg = cfg.get("frontend_cache", {})

    if cache_dir is None and record is not None:
        cache_dir = getattr(record, "metadata", {}).get("frontend_cache_dir")
    if cache_dir is None and record is not None and cache_cfg.get("cache_root"):
        cache_dir = Path(cache_cfg["cache_root"]) / getattr(record, "image_id", "image")
    if not use_cache and cache_cfg.get("use_cache", False) and cache_dir is not None:
        use_cache = True
    if not save_cache and cache_cfg.get("save_cache", False) and cache_dir is not None:
        save_cache = True

    if use_cache and cache_dir is not None:
        from shapesplat.cache.frontend_cache import frontend_cache_exists, load_frontend_output
        from shapesplat.cache.validate_cache import validate_frontend_cache_dir

        if frontend_cache_exists(cache_dir):
            if cache_cfg.get("validate_on_load", True):
                validation = validate_frontend_cache_dir(cache_dir, image_hw=tuple(image.shape[-2:]))
                if validation["valid"]:
                    return load_frontend_output(cache_dir, image)
                if not cache_cfg.get("fallback_to_compute", True):
                    raise RuntimeError(f"frontend cache invalid: {validation}")
            else:
                return load_frontend_output(cache_dir, image)
        elif not cache_cfg.get("fallback_to_compute", True):
            raise FileNotFoundError(f"frontend cache missing or incomplete: {cache_dir}")

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
            save_dino_features=bool(cache_cfg.get("save_dino_features", False)),
            save_visuals=True,
        )
    return front

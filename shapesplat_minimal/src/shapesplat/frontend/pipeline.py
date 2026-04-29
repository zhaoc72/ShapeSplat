from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch

from shapesplat.frontend.sam_backend import build_sam_backend
from shapesplat.frontend.dino_backend import build_dino_backend
from shapesplat.frontend.depth_backend import build_depth_backend
from shapesplat.frontend.depth_normalization import normalize_depth_to_canonical
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


def build_frontend(image: torch.Tensor, cfg: Dict[str, Any]) -> FrontEndOutput:
    """构建 frozen front-end 输出。

    SAM backend 负责 where：retained visible masks。
    DINO backend 负责 what：dense features 和 mask-guided instance descriptors。
    二者在主方法中 frozen，不由 reconstruction loss 更新。
    当前最小版本不实现复杂的 DINO-assisted merge/split/re-prompt refinement。
    """
    device = torch.device(cfg["device"])
    image = image.to(device)
    # SAM backend 可在 stub / real / auto 间切换，但输出接口固定为 MaskSet。
    sam = build_sam_backend(cfg)
    masks = sam.predict_masks(image)
    # DINO backend 可在 stub / real / auto 间切换，但输出接口固定。
    dino = build_dino_backend(cfg)
    feats = dino.extract_dense_features(image)
    desc = dino.pool_descriptors(feats, masks.masks)
    # depth backend 可替换，但进入 Gaussian 初始化前必须归一化到 canonical camera range。
    depth_model = build_depth_backend(cfg)
    raw_depth = depth_model.predict_depth(image)
    depth = normalize_depth_to_canonical(raw_depth, masks.masks, cfg, cfg["camera"]["z_near"], cfg["camera"]["z_far"])
    _, h, w = image.shape
    camera = Camera.canonical(w, h, cfg["camera"]["focal_scale"], device)
    return FrontEndOutput(image, masks.masks, masks.confidences, masks.boxes, feats, desc, depth, camera)

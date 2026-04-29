from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch

from shapesplat.frontend.sam_backend import build_sam_backend
from shapesplat.frontend.dino_backend import build_dino_backend
from shapesplat.frontend.depth_stub import DepthStub
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
    depth = DepthStub(cfg["camera"]["z_near"], cfg["camera"]["z_far"]).predict_depth(image)
    _, h, w = image.shape
    camera = Camera.canonical(w, h, cfg["camera"]["focal_scale"], device)
    return FrontEndOutput(image, masks.masks, masks.confidences, masks.boxes, feats, desc, depth, camera)

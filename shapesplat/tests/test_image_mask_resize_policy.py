from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import torch
from PIL import Image

from shapesplat.data.image_io import image_resize_kwargs_from_cfg, load_image, resize_mask_nearest
from shapesplat.frontend.file_mask_loader import standardize_file_masks


def _work_dir(name: str) -> Path:
    root = Path("outputs") / "test_image_mask_resize_policy" / f"{name}_{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_image_resize_none_preserves_shape():
    tmp_path = _work_dir("none")
    path = tmp_path / "img.png"
    Image.fromarray(np.zeros((31, 47, 3), dtype=np.uint8)).save(path)
    image = load_image(path, resize_mode="none")
    assert list(image.shape) == [3, 31, 47]


def test_image_resize_keep_aspect_long_side():
    tmp_path = _work_dir("keep")
    path = tmp_path / "img.png"
    Image.fromarray(np.zeros((479, 640, 3), dtype=np.uint8)).save(path)
    image = load_image(path, resize_mode="keep_aspect", long_side=320, size=320)
    assert max(image.shape[-2:]) == 320
    assert image.shape[-2:] == (240, 320)


def test_image_resize_square_keeps_old_behavior():
    tmp_path = _work_dir("square")
    path = tmp_path / "img.png"
    Image.fromarray(np.zeros((20, 40, 3), dtype=np.uint8)).save(path)
    image = load_image(path, resize_mode="square", size=64)
    assert list(image.shape) == [3, 64, 64]


def test_resize_kwargs_from_cfg_keep_aspect():
    kwargs = image_resize_kwargs_from_cfg({"image": {"resize_mode": "keep_aspect", "long_side": 640, "size": 640}})
    assert kwargs["resize_mode"] == "keep_aspect"
    assert kwargs["long_side"] == 640


def test_nearest_mask_resize_preserves_binary_values():
    mask = torch.zeros(1, 5, 7)
    mask[:, 1:4, 2:5] = 1
    resized = resize_mask_nearest(mask, (11, 13))
    assert sorted(float(v) for v in torch.unique(resized)) == [0.0, 1.0]


def test_standardize_file_masks_records_resize_metadata():
    masks = torch.zeros(1, 4, 4)
    masks[:, 1:3, 1:3] = 1
    cfg = {"device": "cpu", "frontend": {"mask_resize_mode": "nearest", "mask_min_area_ratio": 0.0}}
    mask_set = standardize_file_masks(masks, (8, 6), cfg)
    assert list(mask_set.masks.shape) == [1, 8, 6]
    assert mask_set.metadata["mask_resize_applied"] is True
    assert mask_set.metadata["mask_resize_mode"] == "nearest"
    assert sorted(float(v) for v in torch.unique(mask_set.masks)) == [0.0, 1.0]

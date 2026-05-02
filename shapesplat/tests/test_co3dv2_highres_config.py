from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image, resize_mask_nearest


REQUIRED_DIAGNOSTIC_FIELDS = [
    "original_image_shape",
    "original_mask_shape",
    "working_image_shape",
    "working_mask_shape",
    "renderer_image_shape",
    "frontend_cache_used",
    "frontend_cache_dir",
    "mask_source",
    "mask_resize_applied",
    "mask_resize_mode",
    "dino_input_size",
    "dino_feature_shape",
    "debug_iteration_cap_applied",
    "visible_steps",
    "hidden_steps",
    "joint_steps",
    "edit_steps",
    "shape_bank_backend",
    "renderer_backend",
    "renderer_fallback",
    "shape_bank_fallback",
]


def _cpu_load(path: str):
    return load_config(path, runtime_overrides={"device": "cpu", "allow_cpu_fallback": True, "require_cuda_for_experiments": False})


def test_highres_configs_load():
    assert _cpu_load("configs/co3dv2_real_frontend_highres.yaml")["frontend"]["mask_source"] == "file"
    assert _cpu_load("configs/final_ours_co3dv2_highres.yaml")["frontend_cache"]["use_cache"] is True


def test_highres_config_not_128():
    front_cfg = _cpu_load("configs/co3dv2_real_frontend_highres.yaml")
    ours_cfg = _cpu_load("configs/final_ours_co3dv2_highres.yaml")
    assert int(front_cfg["image"]["long_side"]) >= 640
    assert int(ours_cfg["image"]["long_side"]) >= 640
    assert int(front_cfg["frontend"]["dino_input_size"]) >= 448
    assert int(ours_cfg["frontend"]["dino_input_size"]) >= 448
    assert int(front_cfg["image"]["size"]) != 128


def test_mask_resize_mode_nearest():
    cfg = _cpu_load("configs/final_ours_co3dv2_highres.yaml")
    assert cfg["frontend"]["mask_resize_mode"] == "nearest"


def test_image_resize_keep_aspect():
    out = Path("outputs/test_co3dv2_highres_config")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fake.png"
    Image.fromarray(np.zeros((479, 640, 3), dtype=np.uint8)).save(path)
    image = load_image(path, resize_mode="keep_aspect", long_side=640, size=640)
    assert list(image.shape) == [3, 479, 640]


def test_mask_resize_nearest_binary():
    mask = torch.zeros(1, 4, 4)
    mask[:, 1:3, 1:3] = 1
    resized = resize_mask_nearest(mask, (9, 7))
    values = sorted(float(v) for v in torch.unique(resized))
    assert values == [0.0, 1.0]


def test_diagnostics_resolution_fields():
    fake = {key: None for key in REQUIRED_DIAGNOSTIC_FIELDS}
    missing = [key for key in REQUIRED_DIAGNOSTIC_FIELDS if key not in fake]
    assert missing == []

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 7,
    "device": "auto",
    "image": {"size": 64, "input_path": None},
    "frontend": {
        "sam_backend": "stub",
        "sam3_checkpoint": None,
        "sam3_model_type": None,
        "sam3_device": "auto",
        "sam3_prompt_mode": "automatic",
        "sam3_text_prompts": ["object"],
        "sam3_score_threshold": 0.5,
        "sam3_max_masks": 8,
        "max_num_objects": 4,
        "min_area_ratio": 0.002,
        "mask_conf_threshold": 0.2,
    },
    "camera": {"focal_scale": 1.2, "z_near": 1.0, "z_far": 3.2},
    "gaussians": {
        "visible_min": 32,
        "visible_max": 64,
        "visible_density": 0.01,
        "hidden_base": 16,
        "init_log_scale": -3.5,
        "init_opacity": 0.25,
        "hidden_opacity": 0.08,
        "use_hidden": True,
        "use_densification": False,
    },
    "retrieval": {"top_k": 3, "confidence_low": 0.15, "confidence_high": 0.80, "support_sigma": 0.18},
    "renderer": {"beta_depth": 1.5, "min_sigma_px": 1.0, "max_sigma_px": 4.0},
    "training": {
        "lr": 0.025,
        "visible_warmup_iters": 5,
        "hidden_prior_iters": 5,
        "joint_ownership_iters": 5,
        "edit_finetune_iters": 3,
        "log_every": 1,
    },
    "loss_weights": {
        "visible_rgb": 1.0,
        "visible_alpha": 0.5,
        "visible_depth": 0.02,
        "hidden_prior": 0.10,
        "bridge": 0.02,
        "scene": 1.0,
        "identity": 0.50,
        "layout": 0.0,
        "bg": 0.05,
        "reg": 0.001,
        "edit": 0.20,
        "edit_alpha": 0.05,
    },
    "edit": {"dilate_radius": 3},
    "ablation": {
        "use_visible_hidden_split": True,
        "use_hidden_branch": True,
        "use_hidden_prior": True,
        "use_confidence_weighting": True,
        "use_dino_retrieval": True,
        "use_scene_loss": True,
        "use_ownership_loss": True,
        "use_depth_loss": True,
        "use_bg_loss": True,
        "use_bridge_loss": True,
        "use_edit_consistency": True,
        "use_layout_loss": True,
    },
    "ablation_name": "full",
}


def merge_config(default: Dict[str, Any], yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置。

    default 提供最小项目所需的全部键；yaml_config 只覆盖用户显式设置的部分。
    这样后续新增配置项时，旧配置文件仍可运行。
    """
    out = deepcopy(default)
    for key, value in (yaml_config or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = value
    return out


def resolve_device(device_str: str) -> torch.device:
    """解析 device 配置。

    device: auto 时优先使用 CUDA；没有 GPU 时回退 CPU，保证普通 Python+PyTorch 环境也能跑通。
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_config(path: str | Path) -> Dict[str, Any]:
    """读取 yaml 并合并默认配置。"""
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = merge_config(DEFAULT_CONFIG, user_cfg)
    cfg["device"] = str(resolve_device(cfg["device"]))
    return cfg

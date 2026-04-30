from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from shapesplat.baselines.protocol import BaselineOutputSpec, write_baseline_output_spec
from shapesplat.data.image_io import save_tensor_image
from shapesplat.evaluation.report import save_metrics_json


def _union(masks: torch.Tensor) -> torch.Tensor:
    return (masks.float().amax(dim=0) > 0.5).float()


def _normalize_ownership(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = raw.sum(dim=0, keepdim=True).clamp_min(eps)
    return raw / denom


def identity_mask_baseline(image: torch.Tensor, masks: torch.Tensor) -> dict:
    """理想 2D mask ownership baseline。

    这个 dummy baseline 只用于检查协议和指标上限，不代表 3D 重建方法。
    """

    masks = (masks.float() > 0.5).float()
    fg = _union(masks)
    ownership = _normalize_ownership(masks)
    rgb = image.float() * fg[None] + (1.0 - fg[None])
    return {
        "rgb": rgb.clamp(0, 1),
        "alpha": fg,
        "ownership": ownership,
        "bg_ownership": 1.0 - fg,
    }


def independent_blob_baseline(image: torch.Tensor, masks: torch.Tensor) -> dict:
    """粗糙 per-object blob baseline。

    每个 object 用 mask 内平均颜色和模糊 alpha 表示，模拟独立物体重建的
    low-detail 输出。这里只做协议 smoke test，不用于论文结论。
    """

    masks = (masks.float() > 0.5).float()
    n, h, w = masks.shape
    kernel = max(3, int(min(h, w) * 0.09) | 1)
    pad = kernel // 2
    raw_alpha = F.avg_pool2d(masks[:, None], kernel_size=kernel, stride=1, padding=pad)[:, 0].clamp(0, 1)
    ownership = _normalize_ownership(raw_alpha)
    alpha = raw_alpha.amax(dim=0).clamp(0, 1)

    rgb = torch.ones_like(image)
    for i in range(n):
        m = masks[i] > 0.5
        if bool(m.any()):
            color = image[:, m].mean(dim=1)
        else:
            color = torch.ones(3, device=image.device)
        rgb = rgb * (1 - raw_alpha[i][None]) + color.view(3, 1, 1) * raw_alpha[i][None]
    return {
        "rgb": rgb.clamp(0, 1),
        "alpha": alpha,
        "ownership": ownership,
        "bg_ownership": 1.0 - alpha,
    }


def scene_union_baseline(image: torch.Tensor, masks: torch.Tensor) -> dict:
    """holistic foreground baseline。

    只重建 union foreground，不保留 object ownership。ownership 均分给所有
    object，用于暴露 scene-level 方法缺少 object attribution 的问题。
    """

    masks = (masks.float() > 0.5).float()
    fg = _union(masks)
    n = masks.shape[0]
    ownership = fg[None].repeat(n, 1, 1) / max(1, n)
    rgb = image.float() * fg[None] + (1.0 - fg[None])
    return {
        "rgb": rgb.clamp(0, 1),
        "alpha": fg,
        "ownership": ownership,
        "bg_ownership": 1.0 - fg,
    }


DUMMY_BASELINES = {
    "identity_mask": identity_mask_baseline,
    "independent_blob": independent_blob_baseline,
    "scene_union": scene_union_baseline,
}


def run_dummy_baseline(method_name: str, image: torch.Tensor, masks: torch.Tensor) -> dict:
    """按名称运行一个 dummy baseline。"""

    if method_name not in DUMMY_BASELINES:
        raise KeyError(f"Unknown dummy baseline: {method_name}")
    return DUMMY_BASELINES[method_name](image, masks)


def save_baseline_prediction(
    prediction: dict,
    out_dir: str | Path,
    method_name: str,
    image_id: str,
    metrics: dict | None = None,
) -> BaselineOutputSpec:
    """按 baseline output protocol 保存预测结果。"""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_path = out_dir / "render.png"
    alpha_path = out_dir / "alpha.png"
    ownership_path = out_dir / "ownership.npy"
    metrics_path = out_dir / "metrics.json"
    spec_path = out_dir / "output_spec.json"

    save_tensor_image(prediction["rgb"], render_path)
    save_tensor_image(prediction["alpha"], alpha_path)
    ownership = prediction["ownership"].detach().cpu().float()
    np.save(ownership_path, ownership.numpy().astype("float32"))
    object_alpha_paths = []
    for i in range(ownership.shape[0]):
        p = out_dir / f"object_{i}_alpha.png"
        save_tensor_image(ownership[i], p)
        object_alpha_paths.append(str(p))
    if metrics is None:
        metrics = {}
    save_metrics_json(metrics, metrics_path)

    spec = BaselineOutputSpec(
        method_name=method_name,
        image_id=image_id,
        output_dir=str(out_dir),
        render_path=str(render_path),
        alpha_path=str(alpha_path),
        ownership_path=str(ownership_path),
        metrics_path=str(metrics_path),
        object_alpha_paths=object_alpha_paths,
        metadata={"dummy_baseline": True},
    )
    write_baseline_output_spec(spec, spec_path)
    return spec


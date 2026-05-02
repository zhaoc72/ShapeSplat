from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shapesplat.baselines.protocol import BaselineOutputSpec, write_baseline_output_spec
from shapesplat.data.image_io import save_tensor_image
from shapesplat.data.image_io import load_image
from shapesplat.evaluation.report import save_metrics_json
from shapesplat.utils.logging import save_json
from shapesplat.utils.visualization import save_input_with_mask_overlay, save_mask_grid, save_object_alphas, save_ownership_argmax


def export_scene_pointcloud(scene, out_path, mode: str = "gaussian_means", include_hidden: bool = True):
    """从 ObjectGaussianScene 导出轻量 pred pointcloud。

    当前导出 Gaussian centers，是 geometry proxy；正式论文可替换为 surface samples、
    mesh samples 或 Gaussian splat samples。
    """
    pts = []
    for obj in getattr(scene, "objects", []):
        means = obj.means.detach()
        branch = getattr(obj, "branch_ids", None)
        if mode == "visible_only":
            keep = branch == 0 if branch is not None else torch.ones(means.shape[0], dtype=torch.bool, device=means.device)
            means = means[keep]
        elif mode == "hidden_only":
            keep = branch == 1 if branch is not None else torch.zeros(means.shape[0], dtype=torch.bool, device=means.device)
            means = means[keep]
        elif not include_hidden and branch is not None:
            means = means[branch == 0]
        pts.append(means)
    points = torch.cat(pts, dim=0) if pts else torch.zeros((0, 3))
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, points.detach().cpu().float().numpy().astype("float32"))
    return points


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _config_hash(cfg: dict | None) -> str | None:
    if cfg is None:
        return None
    payload = json.dumps(_jsonable(cfg), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _scene_counts(scene) -> dict:
    if scene is None:
        return {"gaussian_count_total": None, "gaussian_count_visible": None, "gaussian_count_hidden": None}
    visible, hidden, total = 0, 0, 0
    for obj in getattr(scene, "objects", []):
        branch = getattr(obj, "branch_ids", None)
        if branch is None:
            k = int(getattr(obj, "means").shape[0])
            total += k
            visible += k
            continue
        total += int(branch.numel())
        visible += int((branch == 0).sum().detach().cpu())
        hidden += int((branch == 1).sum().detach().cpu())
    return {
        "gaussian_count_total": total,
        "gaussian_count_visible": visible,
        "gaussian_count_hidden": hidden,
    }


def save_ours_output(
    out_dir: str | Path,
    image,
    masks,
    render,
    metrics: dict,
    scene=None,
    cfg: dict | None = None,
    diagnostics: dict | None = None,
    save_checkpoint: bool = False,
    image_id: str = "image",
) -> dict:
    """保存 Ours 输出，并兼容 baseline output protocol。

    这样 Ours 可以先独立批量运行，再被 comparison runner 像普通 baseline 一样读取，
    保证论文主表和外部 baseline 使用同一套 render/alpha/ownership 文件协议。
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    save_tensor_image(image, root / "input.png")
    save_tensor_image(image, root / "input_working.png")
    save_mask_grid(masks, root / "masks.png")
    save_mask_grid(masks, root / "masks_working.png")
    save_input_with_mask_overlay(image, masks, root / "input_mask_overlay.png")
    save_input_with_mask_overlay(image, masks, root / "input_mask_overlay_working.png")
    image_path = (diagnostics or {}).get("frontend", {}).get("image_path") or (diagnostics or {}).get("image_path")
    image_path = image_path or (diagnostics or {}).get("frontend_cache_image_path")
    if diagnostics and diagnostics.get("frontend_cache_dir") and not image_path:
        image_path = (diagnostics.get("frontend", {}) or {}).get("image_path")
    if image_path:
        try:
            # 中文注释：fullres 副本用于诊断 report 缩略图和工作分辨率是否混淆。
            full_image = load_image(image_path, resize_mode="none")
            save_tensor_image(full_image, root / "input_fullres_copy.png")
            if tuple(full_image.shape[-2:]) == tuple(masks.shape[-2:]):
                save_input_with_mask_overlay(full_image, masks, root / "input_mask_overlay_fullres.png")
        except Exception:
            pass
    save_tensor_image(render.rgb, root / "render_final.png")
    save_tensor_image(render.alpha, root / "alpha_final.png")
    save_tensor_image(render.rgb, root / "render.png")
    save_tensor_image(render.alpha, root / "alpha.png")
    save_ownership_argmax(render.ownership, root / "ownership_argmax.png")
    save_object_alphas(render.ownership, root)
    ownership_path = root / "ownership.npy"
    np.save(ownership_path, render.ownership.detach().cpu().float().numpy().astype("float32"))

    save_metrics_json(metrics, root / "metrics.json")
    if diagnostics is not None:
        save_json(_jsonable(diagnostics), root / "diagnostics.json")

    counts = _scene_counts(scene)
    meta = {
        "image_id": image_id,
        "num_objects": int(masks.shape[0]),
        **counts,
        "renderer_backend": (getattr(render, "extras", {}) or {}).get("renderer_backend", cfg.get("renderer", {}).get("backend") if cfg else None),
        "shape_bank_backend": cfg.get("shape_bank", {}).get("backend") if cfg else None,
        "frontend_cache_used": bool((diagnostics or {}).get("frontend", {}).get("frontend_cache_used", False)),
        "variant": cfg.get("ours", {}).get("variant", cfg.get("ablation_name", "full")) if cfg else "full",
        "config_hash": _config_hash(cfg),
    }
    save_json(_jsonable(meta), root / "reconstruction_meta.json")

    if save_checkpoint and scene is not None:
        torch.save({"scene": scene.state_dict(), "reconstruction_meta": meta}, root / "checkpoint.pt")
    if scene is not None and cfg is not None and cfg.get("ours", {}).get("save_pred_pointcloud", False):
        export_scene_pointcloud(scene, root / "pred_pointcloud.npy", mode="gaussian_means", include_hidden=True)
        export_scene_pointcloud(scene, root / "pred_pointcloud_visible.npy", mode="visible_only")
        export_scene_pointcloud(scene, root / "pred_pointcloud_hidden.npy", mode="hidden_only")

    object_paths = [str((root / f"object_{i}_alpha.png").resolve()) for i in range(int(render.ownership.shape[0]))]
    spec = BaselineOutputSpec(
        method_name="ours",
        image_id=image_id,
        output_dir=str(root.resolve()),
        render_path=str((root / "render_final.png").resolve()),
        alpha_path=str((root / "alpha_final.png").resolve()),
        ownership_path=str(ownership_path.resolve()),
        metrics_path=str((root / "metrics.json").resolve()),
        object_alpha_paths=object_paths,
        metadata=meta,
    )
    write_baseline_output_spec(spec, root / "output_spec.json")
    return {"output_dir": str(root), "output_spec": str(root / "output_spec.json"), **meta}

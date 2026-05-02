from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from shapesplat.baselines.evaluate_baseline import evaluate_baseline_prediction
from shapesplat.baselines.load_outputs import load_baseline_output
from shapesplat.data.image_io import load_image
from shapesplat.datasets.manifest import load_manifest
from shapesplat.evaluation.geometry_metrics import compute_geometry_metrics_from_paths
from shapesplat.evaluation.report import flatten_metrics, save_metrics_csv, save_metrics_json
from shapesplat.frontend.file_mask_loader import load_mask_file


def _resize_prediction(pred: dict, masks: torch.Tensor) -> dict:
    h, w = masks.shape[-2:]
    out = dict(pred)
    if "rgb" in out and tuple(out["rgb"].shape[-2:]) != (h, w):
        out["rgb"] = F.interpolate(out["rgb"][None].float(), size=(h, w), mode="bilinear", align_corners=False)[0].clamp(0, 1)
    if "alpha" in out and tuple(out["alpha"].shape[-2:]) != (h, w):
        out["alpha"] = F.interpolate(out["alpha"][None, None].float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0].clamp(0, 1)
    if "ownership" in out and tuple(out["ownership"].shape[-2:]) != (h, w):
        out["ownership"] = F.interpolate(out["ownership"][None].float(), size=(h, w), mode="bilinear", align_corners=False)[0].clamp(0, 1)
        out["ownership"] = out["ownership"] / out["ownership"].sum(dim=0, keepdim=True).clamp_min(1e-6)
    out["bg_ownership"] = (1.0 - out.get("alpha", out["ownership"].sum(dim=0))).clamp(0, 1)
    return out


def _resolve_output_dir(root: str | Path, method_name: str, image_id: str) -> Path:
    base = Path(root)
    candidates = [
        base / "per_image" / image_id,
        base / "per_image" / image_id / method_name,
        base / "per_image" / image_id / "baselines" / method_name,
        base / image_id / method_name,
        base / image_id,
        base / method_name / image_id,
        base / method_name,
        base,
    ]
    for path in candidates:
        if path.exists() and ((path / "ownership.npy").exists() or (path / "output_spec.json").exists() or list(path.glob("object_*_alpha.png"))):
            return path
    return base / "per_image" / image_id


def _gt_pointcloud_path(record) -> str | None:
    md = getattr(record, "metadata", {}) or {}
    for key in ("gt_pointcloud_path", "visible_pointcloud_path"):
        if md.get(key):
            return md[key]
    return None


def evaluate_method_output(
    method_name: str,
    output_dir: str | Path,
    image: torch.Tensor,
    masks: torch.Tensor,
    record=None,
    cfg: dict | None = None,
) -> dict:
    """统一评估 Ours / internal baseline / external baseline 输出。

    Geometry metrics 是 optional：只有 pred_pointcloud.npy 和 manifest 中的 GT pointcloud
    同时存在时才计算，否则记录 available=false。
    """
    out_dir = Path(output_dir)
    pred = _resize_prediction(load_baseline_output(out_dir, method_name, getattr(record, "image_id", "image")), masks)
    metrics = evaluate_baseline_prediction(pred, masks, image=image)
    metrics.update({"method": method_name, "output_dir": str(out_dir)})

    pred_pc = out_dir / "pred_pointcloud.npy"
    gt_pc = _gt_pointcloud_path(record)
    geom_cfg = (cfg or {}).get("geometry", {}) or (cfg or {}).get("metrics", {}).get("geometry", {})
    geometry = compute_geometry_metrics_from_paths(
        pred_pc,
        gt_pc,
        threshold=float(geom_cfg.get("threshold", 0.01)),
        normalize=geom_cfg.get("alignment", geom_cfg.get("normalize", "none")),
        num_sample_points=geom_cfg.get("num_sample_points"),
    )
    metrics["GeometryAvailable"] = bool(geometry.get("available", False))
    if geometry.get("available"):
        metrics.update({k: v for k, v in geometry.items() if k != "available"})
    else:
        metrics["GeometryReason"] = geometry.get("reason", "unavailable")
    return metrics


def _summarize(rows: list[dict], method_name: str) -> dict:
    ok = [r for r in rows if r.get("status") == "success"]
    summary = {"method": method_name, "num_images": len(rows), "num_success": len(ok), "num_failed": len(rows) - len(ok)}
    keys = sorted({k for r in ok for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    for key in keys:
        vals = [float(r[key]) for r in ok if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
        if vals:
            summary[f"{key}_mean"] = sum(vals) / len(vals)
    summary["NumGeometryAvailable"] = sum(1 for r in ok if r.get("GeometryAvailable"))
    return summary


def evaluate_method_dataset(
    method_name: str,
    method_output_root: str | Path,
    manifest_path: str | Path,
    cfg: dict | None,
    out_dir: str | Path,
    max_images: int | None = None,
) -> list[dict]:
    """对一个 method 的全数据集输出做统一评估，单张失败不影响整体。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_manifest(manifest_path)
    if max_images is not None:
        records = records[: int(max_images)]
    rows: list[dict] = []
    for record in records:
        try:
            image = load_image(record.image_path, size=int((cfg or {}).get("image", {}).get("size", 128)))
            mask_path = record.metadata.get("mask_path")
            if not mask_path:
                raise FileNotFoundError(f"record has no mask_path: {record.image_id}")
            masks = load_mask_file(mask_path, image_hw=image.shape[-2:], cfg=cfg or {}).masks
            method_dir = _resolve_output_dir(method_output_root, method_name, record.image_id)
            metrics = evaluate_method_output(method_name, method_dir, image, masks, record=record, cfg=cfg)
            rows.append({"image_id": record.image_id, "status": "success", **metrics})
        except Exception as exc:
            rows.append({"image_id": record.image_id, "method": method_name, "status": "failed", "error": str(exc)})
    flat = [flatten_metrics(r) for r in rows]
    save_metrics_json(rows, out / "method_per_image_metrics.json")
    save_metrics_csv(flat, out / "method_per_image_metrics.csv")
    summary = _summarize(flat, method_name)
    save_metrics_json(summary, out / "method_summary.json")
    save_metrics_csv([summary], out / "method_summary.csv")
    return rows

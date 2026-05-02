from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from shapesplat.baselines.dummy_baselines import DUMMY_BASELINES, run_dummy_baseline, save_baseline_prediction
from shapesplat.baselines.evaluate_baseline import evaluate_baseline_prediction
from shapesplat.baselines.export_inputs import export_baseline_inputs
from shapesplat.baselines.independent_gaussian import run_independent_gaussian_baseline
from shapesplat.baselines.load_outputs import load_baseline_output
from shapesplat.data.image_io import save_tensor_image
from shapesplat.evaluation.report import flatten_metrics, save_metrics_csv, save_metrics_json
from shapesplat.experiments.single_image import run_single_image_experiment
from shapesplat.frontend.file_mask_loader import load_mask_file
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.utils.comparison_visualization import make_comparison_grid


def _save_shared_masks(masks: torch.Tensor, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shared_masks.npy"
    np.save(path, (masks.detach().cpu().float() > 0.5).numpy().astype("uint8"))
    return path


def _row(method: str, image_id: str, status: str, output_dir: Path, metrics: dict | None = None, error: str | None = None) -> dict:
    item = {"image_id": image_id, "method": method, "status": status, "output_dir": str(output_dir)}
    if metrics:
        item.update(metrics)
    if error:
        item["error"] = error
    return item


def _resize_prediction_to_masks(pred: dict, masks: torch.Tensor) -> dict:
    """把已保存的 Ours/baseline 输出对齐到当前 same-mask 分辨率。"""
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


def _resolve_external_output_dir(base_dir: str | Path, method: str, image_id: str) -> Path:
    """按常见 external baseline dataset 输出结构查找某张图的输出目录。

    支持直接传 method 输出目录，也支持传 dataset 根目录：root/image_id/method。
    找不到时返回原始目录，让后续 loader 给出清晰错误并记录 warning。
    """

    base = Path(base_dir)
    candidates = [
        base,
        base / image_id / method,
        base / image_id,
        base / "per_image" / image_id / "external" / method,
        base / "per_image" / image_id / method,
    ]
    for path in candidates:
        if path.exists() and ((path / "ownership.npy").exists() or list(path.glob("object_*_alpha.png"))):
            return path
    return base


def _resolve_ours_output_dir(base_dir: str | Path, image_id: str) -> Path:
    """在已批量运行的 Ours benchmark 输出中按 image_id 找结果目录。"""
    base = Path(base_dir)
    candidates = [
        base / "per_image" / image_id,
        base / image_id,
        base / "ours",
        base,
    ]
    for path in candidates:
        if path.exists() and ((path / "ownership.npy").exists() or (path / "output_spec.json").exists()):
            return path
    return base / "per_image" / image_id


def run_comparison_for_image(
    image: torch.Tensor,
    masks: torch.Tensor,
    cfg: dict,
    out_dir: str | Path,
    image_id: str,
    run_ours: bool = True,
    run_dummy_baselines: bool = True,
    run_independent_gaussian: bool = False,
    external_baseline_dirs: dict[str, str] | None = None,
    save_visuals: bool = True,
    save_checkpoint: bool = False,
    frontend_cache_dir=None,
    use_frontend_cache: bool = False,
    save_frontend_cache: bool = False,
    ours_output_dir: str | Path | None = None,
) -> list[dict]:
    """对单张图运行 same-mask comparison。

    每个 method 独立 try/except，某个 baseline 或 ours 失败时只记录该方法失败，
    不阻断其他方法。所有方法共享同一组 visible masks。
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    method_outputs: dict[str, dict] = {}
    shared_mask_path = _save_shared_masks(masks, out_dir)
    save_tensor_image(image, out_dir / "input.png")

    if run_ours:
        ours_dir = out_dir / "ours"
        try:
            if ours_output_dir is not None:
                # 允许先批量跑 Ours，再在 comparison 中直接读取标准输出，避免重复训练主方法。
                resolved = _resolve_ours_output_dir(ours_output_dir, image_id)
                pred = load_baseline_output(resolved, "ours", image_id)
                pred = _resize_prediction_to_masks(pred, masks)
                metrics = evaluate_baseline_prediction(pred, masks, image=image)
                rows.append(_row("ours", image_id, "success", resolved, metrics=metrics))
                method_outputs["ours"] = pred
            else:
                cfg_ours = copy.deepcopy(cfg)
                cfg_ours.setdefault("frontend", {})
                cfg_ours["frontend"]["mask_source"] = "file"
                cfg_ours["frontend"]["mask_path"] = str(shared_mask_path)
                ours_row = run_single_image_experiment(
                    image,
                    cfg_ours,
                    ours_dir,
                    image_id=image_id,
                    save_visuals=save_visuals,
                    save_checkpoint=save_checkpoint,
                    eval_metrics=True,
                    frontend_cache_dir=frontend_cache_dir,
                    use_frontend_cache=use_frontend_cache,
                    save_frontend_cache=save_frontend_cache,
                )
                rows.append(_row("ours", image_id, "success", ours_dir, metrics=ours_row))
                method_outputs["ours"] = load_baseline_output(ours_dir, "ours", image_id)
        except Exception as exc:
            ours_dir.mkdir(parents=True, exist_ok=True)
            err = _row("ours", image_id, "failed", ours_dir, error=str(exc))
            save_metrics_json(err, ours_dir / "metrics.json")
            rows.append(err)

    if run_dummy_baselines:
        baseline_root = out_dir / "baselines"
        for method in DUMMY_BASELINES:
            method_dir = baseline_root / method
            try:
                pred = run_dummy_baseline(method, image, masks)
                metrics = evaluate_baseline_prediction(pred, masks, image=image)
                save_baseline_prediction(pred, method_dir, method, image_id, metrics=metrics)
                rows.append(_row(method, image_id, "success", method_dir, metrics=metrics))
                method_outputs[method] = pred
            except Exception as exc:
                method_dir.mkdir(parents=True, exist_ok=True)
                err = _row(method, image_id, "failed", method_dir, error=str(exc))
                save_metrics_json(err, method_dir / "metrics.json")
                rows.append(err)

    if run_independent_gaussian:
        method_dir = out_dir / "baselines" / "independent_gaussian"
        try:
            row = run_independent_gaussian_baseline(image, masks, cfg, method_dir, image_id=image_id, save_visuals=save_visuals)
            rows.append(row)
            method_outputs["independent_gaussian"] = load_baseline_output(method_dir, "independent_gaussian", image_id)
        except Exception as exc:
            method_dir.mkdir(parents=True, exist_ok=True)
            err = _row("independent_gaussian", image_id, "failed", method_dir, error=str(exc))
            save_metrics_json(err, method_dir / "metrics.json")
            rows.append(err)

    for method, directory in (external_baseline_dirs or {}).items():
        method_dir = out_dir / "external" / method
        try:
            resolved_dir = _resolve_external_output_dir(directory, method, image_id)
            pred = load_baseline_output(resolved_dir, method, image_id)
            metrics = evaluate_baseline_prediction(pred, masks, image=image)
            method_dir.mkdir(parents=True, exist_ok=True)
            save_metrics_json(metrics, method_dir / "metrics.json")
            rows.append(_row(method, image_id, "success", resolved_dir, metrics=metrics))
            method_outputs[method] = pred
        except Exception as exc:
            method_dir.mkdir(parents=True, exist_ok=True)
            err = _row(method, image_id, "failed", method_dir, error=str(exc))
            save_metrics_json(err, method_dir / "metrics.json")
            rows.append(err)

    save_metrics_json(rows, out_dir / "comparison.json")
    save_metrics_csv([flatten_metrics(r) for r in rows], out_dir / "comparison.csv")
    if save_visuals:
        make_comparison_grid(image, masks, method_outputs, out_dir / "qualitative_grid.png", title=image_id)
    return rows


def _load_record_masks_or_fallback(image: torch.Tensor, cfg: dict, record) -> torch.Tensor:
    if cfg.get("frontend_cache", {}).get("use_cache", False):
        cache_dir = getattr(record, "metadata", {}).get("frontend_cache_dir") if record is not None else None
        if cache_dir:
            from shapesplat.cache.frontend_cache import load_frontend_output
            return load_frontend_output(cache_dir, image).masks
    mask_path = getattr(record, "metadata", {}).get("mask_path") if record is not None else None
    source = cfg.get("frontend", {}).get("mask_source", "sam")
    if mask_path:
        return load_mask_file(mask_path, image_hw=image.shape[-2:], cfg=cfg).masks
    if source == "file":
        raise FileNotFoundError(f"record has no mask_path in file mask mode: {getattr(record, 'image_id', 'unknown')}")
    print(f"Warning: no mask_path for {getattr(record, 'image_id', 'unknown')}; falling back to front-end masks.")
    return build_frontend(image, cfg, record=record).masks


def run_comparison_dataset(
    dataset,
    cfg: dict,
    out_dir: str | Path,
    max_images: int | None = None,
    skip_existing: bool = False,
    run_ours: bool = True,
    run_dummy_baselines: bool = True,
    run_independent_gaussian: bool = False,
    save_visuals: bool = True,
    save_checkpoint: bool = False,
    save_frontend_cache: bool = False,
    ours_output_dir: str | Path | None = None,
) -> list[dict]:
    """在 manifest dataset 上批量运行 same-mask comparison。

    单张图失败会写入 error.json 并继续，保证批量实验不中断。
    """

    out_dir = Path(out_dir)
    per_image_root = out_dir / "per_image"
    rows: list[dict] = []
    total = len(dataset) if max_images is None else min(len(dataset), int(max_images))
    for idx in range(total):
        item = dataset[idx]
        image_id = item["image_id"]
        image_dir = per_image_root / image_id
        comp_path = image_dir / "comparison.json"
        if skip_existing and comp_path.exists():
            with open(comp_path, "r", encoding="utf-8") as f:
                rows.extend(json.load(f))
            continue
        try:
            print(f"[{idx + 1}/{total}] comparison: {image_id}")
            masks = _load_record_masks_or_fallback(item["image"], cfg, item.get("record"))
            export_baseline_inputs(item["image"], masks, image_dir / "baseline_inputs", image_id)
            rows.extend(
                run_comparison_for_image(
                    item["image"],
                    masks,
                    cfg,
                    image_dir,
                    image_id,
                    run_ours=run_ours,
                    run_dummy_baselines=run_dummy_baselines,
                    run_independent_gaussian=run_independent_gaussian,
                    save_visuals=save_visuals,
                    save_checkpoint=save_checkpoint,
                    frontend_cache_dir=getattr(item.get("record"), "metadata", {}).get("frontend_cache_dir"),
                    use_frontend_cache=bool(cfg.get("frontend_cache", {}).get("use_cache", False)),
                    save_frontend_cache=save_frontend_cache or bool(cfg.get("frontend_cache", {}).get("save_cache", False)),
                    ours_output_dir=ours_output_dir,
                )
            )
        except Exception as exc:
            image_dir.mkdir(parents=True, exist_ok=True)
            err = _row("comparison", image_id, "failed", image_dir, error=str(exc))
            save_metrics_json(err, image_dir / "error.json")
            rows.append(err)
            print(f"Warning: comparison failed on {image_id}: {exc}")
    return rows

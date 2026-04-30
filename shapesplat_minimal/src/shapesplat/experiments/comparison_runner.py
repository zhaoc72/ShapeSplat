from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from shapesplat.baselines.dummy_baselines import DUMMY_BASELINES, run_dummy_baseline, save_baseline_prediction
from shapesplat.baselines.evaluate_baseline import evaluate_baseline_prediction
from shapesplat.baselines.export_inputs import export_baseline_inputs
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


def run_comparison_for_image(
    image: torch.Tensor,
    masks: torch.Tensor,
    cfg: dict,
    out_dir: str | Path,
    image_id: str,
    run_ours: bool = True,
    run_dummy_baselines: bool = True,
    external_baseline_dirs: dict[str, str] | None = None,
    save_visuals: bool = True,
    save_checkpoint: bool = False,
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

    for method, directory in (external_baseline_dirs or {}).items():
        method_dir = out_dir / "external" / method
        try:
            pred = load_baseline_output(directory, method, image_id)
            metrics = evaluate_baseline_prediction(pred, masks, image=image)
            method_dir.mkdir(parents=True, exist_ok=True)
            save_metrics_json(metrics, method_dir / "metrics.json")
            rows.append(_row(method, image_id, "success", Path(directory), metrics=metrics))
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
    save_visuals: bool = True,
    save_checkpoint: bool = False,
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
                    save_visuals=save_visuals,
                    save_checkpoint=save_checkpoint,
                )
            )
        except Exception as exc:
            image_dir.mkdir(parents=True, exist_ok=True)
            err = _row("comparison", image_id, "failed", image_dir, error=str(exc))
            save_metrics_json(err, image_dir / "error.json")
            rows.append(err)
            print(f"Warning: comparison failed on {image_id}: {exc}")
    return rows


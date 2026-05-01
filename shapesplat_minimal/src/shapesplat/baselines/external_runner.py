from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from shapesplat.baselines.evaluate_baseline import evaluate_baseline_prediction
from shapesplat.baselines.load_outputs import load_baseline_output
from shapesplat.baselines.protocol import BaselineInputSpec, write_baseline_output_spec
from shapesplat.baselines.registry import get_adapter
from shapesplat.baselines.validate_outputs import validate_baseline_output_dir
from shapesplat.data.image_io import load_image
from shapesplat.evaluation.report import flatten_metrics, save_metrics_csv, save_metrics_json


def _load_masks(path: str | Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).float()


def run_external_baseline_for_image(
    adapter_name: str,
    input_spec: BaselineInputSpec,
    out_dir: str | Path,
    cfg: dict,
    dry_run: bool = False,
) -> dict:
    """对单张图运行 external baseline adapter。

    这是后续真实 baseline 的统一单图入口；当前只提供 dummy/mock 与 command
    template，不下载模型、不安装外部 repo。
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(cfg, out_dir / "adapter_config.json")
    adapter = get_adapter(adapter_name)
    try:
        spec = adapter.run(input_spec, out_dir, cfg, dry_run=dry_run)
        write_baseline_output_spec(spec, out_dir / "output_spec.json")
        if dry_run:
            validation = {"valid": True, "warnings": ["dry_run: outputs were not produced"], "errors": [], "found_files": {}}
            save_metrics_json(validation, out_dir / "validation.json")
            return {
                "method": adapter_name,
                "image_id": input_spec.image_id,
                "status": "dry_run",
                "output_dir": str(out_dir),
                "command": spec.metadata.get("command"),
            }

        masks = _load_masks(input_spec.masks_path)
        image_hw = tuple(masks.shape[-2:])
        validation = validate_baseline_output_dir(
            out_dir,
            expected_num_objects=input_spec.num_objects,
            image_hw=image_hw,
            strict=bool(cfg.get("strict_validation", False)),
        )
        save_metrics_json(validation, out_dir / "validation.json")
        if not validation.get("valid"):
            row = {
                "method": adapter_name,
                "image_id": input_spec.image_id,
                "status": "failed",
                "output_dir": str(out_dir),
                "error": "; ".join(validation.get("errors", [])),
            }
            save_metrics_json(row, out_dir / "metrics.json")
            return row
        prediction = load_baseline_output(out_dir, adapter_name, input_spec.image_id)
        image = load_image(input_spec.image_path, size=None)
        metrics = evaluate_baseline_prediction(prediction, masks, image=image)
        row = {
            "method": adapter_name,
            "image_id": input_spec.image_id,
            "status": "success",
            "output_dir": str(out_dir),
            **metrics,
        }
        save_metrics_json(row, out_dir / "metrics.json")
        return row
    except Exception as exc:
        row = {
            "method": adapter_name,
            "image_id": input_spec.image_id,
            "status": "failed",
            "output_dir": str(out_dir),
            "error": str(exc),
        }
        save_metrics_json(row, out_dir / "metrics.json")
        return row


def run_external_baseline_dataset(
    adapter_name: str,
    input_specs: list[BaselineInputSpec],
    out_dir: str | Path,
    cfg: dict,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> list[dict]:
    """批量运行 external baseline adapter。

    单张失败不影响整个 dataset；结果保存为 external_baseline_summary.json/csv。
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx, spec in enumerate(input_specs, start=1):
        image_out = out_dir / spec.image_id / adapter_name
        metrics_path = image_out / "metrics.json"
        if skip_existing and metrics_path.exists():
            import json

            with open(metrics_path, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
            continue
        print(f"[{idx}/{len(input_specs)}] external baseline {adapter_name}: {spec.image_id}")
        rows.append(run_external_baseline_for_image(adapter_name, spec, image_out, cfg, dry_run=dry_run))
    save_metrics_json(rows, out_dir / "external_baseline_summary.json")
    save_metrics_csv([flatten_metrics(r) for r in rows], out_dir / "external_baseline_summary.csv")
    return rows


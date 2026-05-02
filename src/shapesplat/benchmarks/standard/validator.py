from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from shapesplat.benchmarks.standard.manifest_schema import REQUIRED_COLUMNS, normalize_manifest_row
from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.image_io import load_image
from shapesplat.evaluation.report import save_metrics_csv
from shapesplat.frontend.file_mask_loader import load_mask_file


def _mask_duplicate_warnings(masks: torch.Tensor, threshold: float = 0.95) -> list[str]:
    warnings: list[str] = []
    if masks.shape[0] < 2:
        return warnings
    flat = masks.flatten(1).float()
    for i in range(masks.shape[0]):
        for j in range(i + 1, masks.shape[0]):
            inter = torch.minimum(flat[i], flat[j]).sum()
            union = torch.maximum(flat[i], flat[j]).sum().clamp_min(1e-6)
            iou = float((inter / union).detach().cpu())
            if iou > threshold:
                warnings.append(f"masks {i} and {j} are highly overlapping/duplicated: IoU={iou:.3f}")
    return warnings


def validate_benchmark_manifest(manifest_path: str | Path, cfg: dict | None = None, max_rows: int | None = None) -> dict:
    """验证正式 same-mask benchmark 数据协议。

    validator 用于实验前检查 image/mask/metadata 是否齐全，以及 mask 是否像 retained visible instance masks。
    """
    manifest = Path(manifest_path)
    report = {"valid": False, "num_rows": 0, "num_valid": 0, "num_failed": 0, "rows": [], "warnings": [], "errors": []}
    if not manifest.exists():
        report["errors"].append(f"manifest not found: {manifest}")
        return report
    cfg = merge_config(DEFAULT_CONFIG, cfg or {})
    with open(manifest, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing_cols:
            report["errors"].append(f"manifest missing columns: {missing_cols}")
            return report
        rows = list(reader)
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    report["num_rows"] = len(rows)
    for idx, raw in enumerate(rows):
        row_report = {"row": idx, "image_id": raw.get("image_id", ""), "status": "failed", "warnings": [], "errors": []}
        try:
            row = normalize_manifest_row(raw, manifest.parent)
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"image missing: {image_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"mask missing: {mask_path}")
            image = load_image(image_path, size=None)
            mask_set = load_mask_file(mask_path, image_hw=image.shape[-2:], cfg=cfg)
            masks = mask_set.masks.detach().cpu()
            h, w = image.shape[-2:]
            areas = masks.flatten(1).sum(dim=1)
            min_area = float(cfg.get("frontend", {}).get("mask_min_area_ratio", 0.002)) * h * w
            small = [int(i) for i, a in enumerate(areas) if float(a) <= min_area]
            if small:
                row_report["warnings"].append(f"small masks below min area: {small}")
            row_report["warnings"].extend(_mask_duplicate_warnings(masks))
            if row.get("metadata_path") and not Path(row["metadata_path"]).exists():
                row_report["warnings"].append(f"metadata missing: {row['metadata_path']}")
            if row.get("metadata_path") and Path(row["metadata_path"]).exists():
                json.loads(Path(row["metadata_path"]).read_text(encoding="utf-8"))
            if row.get("num_objects"):
                expected = int(row["num_objects"])
                if expected != masks.shape[0]:
                    row_report["warnings"].append(f"num_objects={expected} but loaded masks={masks.shape[0]}")
            if not row.get("split"):
                row_report["warnings"].append("empty split")
            if not row.get("subset"):
                row_report["warnings"].append("empty subset")
            row_report.update(
                {
                    "status": "valid",
                    "num_masks": int(masks.shape[0]),
                    "image_shape": list(image.shape),
                    "mask_shape": list(masks.shape),
                }
            )
            report["num_valid"] += 1
        except Exception as exc:
            row_report["errors"].append(str(exc))
            report["num_failed"] += 1
        report["rows"].append(row_report)
    report["valid"] = report["num_failed"] == 0 and not report["errors"]
    return report


def save_validation_report(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_metrics_csv(report.get("rows", []), out / "benchmark_validation.csv")

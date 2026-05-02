from __future__ import annotations

import json
from pathlib import Path

import torch

from shapesplat.cache.validate_cache import validate_frontend_cache_dir
from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.image_io import load_image
from shapesplat.datasets.benchmark.manifest_v2 import load_benchmark_manifest
from shapesplat.datasets.benchmark.schema import OPTIONAL_GT_COLUMNS, REQUIRED_COLUMNS
from shapesplat.evaluation.report import save_metrics_csv
from shapesplat.frontend.file_mask_loader import load_mask_file


def _dup_warnings(masks: torch.Tensor, threshold: float = 0.95) -> list[str]:
    warnings = []
    if masks.shape[0] < 2:
        return warnings
    flat = masks.flatten(1).float()
    for i in range(masks.shape[0]):
        for j in range(i + 1, masks.shape[0]):
            inter = torch.minimum(flat[i], flat[j]).sum()
            union = torch.maximum(flat[i], flat[j]).sum().clamp_min(1e-6)
            iou = float((inter / union).detach().cpu())
            if iou > threshold:
                warnings.append(f"duplicated masks {i}/{j}: IoU={iou:.3f}")
    return warnings


def validate_benchmark_v2(manifest_path: str | Path, cfg: dict | None = None, max_rows: int | None = None, check_optional_gt: bool = True, check_cache: bool = False) -> dict:
    """正式 benchmark v2 validator。

    optional GT 缺失不会导致失败；只有显式给出路径但不存在才 warning/error。
    check_cache=True 时会额外验证 frontend cache 完整性。
    """

    manifest = Path(manifest_path)
    report = {"valid": False, "num_rows": 0, "num_valid": 0, "num_failed": 0, "num_warnings": 0, "rows": [], "warnings": [], "errors": []}
    if not manifest.exists():
        report["errors"].append(f"manifest not found: {manifest}")
        return report
    cfg = merge_config(DEFAULT_CONFIG, cfg or {})
    if str(cfg.get("device", "cpu")).lower() == "auto":
        cfg["device"] = "cpu"
    try:
        records = load_benchmark_manifest(manifest)
    except Exception as exc:
        report["errors"].append(f"failed to load manifest: {exc}")
        return report
    if max_rows is not None:
        records = records[: int(max_rows)]
    report["num_rows"] = len(records)
    valid_splits = {"train", "val", "test", "diagnostic"}
    for idx, record in enumerate(records):
        row = {"row": idx, "image_id": record.image_id, "status": "failed", "warnings": [], "errors": []}
        try:
            for col in REQUIRED_COLUMNS:
                if not getattr(record, col):
                    row["errors"].append(f"missing required column value: {col}")
            image_path = Path(record.image_path)
            mask_path = Path(record.mask_path)
            if not image_path.exists():
                row["errors"].append(f"image missing: {image_path}")
            if not mask_path.exists():
                row["errors"].append(f"mask missing: {mask_path}")
            if row["errors"]:
                raise FileNotFoundError("; ".join(row["errors"]))
            image = load_image(image_path, size=None)
            masks = load_mask_file(mask_path, image_hw=image.shape[-2:], cfg=cfg).masks.detach().cpu()
            if masks.shape[0] < 1:
                row["errors"].append("mask count < 1")
            areas = masks.flatten(1).sum(dim=1)
            empty = [int(i) for i, area in enumerate(areas) if float(area) <= 0]
            if empty:
                row["errors"].append(f"empty masks: {empty}")
            row["warnings"].extend(_dup_warnings(masks))
            if record.num_objects is not None and int(record.num_objects) != int(masks.shape[0]):
                row["warnings"].append(f"num_objects={record.num_objects} but masks={masks.shape[0]}")
            if record.split not in valid_splits:
                row["warnings"].append(f"non-standard split: {record.split}")
            if not record.subset:
                row["warnings"].append("empty subset")
            if record.metadata_path:
                meta = Path(record.metadata_path)
                if meta.exists():
                    json.loads(meta.read_text(encoding="utf-8"))
                else:
                    row["warnings"].append(f"metadata missing: {meta}")
            if check_optional_gt:
                for col in OPTIONAL_GT_COLUMNS:
                    value = getattr(record, col)
                    if value and not Path(value).exists():
                        row["warnings"].append(f"optional GT path missing: {col}={value}")
            if check_cache and record.frontend_cache_dir:
                cache_report = validate_frontend_cache_dir(record.frontend_cache_dir, image_hw=tuple(image.shape[-2:]))
                if not cache_report["valid"]:
                    row["warnings"].append(f"frontend cache invalid: {cache_report['errors']}")
            elif check_cache:
                row["warnings"].append("frontend_cache_dir missing")
            if row["errors"]:
                raise ValueError("; ".join(row["errors"]))
            row.update({"status": "valid", "num_masks": int(masks.shape[0]), "image_shape": list(image.shape), "mask_shape": list(masks.shape)})
            report["num_valid"] += 1
        except Exception as exc:
            if not row["errors"]:
                row["errors"].append(str(exc))
            report["num_failed"] += 1
        report["num_warnings"] += len(row["warnings"])
        report["rows"].append(row)
    report["valid"] = report["num_failed"] == 0 and not report["errors"]
    return report


def save_benchmark_v2_validation(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_v2_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    save_metrics_csv(report.get("rows", []), out / "benchmark_v2_validation.csv")

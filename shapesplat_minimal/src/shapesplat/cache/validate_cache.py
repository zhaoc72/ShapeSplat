from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shapesplat.cache.frontend_cache import load_frontend_cache_manifest


def validate_frontend_cache_dir(cache_dir: str | Path, image_hw: tuple[int, int] | None = None) -> dict:
    """验证单个 frontend cache 是否完整。

    真实前端批量实验前建议先跑 cache validation，避免后续 comparison /
    editing 读到缺失或 shape 不一致的 cache。
    """

    root = Path(cache_dir)
    errors: list[str] = []
    warnings: list[str] = []
    required = ["masks.npy", "descriptors.npy", "depth.npy", "frontend_meta.json"]
    for name in required:
        if not (root / name).exists():
            errors.append(f"missing {name}")
    num_masks = descriptor_dim = height = width = None
    if not errors:
        try:
            masks = np.load(root / "masks.npy")
            desc = np.load(root / "descriptors.npy")
            depth = np.load(root / "depth.npy")
            if masks.ndim != 3:
                errors.append(f"masks must be [N,H,W], got {masks.shape}")
            else:
                num_masks, height, width = int(masks.shape[0]), int(masks.shape[1]), int(masks.shape[2])
                if num_masks < 1:
                    errors.append("cache has zero masks")
                if not np.any(masks > 0):
                    errors.append("all masks are empty")
            if desc.ndim != 2:
                errors.append(f"descriptors must be [N,D], got {desc.shape}")
            else:
                descriptor_dim = int(desc.shape[1])
                if num_masks is not None and desc.shape[0] != num_masks:
                    errors.append("descriptor count does not match masks")
                if not np.isfinite(desc).all():
                    errors.append("descriptors contain NaN/Inf")
            if depth.ndim != 2:
                errors.append(f"depth must be [H,W], got {depth.shape}")
            elif height is not None and depth.shape != (height, width):
                errors.append("depth shape does not match masks")
            elif not np.isfinite(depth).all():
                errors.append("depth contains NaN/Inf")
            if image_hw is not None and height is not None and (height, width) != tuple(image_hw):
                errors.append(f"cache HW {(height, width)} != image HW {image_hw}")
        except Exception as exc:
            errors.append(str(exc))
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_masks": num_masks,
        "descriptor_dim": descriptor_dim,
        "height": height,
        "width": width,
        "cache_dir": str(root),
    }


def validate_frontend_cache_manifest(cache_manifest: str | Path) -> dict:
    """逐行验证 cache manifest。"""

    records = load_frontend_cache_manifest(cache_manifest)
    rows = []
    for image_id, record in records.items():
        row = validate_frontend_cache_dir(record.cache_dir)
        row["image_id"] = image_id
        rows.append(row)
    return {
        "valid": all(r["valid"] for r in rows),
        "num_records": len(rows),
        "num_valid": sum(bool(r["valid"]) for r in rows),
        "num_failed": sum(not bool(r["valid"]) for r in rows),
        "rows": rows,
    }


def save_cache_validation(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

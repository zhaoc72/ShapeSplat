from __future__ import annotations

import csv
import json
from pathlib import Path


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_resolution_rows(root: str | Path, max_items: int | None = None) -> list[dict]:
    """收集 per-image diagnostics 中的分辨率字段。

    中文注释：report 中的 qualitative 图通常是缩略图；这里保存 full/working/render
    分辨率统计，帮助确认 CO3Dv2 是否真的没有走 minimal 128 链路。
    """
    out = Path(root)
    rows: list[dict] = []
    per_image = out / "per_image"
    dirs = sorted([p for p in per_image.iterdir() if p.is_dir()]) if per_image.exists() else []
    if max_items is not None:
        dirs = dirs[: int(max_items)]
    for d in dirs:
        data = _read_json(d / "diagnostics.json") or {}
        rows.append(
            {
                "image_id": d.name,
                "original_image_shape": data.get("original_image_shape"),
                "original_mask_shape": data.get("original_mask_shape"),
                "working_image_shape": data.get("working_image_shape"),
                "working_mask_shape": data.get("working_mask_shape"),
                "renderer_image_shape": data.get("renderer_image_shape"),
                "dino_input_size": data.get("dino_input_size"),
                "debug_iteration_cap_applied": data.get("debug_iteration_cap_applied"),
                "renderer_backend": data.get("renderer_backend"),
                "shape_bank_backend": data.get("shape_bank_backend"),
                "frontend_cache_used": data.get("frontend_cache_used"),
                "output_dir": str(d),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()}) if rows else ["image_id"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_co3dv2_highres_report(root: str | Path, out_dir: str | Path, title: str = "CO3Dv2 High-Resolution Diagnostics") -> dict:
    """生成 CO3Dv2 high-res diagnostic 报告。

    中文注释：CO3Dv2 single 是单 foreground real-image diagnostic，不是多物体主 benchmark；
    报告会保留 fallback warning，避免把 SoftRenderer/ToyShapeBank 结果误当 paper-final。
    """
    final_root = Path(root)
    out = Path(out_dir)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rows = collect_resolution_rows(final_root)
    summary = _read_json(final_root / "ours_summary.json") or _read_json(final_root / "summary.json") or {}
    readiness = _read_json(final_root / "readiness" / "readiness.json") or _read_json(final_root / "readiness.json") or {}
    _write_csv(tables / "resolution_summary.csv", rows)
    _write_csv(tables / "summary.csv", [summary] if isinstance(summary, dict) and summary else [])
    qualitative_lines = ["# Qualitative Index", ""]
    for row in rows:
        qualitative_lines.append(f"- `{row['image_id']}`: `{row['output_dir']}`")
    (out / "qualitative_index.md").write_text("\n".join(qualitative_lines), encoding="utf-8")
    warnings = []
    for row in rows:
        if row.get("debug_iteration_cap_applied"):
            warnings.append(f"{row['image_id']}: debug iteration cap was applied")
        if row.get("renderer_backend") == "soft":
            warnings.append(f"{row['image_id']}: SoftRenderer diagnostic output")
        if row.get("shape_bank_backend") in ("toy", "auto"):
            warnings.append(f"{row['image_id']}: Toy/auto shape bank diagnostic output")
    warnings.extend(readiness.get("warnings", []) if isinstance(readiness, dict) else [])
    lines = [
        f"# {title}",
        "",
        "CO3Dv2 single is treated here as a real-image diagnostic / single foreground visible-mask benchmark, not as a multi-object main benchmark.",
        "",
        "Images shown in quick reports may be thumbnails. Full-resolution and working-resolution outputs are stored in each per-image output directory.",
        "",
        "## Summary",
        f"- Num rows: `{len(rows)}`",
        f"- Ours summary path: `{final_root / 'ours_summary.json'}`",
        "",
        "## Resolution",
        f"- Resolution table: `{tables / 'resolution_summary.csv'}`",
        "",
        "## Warnings",
        *[f"- {w}" for w in warnings],
        "",
        "## Qualitative Outputs",
        f"- Index: `{out / 'qualitative_index.md'}`",
    ]
    report_path = out / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "root": str(final_root),
        "report_path": str(report_path),
        "resolution_summary": str(tables / "resolution_summary.csv"),
        "summary_csv": str(tables / "summary.csv"),
        "qualitative_index": str(out / "qualitative_index.md"),
        "num_rows": len(rows),
        "warnings": warnings,
    }
    (out / "report_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest

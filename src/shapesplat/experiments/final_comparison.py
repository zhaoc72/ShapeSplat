from __future__ import annotations

from pathlib import Path

from shapesplat.baselines.method_catalog import get_enabled_methods, group_methods_by_family, load_method_catalog
from shapesplat.evaluation.method_output_evaluator import evaluate_method_dataset
from shapesplat.evaluation.report import flatten_metrics, save_metrics_csv, save_metrics_json


def _summarize_method(rows: list[dict], method, family: str) -> dict:
    ok = [r for r in rows if r.get("status") == "success"]
    out = {"method": method, "family": family, "num_images": len(rows), "num_success": len(ok), "num_failed": len(rows) - len(ok)}
    keys = sorted({k for r in ok for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    for key in keys:
        vals = [float(r[key]) for r in ok if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
        if vals:
            out[f"{key}_mean"] = sum(vals) / len(vals)
    out["NumGeometryAvailable"] = sum(1 for r in ok if r.get("GeometryAvailable"))
    return out


def _family_summary(method_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in method_rows:
        groups.setdefault(row.get("family", "unknown"), []).append(row)
    out = []
    for family, rows in groups.items():
        item = {"family": family, "num_methods": len(rows), "num_success": sum(int(r.get("num_success", 0)) for r in rows)}
        for key in sorted({k for r in rows for k, v in r.items() if k.endswith("_mean") and isinstance(v, (int, float))}):
            vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                item[key] = sum(vals) / len(vals)
        out.append(item)
    return out


def run_final_comparison(
    manifest_path: str | Path,
    method_catalog_path: str | Path,
    method_output_roots: dict[str, str],
    cfg: dict,
    out_dir: str | Path,
    max_images: int | None = None,
) -> dict:
    """正式主表评估入口：统一评估 Ours、内部 baseline 和外部 baseline outputs。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = load_method_catalog(method_catalog_path)
    enabled = get_enabled_methods(methods)
    method_rows: list[dict] = []
    all_rows: list[dict] = []
    warnings: list[str] = []
    for method in enabled:
        root = method_output_roots.get(method.name)
        if not root or not Path(root).exists():
            warnings.append(f"missing output root for enabled method {method.name}: {root}")
            continue
        rows = evaluate_method_dataset(method.name, root, manifest_path, cfg, out / "methods" / method.name, max_images=max_images)
        all_rows.extend(rows)
        method_rows.append(_summarize_method([flatten_metrics(r) for r in rows], method.name, method.family))
    family_rows = _family_summary(method_rows)
    save_metrics_json(all_rows, out / "final_per_image_metrics.json")
    save_metrics_csv([flatten_metrics(r) for r in all_rows], out / "final_per_image_metrics.csv")
    save_metrics_json(method_rows, out / "final_method_summary.json")
    save_metrics_csv(method_rows, out / "final_method_summary.csv")
    save_metrics_json(family_rows, out / "final_family_summary.json")
    save_metrics_csv(family_rows, out / "final_family_summary.csv")
    save_metrics_json({"warnings": warnings, "num_methods": len(method_rows)}, out / "final_comparison_meta.json")
    return {"method_summary": method_rows, "family_summary": family_rows, "warnings": warnings}

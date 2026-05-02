from __future__ import annotations

import math
from pathlib import Path

from shapesplat.editing.metrics import compute_edit_metrics
from shapesplat.editing.ops import apply_edit
from shapesplat.editing.visualization import save_edit_visuals
from shapesplat.evaluation.report import save_metrics_csv, save_metrics_json


DEFAULT_EDIT_OPS = ["remove", "translate", "scale", "isolate", "object_only"]


def _spec_for_op(op: str, object_id: int, cfg: dict | None = None) -> dict:
    editing = (cfg or {}).get("editing", {})
    return {
        "op": op,
        "object_id": int(object_id),
        "translation": editing.get("translate_vector", [0.12, 0.0, 0.0]),
        "scale": editing.get("scale_factor", 1.2),
    }


def run_single_edit(
    scene,
    renderer,
    front,
    object_id: int,
    edit_spec: dict,
    out_dir: str | Path,
    save_visuals: bool = True,
) -> dict:
    """对单个 object 执行一次推理阶段编辑，并保存指标/可视化。"""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_render = renderer(scene)
    edited_scene = apply_edit(scene, edit_spec)
    edited_render = renderer(edited_scene)
    op = edit_spec.get("op", "remove")
    metrics = compute_edit_metrics(base_render, edited_render, front.masks, int(object_id), op)
    if save_visuals:
        save_edit_visuals(base_render, edited_render, front.masks, int(object_id), out, op)
    save_metrics_json(metrics, out / "metrics.json")
    return metrics


def summarize_edit_metrics(rows: list[dict]) -> dict:
    """按 edit op 聚合指标；不依赖 pandas。"""

    keys = ["CollateralL1", "AlphaCollateral", "EditLocality", "DeletionResidual", "ObjectSupportIoU"]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("EditOp", "unknown")), []).append(row)
    summary = {}
    for op, items in sorted(groups.items()):
        out = {"op": op, "num_edits": len(items)}
        for key in keys:
            vals = []
            for item in items:
                try:
                    v = float(item.get(key))
                    if math.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            out[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
        summary[op] = out
    return summary


def run_edit_suite_for_scene(
    scene,
    renderer,
    front,
    out_dir: str | Path,
    object_ids: list[int] | None = None,
    edit_ops: list[str] | None = None,
    save_visuals: bool = True,
    cfg: dict | None = None,
) -> list[dict]:
    """object-level editability 的推理评估套件。"""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if object_ids is None:
        max_objects = int((cfg or {}).get("editing", {}).get("max_objects", 3))
        object_ids = list(range(min(len(scene.objects), max_objects)))
    edit_ops = edit_ops or list((cfg or {}).get("editing", {}).get("default_ops", DEFAULT_EDIT_OPS))
    rows: list[dict] = []
    for oid in object_ids:
        for op in edit_ops:
            edit_dir = out / f"object_{int(oid):03d}" / op
            spec = _spec_for_op(op, int(oid), cfg)
            metrics = run_single_edit(scene, renderer, front, int(oid), spec, edit_dir, save_visuals=save_visuals)
            rows.append(metrics)
    summary = summarize_edit_metrics(rows)
    save_metrics_json(rows, out / "edit_metrics.json")
    save_metrics_csv(rows, out / "edit_metrics.csv")
    save_metrics_json(summary, out / "edit_summary.json")
    save_metrics_csv(list(summary.values()), out / "edit_summary.csv")
    return rows


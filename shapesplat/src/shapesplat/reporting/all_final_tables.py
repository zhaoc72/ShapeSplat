from __future__ import annotations

from pathlib import Path

from shapesplat.reporting.final_tables import export_final_tables
from shapesplat.reporting.io import load_json
from shapesplat.reporting.paper_tables import export_paper_table
from shapesplat.reporting.tables import flatten_summary


def _export_optional(root: Path, out: Path, source: Path, kind: str, csv_name: str) -> None:
    """导出可选 final 子实验表；缺失文件说明该 smoke run 没跑对应步骤。"""
    if not source.exists():
        return
    rows = flatten_summary(load_json(source))
    export_paper_table(rows, kind, out / f"{csv_name}.csv", out / f"{csv_name}.tex", f"{csv_name} summary.", f"tab:{csv_name}")


def export_all_final_tables(root: str | Path, out_dir: str | Path, columns: str | Path | None = None) -> dict:
    """导出 final paper 相关表格；这是可复用逻辑，CLI 只负责参数解析。"""
    root = Path(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    final_summary = root / "comparison" / "final_method_summary.json"
    if final_summary.exists():
        written.update(export_final_tables(final_summary, out))
    _export_optional(root, out, root / "variants" / "variant_summary.json", "ablation", "ablation")
    _export_optional(root, out, root / "stress" / "stress_subset_summary.json", "stress", "stress")
    _export_optional(root, out, root / "editing" / "edit_summary.json", "editing", "editing")
    return written

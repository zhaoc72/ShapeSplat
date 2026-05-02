from __future__ import annotations

from pathlib import Path

from shapesplat.reporting.io import load_json, save_csv_rows
from shapesplat.reporting.latex import save_latex_table
from shapesplat.reporting.tables import format_float, select_table_columns


MAIN_COLUMNS = [
    "method",
    "family",
    "num_success",
    "AttrAcc_mean",
    "AttrPurity_mean_mean",
    "Leakage_mean",
    "InstIoU_mean_mean",
    "IsoIoU_mean_mean",
    "ForegroundRGBL1_mean",
    "ChamferL2_mean",
    "FScore_mean",
    "EditLocality_mean",
    "DeletionResidual_mean",
]

GEOMETRY_COLUMNS = ["method", "ChamferL2_mean", "FScore_mean", "Precision_mean", "Recall_mean", "NumGeometryAvailable"]
OWNERSHIP_COLUMNS = ["method", "family", "AttrAcc_mean", "AttrPurity_mean_mean", "Leakage_mean", "InstIoU_mean_mean", "IsoIoU_mean_mean"]
EDITING_COLUMNS = ["method", "family", "EditLocality_mean", "DeletionResidual_mean", "CollateralL1_mean"]


def _table(rows: list[dict], columns: list[str]) -> list[dict]:
    return [{k: format_float(v) for k, v in row.items()} for row in select_table_columns(rows, columns)]


def make_main_comparison_table(rows: list[dict]) -> list[dict]:
    return _table(rows, MAIN_COLUMNS)


def make_geometry_table(rows: list[dict]) -> list[dict]:
    return _table(rows, GEOMETRY_COLUMNS)


def export_final_tables(final_summary_path, out_dir) -> dict:
    """导出 final comparison 的 CSV/LaTeX 表格草稿。"""
    rows = load_json(final_summary_path)
    if isinstance(rows, dict):
        rows = rows.get("method_summary", [])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = {
        "main_comparison": (make_main_comparison_table(rows), MAIN_COLUMNS, "Final main comparison.", "tab:final_main"),
        "geometry": (make_geometry_table(rows), GEOMETRY_COLUMNS, "Optional geometry metrics.", "tab:geometry"),
        "ownership": (_table(rows, OWNERSHIP_COLUMNS), OWNERSHIP_COLUMNS, "Ownership metrics.", "tab:ownership"),
        "editing": (_table(rows, EDITING_COLUMNS), EDITING_COLUMNS, "Editing diagnostics.", "tab:editing"),
    }
    written = {}
    for name, (table_rows, columns, caption, label) in specs.items():
        csv_path = out / f"{name}.csv"
        tex_path = out / f"{name}.tex"
        save_csv_rows(table_rows, csv_path)
        save_latex_table(table_rows, columns, caption, label, tex_path)
        written[name] = {"csv": str(csv_path), "tex": str(tex_path)}
    return written

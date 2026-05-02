from __future__ import annotations

from pathlib import Path

from shapesplat.reporting.io import save_csv_rows
from shapesplat.reporting.latex import save_latex_table
from shapesplat.reporting.tables import format_float, select_table_columns


DEFAULT_COLUMNS = {
    "main_comparison": ["method", "num_success", "AttrAcc_mean", "AttrPurity_mean_mean", "Leakage_mean", "InstIoU_mean_mean", "IsoIoU_mean_mean", "EditLocality_mean", "DeletionResidual_mean"],
    "ablation": ["name", "AttrAcc", "AttrPurity_mean", "Leakage", "InstIoU_mean", "IsoIoU_mean", "CollateralL1", "EditLocality", "DeletionResidual"],
    "stress": ["subset", "num_images", "AttrAcc_mean", "Leakage_mean", "InstIoU_mean_mean", "SwapRateProxy_mean", "OrderAccProxy_mean", "OcclusionRecallProxy_mean", "EditLocality_mean"],
    "editing": ["op", "num_edits", "CollateralL1_mean", "AlphaCollateral_mean", "EditLocality_mean", "DeletionResidual_mean", "ObjectSupportIoU_mean"],
}


def make_paper_table(rows: list[dict], kind: str, columns_config: dict | None = None) -> list[dict]:
    """根据 paper table column config 生成论文表格草稿行。"""
    columns = (columns_config or {}).get(kind) or DEFAULT_COLUMNS.get(kind) or (list(rows[0].keys()) if rows else [])
    selected = select_table_columns(rows, columns)
    return [{k: format_float(v) for k, v in row.items()} for row in selected]


def export_paper_table(
    rows: list[dict],
    kind: str,
    out_csv: str | Path,
    out_tex: str | Path,
    caption: str,
    label: str,
    columns_config: dict | None = None,
) -> None:
    """导出 CSV 和 LaTeX 论文表格草稿；投稿前仍需人工检查排版。"""
    columns = (columns_config or {}).get(kind) or DEFAULT_COLUMNS.get(kind) or (list(rows[0].keys()) if rows else [])
    table_rows = make_paper_table(rows, kind, columns_config)
    save_csv_rows(table_rows, out_csv)
    save_latex_table(table_rows, columns, caption, label, out_tex)

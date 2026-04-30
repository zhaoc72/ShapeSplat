from __future__ import annotations

import math
from typing import Any


COMPARISON_COLUMNS = [
    "method",
    "num_success",
    "num_failed",
    "AttrAcc_mean",
    "AttrPurity_mean_mean",
    "Leakage_mean",
    "InstIoU_mean_mean",
    "IsoIoU_mean_mean",
    "ForegroundAlphaError_mean",
    "CollateralL1_mean",
    "EditLocality_mean",
    "DeletionResidual_mean",
]

ABLATION_COLUMNS = [
    "name",
    "AttrAcc",
    "AttrPurity_mean",
    "Leakage",
    "InstIoU_mean",
    "IsoIoU_mean",
    "CollateralL1",
    "EditLocality",
    "DeletionResidual",
]


def _flatten_nested(prefix: str, value: Any, out: dict) -> None:
    if isinstance(value, dict):
        if set(value.keys()) <= {"mean", "std"}:
            if "mean" in value:
                out[f"{prefix}_mean"] = value["mean"]
            if "std" in value:
                out[f"{prefix}_std"] = value["std"]
        else:
            for k, v in value.items():
                _flatten_nested(f"{prefix}_{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = value


def flatten_summary(summary: dict | list) -> list[dict]:
    """将 per-method / ablation summary 展平成 rows。

    该函数兼容 dict-of-dict、list-of-dict 和带 metrics 嵌套的格式。
    """

    if isinstance(summary, list):
        items = [(None, item) for item in summary]
    else:
        items = list(summary.items())
    rows = []
    for key, item in items:
        row: dict = {}
        if isinstance(item, dict):
            if key is not None and "method" not in item and "name" not in item:
                row["method"] = key
            for k, v in item.items():
                _flatten_nested(k, v, row)
        else:
            row["method"] = key
            row["value"] = item
        rows.append(row)
    return rows


def select_table_columns(rows: list[dict], preferred_columns: list[str]) -> list[dict]:
    """按推荐列顺序选择表格列；缺失列填空。"""

    return [{col: row.get(col, "") for col in preferred_columns} for row in rows]


def format_float(value, ndigits: int = 4):
    """格式化 float，非数值保持原样。"""

    try:
        if isinstance(value, str) and value.strip() == "":
            return ""
        x = float(value)
        if math.isfinite(x):
            return f"{x:.{ndigits}f}"
    except (TypeError, ValueError):
        pass
    return value


def make_markdown_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """生成 Markdown 表格，用于 report.md。

    默认列覆盖 ownership、leakage 和 editing stability 的核心指标。
    """

    if not rows:
        return "_No rows found._"
    columns = columns or list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        vals = [str(format_float(row.get(col, ""))) for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def rank_methods(rows: list[dict], metric: str, higher_is_better: bool = True) -> list[dict]:
    """按指定 metric 排序，方便选出 best/worst method。"""

    def key(row):
        try:
            return float(row.get(metric))
        except (TypeError, ValueError):
            return float("-inf") if higher_is_better else float("inf")

    return sorted(rows, key=key, reverse=higher_is_better)


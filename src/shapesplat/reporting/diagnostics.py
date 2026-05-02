from __future__ import annotations

import math
from collections import Counter, defaultdict


def is_finite_number(x) -> bool:
    """判断是否为有限数值。"""

    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


RANGES = {
    "AttrAcc": (0.0, 1.0),
    "AttrAcc_mean": (0.0, 1.0),
    "Leakage": (0.0, 1.0),
    "Leakage_mean": (0.0, 1.0),
    "InstIoU_mean": (0.0, 1.0),
    "InstIoU_mean_mean": (0.0, 1.0),
    "EditLocality": (0.0, 1.0),
    "EditLocality_mean": (0.0, 1.0),
}


def metric_sanity_check(rows: list[dict]) -> dict:
    """检查 NaN/Inf 与常见指标范围。

    这些诊断用于发现日志/评估异常，不是新的论文指标。
    """

    bad_rows = []
    warnings = []
    for idx, row in enumerate(rows):
        problems = []
        for key, value in row.items():
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip()):
                try:
                    x = float(value)
                except ValueError:
                    continue
                if not math.isfinite(x):
                    problems.append(f"{key}=non_finite")
                if key in RANGES and math.isfinite(x):
                    lo, hi = RANGES[key]
                    if x < lo or x > hi:
                        problems.append(f"{key}_out_of_range")
                if key in {"CollateralL1", "CollateralL1_mean"} and math.isfinite(x) and x < 0:
                    problems.append(f"{key}_negative")
        if problems:
            bad = dict(row)
            bad["row_index"] = idx
            bad["problems"] = problems
            bad_rows.append(bad)
    if bad_rows:
        warnings.append(f"Found {len(bad_rows)} rows with non-finite or out-of-range metrics.")
    return {"num_rows": len(rows), "num_bad_rows": len(bad_rows), "bad_rows": bad_rows, "warnings": warnings}


def select_best_worst_cases(rows: list[dict], metric: str, higher_is_better: bool, top_k: int = 5) -> dict:
    """按指标选择 best/worst cases。"""

    valid = [r for r in rows if is_finite_number(r.get(metric))]
    ranked = sorted(valid, key=lambda r: float(r[metric]), reverse=higher_is_better)
    return {"best": ranked[:top_k], "worst": ranked[-top_k:][::-1]}


def detect_failure_cases(rows: list[dict], thresholds: dict | None = None) -> list[dict]:
    """根据默认阈值筛选 failure cases。

    这些规则用于调试和挑选定性图，不是最终论文判定标准。
    """

    thresholds = thresholds or {
        "AttrAcc": 0.5,
        "Leakage": 0.2,
        "InstIoU_mean": 0.3,
        "EditLocality": 0.5,
        "DeletionResidual": 0.3,
    }
    failures = []
    for row in rows:
        rules = []
        values = {}
        for key, threshold in thresholds.items():
            if key not in row or not is_finite_number(row.get(key)):
                continue
            value = float(row[key])
            values[key] = value
            if key in {"Leakage", "DeletionResidual"}:
                failed = value > float(threshold)
            else:
                failed = value < float(threshold)
            if failed:
                rules.append(key)
        if rules:
            failures.append(
                {
                    "image_id": row.get("image_id", ""),
                    "method": row.get("method", row.get("name", "")),
                    "failed_rules": rules,
                    "metric_values": values,
                }
            )
    return failures


def summarize_failures(failure_cases: list[dict]) -> dict:
    """按 method 和 failed_rule 统计 failure 数量。"""

    by_method = Counter(case.get("method", "") for case in failure_cases)
    by_rule = Counter(rule for case in failure_cases for rule in case.get("failed_rules", []))
    by_method_rule = defaultdict(int)
    for case in failure_cases:
        for rule in case.get("failed_rules", []):
            by_method_rule[f"{case.get('method', '')}:{rule}"] += 1
    return {
        "num_failures": len(failure_cases),
        "by_method": dict(by_method),
        "by_rule": dict(by_rule),
        "by_method_rule": dict(by_method_rule),
    }


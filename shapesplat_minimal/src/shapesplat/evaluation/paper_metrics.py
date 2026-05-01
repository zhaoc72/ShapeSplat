from __future__ import annotations


DEFAULT_PROTOCOL = {
    "metric_groups": {
        "object_consistency": ["InstIoU_mean", "AttrAcc", "AttrPurity_mean", "Leakage"],
        "editing": ["CollateralL1", "EditLocality", "DeletionResidual", "ObjectSupportIoU"],
        "stress": ["SwapRateProxy", "OrderAccProxy", "OcclusionRecallProxy", "TruncationStabilityProxy"],
        "rendering": ["ForegroundRGBL1", "ForegroundAlphaError"],
        "optional_geometry": ["ChamferL2", "FScore", "VisibleCD", "HiddenCD"],
    }
}


def collect_paper_metrics(metrics: dict, protocol: dict | None = None) -> dict:
    """按 paper metrics protocol 分组整理已有指标。"""
    groups = (protocol or DEFAULT_PROTOCOL).get("metric_groups", DEFAULT_PROTOCOL["metric_groups"])
    out = {}
    for group, keys in groups.items():
        out[group] = {key: metrics[key] for key in keys if key in metrics}
    return out


def validate_paper_metrics(metrics: dict, required_groups: list[str] | None = None) -> dict:
    """检查 paper table 前的指标完整性；geometry 默认可选。"""
    grouped = collect_paper_metrics(metrics)
    required_groups = required_groups or ["object_consistency"]
    missing: list[str] = []
    warnings: list[str] = []
    if "object_consistency" in required_groups:
        if not any(k in grouped["object_consistency"] for k in ["AttrAcc", "Leakage", "InstIoU_mean"]):
            missing.append("object_consistency: AttrAcc/Leakage/InstIoU_mean")
    if "editing" in required_groups and not any(k in grouped["editing"] for k in ["EditLocality", "CollateralL1"]):
        missing.append("editing: EditLocality/CollateralL1")
    if "stress" in required_groups and not grouped["stress"]:
        missing.append("stress metrics")
    if not grouped["optional_geometry"]:
        warnings.append("optional geometry metrics unavailable; this is allowed for minimal/stub experiments")
    return {"valid": len(missing) == 0, "missing": missing, "warnings": warnings}

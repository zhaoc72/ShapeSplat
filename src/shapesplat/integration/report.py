from __future__ import annotations

import json
from pathlib import Path


def make_capability_markdown(capabilities: dict) -> str:
    """生成 backend capability Markdown 表格，方便本地集成排查。"""

    lines = ["# Backend Capabilities", "", "| Backend | Requested | Available | Fallback | Warnings |", "| --- | --- | --- | --- | --- |"]
    for key in ["sam", "dino", "depth", "shape_bank", "renderer"]:
        cap = capabilities.get(key, {})
        fallback = cap.get("fallback_target") if cap.get("will_fallback") else ""
        warnings = "; ".join(cap.get("warnings", []))
        lines.append(f"| {key} | {cap.get('requested', '')} | {cap.get('available', False)} | {fallback or ''} | {warnings} |")
    ext = capabilities.get("external_baselines", {})
    lines.append(f"| external_baselines | config | {ext.get('available', True)} |  | {len(ext.get('entries', []))} entries |")
    return "\n".join(lines) + "\n"


def make_integration_report_markdown(report: dict) -> str:
    """生成 readable 集成报告；这是本地集成 smoke test 的主报告。"""

    lines = ["# Real Integration Smoke Report", "", f"Status: `{report.get('status')}`", ""]
    lines.append("## Backend Capabilities")
    lines.append(make_capability_markdown(report.get("capabilities", {})))
    lines.append("## Frontend Stats")
    for key, value in (report.get("frontend_stats") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Reconstruction Metrics")
    for key, value in (report.get("metrics") or {}).items():
        if key in {"image_id", "status", "output_dir"}:
            continue
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Cache")
    for key, value in (report.get("cache_paths") or {}).items():
        lines.append(f"- {key}: `{value}`")
    warnings = report.get("warnings") or []
    lines.append("")
    lines.append("## Warnings")
    lines.extend([f"- {w}" for w in warnings] or ["- none"])
    lines.append("")
    lines.append("## Next Steps")
    lines.append("- Fill local checkpoints/modules in the backend template if real components are available.")
    lines.append("- Keep auto mode for safe fallback while validating local dependencies.")
    lines.append("- This smoke test validates integration plumbing, not final paper quality.")
    return "\n".join(lines) + "\n"


def save_integration_report(report: dict, out_dir: str | Path) -> None:
    """保存 JSON/Markdown 集成报告和 capability 报告。"""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    caps = report.get("capabilities", {})
    (out / "integration_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "integration_report.md").write_text(make_integration_report_markdown(report), encoding="utf-8")
    (out / "backend_capabilities.json").write_text(json.dumps(caps, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "backend_capabilities.md").write_text(make_capability_markdown(caps), encoding="utf-8")

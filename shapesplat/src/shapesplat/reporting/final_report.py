from __future__ import annotations

from pathlib import Path

from shapesplat.reporting.io import load_json, save_json
from shapesplat.reporting.tables import make_markdown_table


def _load_rows(path: Path):
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, dict):
        return list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
    return data


def generate_final_report(final_root: str | Path, out_dir: str | Path, title: str = "ShapeSplat++ Final Paper Report") -> dict:
    """生成最终论文实验整理稿。

    该报告只汇总已有输出、表格和 readiness warning，不代表 debug profile 是投稿结果。
    """
    root = Path(final_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    readiness = root / "readiness" / "final_readiness.json"
    summary = root / "final_run_summary.json"
    comparison = root / "comparison" / "final_method_summary.json"
    variants = root / "variants" / "variant_summary.json"
    stress = root / "stress" / "stress_subset_summary.json"
    editing = root / "editing" / "edit_summary.json"
    table_files = sorted((root / "tables").glob("*.*")) if (root / "tables").exists() else []

    lines = [f"# {title}", "", "## Overview"]
    if summary.exists():
        s = load_json(summary)
        lines.append(f"- Run status: `{s.get('status')}`")
        lines.append(f"- Steps: `{len(s.get('steps', []))}`")
    if readiness.exists():
        r = load_json(readiness)
        lines += ["", "## Readiness", f"- Ready: `{r.get('ready')}`"]
        for w in r.get("warnings", []):
            lines.append(f"- Warning: {w}")
        for e in r.get("errors", []):
            lines.append(f"- Error: {e}")

    for heading, path in [
        ("Main Comparison", comparison),
        ("Variants / Ablation", variants),
        ("Stress Benchmark", stress),
        ("Editing Evaluation", editing),
    ]:
        rows = _load_rows(path)
        lines += ["", f"## {heading}", make_markdown_table(rows[:12]) if rows else "_Not found._"]

    lines += ["", "## Generated Tables"]
    for p in table_files:
        lines.append(f"- `{p.relative_to(root)}`")
    lines += [
        "",
        "## Notes",
        "- `final_debug` is a smoke test, not a submission result.",
        "- Images shown here may be thumbnails for quick browsing. Full-resolution and working-resolution outputs are stored in each per-image output directory.",
        "- Geometry metrics are only shown when pred/GT pointclouds exist.",
        "- Missing real external baselines are readiness warnings, not runtime failures.",
    ]

    report = out / "final_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    manifest = {"report": str(report), "tables": [str(p) for p in table_files]}
    save_json(manifest, out / "final_report_manifest.json")
    return manifest

from __future__ import annotations

from pathlib import Path

from shapesplat.reporting.diagnostics import (
    detect_failure_cases,
    metric_sanity_check,
    select_best_worst_cases,
    summarize_failures,
)
from shapesplat.reporting.io import find_experiment_outputs, load_json, save_csv_rows, save_json
from shapesplat.reporting.latex import save_latex_table
from shapesplat.reporting.qualitative import (
    find_qualitative_grids,
    make_qualitative_markdown,
    make_selected_cases_markdown,
)
from shapesplat.reporting.tables import (
    ABLATION_COLUMNS,
    COMPARISON_COLUMNS,
    STRESS_COLUMNS,
    flatten_summary,
    make_markdown_table,
    select_table_columns,
)


def _rows_from_output(path: Path) -> list[dict]:
    data = load_json(path)
    return data if isinstance(data, list) else flatten_summary(data)


def _write_table(rows: list[dict], columns: list[str], csv_path: Path, tex_path: Path, caption: str, label: str) -> str:
    selected = select_table_columns(rows, columns)
    save_csv_rows(selected, csv_path)
    save_latex_table(selected, columns, caption, label, tex_path)
    return make_markdown_table(selected, columns)


def generate_experiment_report(
    experiment_root: str | Path,
    out_dir: str | Path,
    title: str = "ShapeSplat++ Experiment Report",
) -> dict:
    """生成实验报告。

    该函数只做结果整理和诊断，不改变任何算法、renderer 或 backend。
    """

    root = Path(experiment_root)
    out = Path(out_dir)
    tables_dir = out / "tables"
    diagnostics_dir = out / "diagnostics"
    qualitative_dir = out / "qualitative"
    for d in (tables_dir, diagnostics_dir, qualitative_dir):
        d.mkdir(parents=True, exist_ok=True)

    found = find_experiment_outputs(root)
    manifest: dict = {"experiment_root": str(root), "out_dir": str(out), "found_outputs": {k: str(v) for k, v in found.items() if not isinstance(v, list)}}
    report_lines = [f"# {title}", "", f"Experiment root: `{root}`", ""]

    report_lines.append("## Found Outputs")
    if found:
        for key, path in found.items():
            if isinstance(path, list):
                report_lines.append(f"- `{key}`: {len(path)} files")
            else:
                report_lines.append(f"- `{key}`: `{path}`")
    else:
        report_lines.append("_No known experiment outputs found._")
    report_lines.append("")

    per_image_rows: list[dict] = []
    if "comparison_summary" in found:
        rows = flatten_summary(load_json(found["comparison_summary"]))
        md = _write_table(
            rows,
            COMPARISON_COLUMNS,
            tables_dir / "comparison_table.csv",
            tables_dir / "comparison_table.tex",
            "Comparison on the same-mask setting.",
            "tab:same_mask_comparison",
        )
        report_lines.extend(["## Comparison Summary", md, ""])
        manifest["comparison_table"] = str(tables_dir / "comparison_table.csv")
    if "ablation_summary" in found:
        rows = _rows_from_output(found["ablation_summary"])
        md = _write_table(
            rows,
            ABLATION_COLUMNS,
            tables_dir / "ablation_table.csv",
            tables_dir / "ablation_table.tex",
            "Ablation study summary.",
            "tab:ablation_summary",
        )
        report_lines.extend(["## Ablation Summary", md, ""])
        manifest["ablation_table"] = str(tables_dir / "ablation_table.csv")
    if "stress_subset_summary" in found:
        rows = _rows_from_output(found["stress_subset_summary"])
        md = _write_table(
            rows,
            STRESS_COLUMNS,
            tables_dir / "stress_table.csv",
            tables_dir / "stress_table.tex",
            "Synthetic stress benchmark subset summary.",
            "tab:stress_benchmark",
        )
        report_lines.extend(["## Stress Benchmark Summary", md, ""])
        manifest["stress_table"] = str(tables_dir / "stress_table.csv")
    if "per_image_comparison" in found:
        per_image_rows = _rows_from_output(found["per_image_comparison"])
    elif "stress_per_image" in found:
        per_image_rows = _rows_from_output(found["stress_per_image"])
    elif "baseline_summary" in found:
        per_image_rows = _rows_from_output(found["baseline_summary"])
    elif "metrics_files" in found:
        per_image_rows = [load_json(p) for p in found["metrics_files"] if p.exists()]

    sanity = metric_sanity_check(per_image_rows)
    save_json(sanity, diagnostics_dir / "metric_sanity.json")
    best_worst = select_best_worst_cases(per_image_rows, "AttrAcc", higher_is_better=True, top_k=5) if per_image_rows else {"best": [], "worst": []}
    failures = detect_failure_cases(per_image_rows)
    failure_summary = summarize_failures(failures)
    save_json(best_worst["best"], diagnostics_dir / "best_cases.json")
    save_json(best_worst["worst"], diagnostics_dir / "worst_cases.json")
    save_json(failures, diagnostics_dir / "failure_cases.json")
    save_json(failure_summary, diagnostics_dir / "failure_summary.json")

    grids = find_qualitative_grids(root)
    make_qualitative_markdown(grids, qualitative_dir / "qualitative_index.md")
    make_selected_cases_markdown(failures[:10], root, qualitative_dir / "selected_failure_cases.md")

    report_lines.extend(
        [
            "## Metric Sanity Check",
            f"- rows: {sanity['num_rows']}",
            f"- bad rows: {sanity['num_bad_rows']}",
            "",
            "## Best Cases",
            make_markdown_table(best_worst["best"][:5]) if best_worst["best"] else "_No best cases available._",
            "",
            "## Worst Cases",
            make_markdown_table(best_worst["worst"][:5]) if best_worst["worst"] else "_No worst cases available._",
            "",
            "## Failure Summary",
            "```json",
            str(failure_summary),
            "```",
            "",
            "## Qualitative Results",
            f"- [Qualitative index](qualitative/qualitative_index.md)",
            f"- [Selected failure cases](qualitative/selected_failure_cases.md)",
            "",
        ]
    )

    report_path = out / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    manifest.update(
        {
            "report": str(report_path),
            "metric_sanity": str(diagnostics_dir / "metric_sanity.json"),
            "failure_cases": str(diagnostics_dir / "failure_cases.json"),
            "qualitative_index": str(qualitative_dir / "qualitative_index.md"),
            "num_failure_cases": len(failures),
            "num_bad_metric_rows": sanity["num_bad_rows"],
        }
    )
    save_json(manifest, out / "report_manifest.json")
    return manifest

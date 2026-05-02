from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.reporting.diagnostics import metric_sanity_check, select_best_worst_cases
from shapesplat.reporting.io import save_json
from shapesplat.reporting.latex import make_latex_table
from shapesplat.reporting.report import generate_experiment_report
from shapesplat.reporting.tables import make_markdown_table


@pytest.fixture
def tmp_path(request) -> Path:
    root = ROOT / "outputs" / "test_reporting_tmp" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_markdown_table() -> None:
    rows = [{"method": "ours", "AttrAcc_mean": 0.81234}]
    md = make_markdown_table(rows, ["method", "AttrAcc_mean"])
    assert "| method | AttrAcc_mean |" in md
    assert "0.8123" in md


def test_latex_table() -> None:
    tex = make_latex_table([{"method": "ours", "AttrAcc": 0.9}], ["method", "AttrAcc"], "Caption", "tab:test")
    assert "\\begin{table}" in tex
    assert "\\caption{Caption}" in tex


def test_metric_sanity_check() -> None:
    rows = [{"AttrAcc": 0.9, "Leakage": 0.1}, {"AttrAcc": 1.2, "Leakage": float("inf")}]
    result = metric_sanity_check(rows)
    assert result["num_bad_rows"] == 1


def test_best_worst_cases() -> None:
    rows = [{"image_id": "a", "AttrAcc": 0.2}, {"image_id": "b", "AttrAcc": 0.8}]
    result = select_best_worst_cases(rows, "AttrAcc", higher_is_better=True, top_k=1)
    assert result["best"][0]["image_id"] == "b"
    assert result["worst"][0]["image_id"] == "a"


def test_generate_report_from_fake_outputs(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    save_json(
        {
            "ours": {"method": "ours", "num_success": 2, "num_failed": 0, "AttrAcc_mean": 0.8},
            "dummy": {"method": "dummy", "num_success": 2, "num_failed": 0, "AttrAcc_mean": 0.5},
        },
        root / "per_method_summary.json",
    )
    save_json(
        [
            {"image_id": "a", "method": "ours", "status": "success", "AttrAcc": 0.8, "Leakage": 0.1, "InstIoU_mean": 0.5},
            {"image_id": "b", "method": "dummy", "status": "success", "AttrAcc": 0.4, "Leakage": 0.3, "InstIoU_mean": 0.2},
        ],
        root / "per_image_comparison.json",
    )
    manifest = generate_experiment_report(root, tmp_path / "report", "Fake Report")
    assert Path(manifest["report"]).exists()
    assert (tmp_path / "report" / "report_manifest.json").exists()
    assert (tmp_path / "report" / "diagnostics" / "metric_sanity.json").exists()
    assert (tmp_path / "report" / "diagnostics" / "failure_cases.json").exists()


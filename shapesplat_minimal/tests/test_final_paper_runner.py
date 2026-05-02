from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.final_paper_runner import load_final_profile, run_final_paper_experiment
from shapesplat.experiments.final_readiness import check_final_paper_ready, save_final_readiness_report
from shapesplat.reporting.all_final_tables import export_all_final_tables
from shapesplat.reporting.final_report import generate_final_report


@pytest.fixture
def tmp_path(request):
    root = ROOT / "outputs" / "test_final_paper_runner_tmp" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_load_final_profile():
    profile = load_final_profile("configs/paper/final_debug.yaml")
    assert profile["profile"] == "final_debug"


def test_check_final_paper_ready_debug(tmp_path):
    profile = load_final_profile("configs/paper/final_debug.yaml")
    report = check_final_paper_ready(profile, tmp_path, strict=False)
    assert "warnings" in report
    assert "errors" in report
    assert "checks" in report


def test_run_final_paper_dry_run(tmp_path):
    profile = load_final_profile("configs/paper/final_debug.yaml")
    summary = run_final_paper_experiment(profile, tmp_path, dry_run=True)
    assert summary["status"] == "dry_run"
    assert (tmp_path / "final_plan.json").exists()
    assert (tmp_path / "command_log.json").exists()


def test_save_final_readiness_report(tmp_path):
    report = {"ready": True, "strict": False, "errors": [], "warnings": ["debug"], "checks": [{"name": "x", "ok": True, "message": "ok"}], "summary": {}}
    save_final_readiness_report(report, tmp_path)
    assert (tmp_path / "final_readiness.json").exists()
    assert (tmp_path / "final_readiness.md").exists()


def test_generate_final_report_fake(tmp_path):
    (tmp_path / "comparison").mkdir()
    (tmp_path / "stress").mkdir()
    (tmp_path / "editing").mkdir()
    (tmp_path / "comparison" / "final_method_summary.json").write_text(json.dumps([{"method": "ours_full", "AttrAcc_mean": 1.0}]), encoding="utf-8")
    (tmp_path / "stress" / "stress_subset_summary.json").write_text(json.dumps([{"subset": "normal", "num_images": 1}]), encoding="utf-8")
    (tmp_path / "editing" / "edit_summary.json").write_text(json.dumps([{"op": "remove", "num_edits": 1}]), encoding="utf-8")
    manifest = generate_final_report(tmp_path, tmp_path / "report")
    assert Path(manifest["report"]).exists()


def test_export_all_final_tables_fake(tmp_path):
    (tmp_path / "comparison").mkdir()
    (tmp_path / "stress").mkdir()
    (tmp_path / "editing").mkdir()
    (tmp_path / "comparison" / "final_method_summary.json").write_text(json.dumps([{"method": "ours_full", "family": "ours", "num_success": 1, "AttrAcc_mean": 1.0}]), encoding="utf-8")
    (tmp_path / "stress" / "stress_subset_summary.json").write_text(json.dumps([{"subset": "normal", "num_images": 1}]), encoding="utf-8")
    (tmp_path / "editing" / "edit_summary.json").write_text(json.dumps([{"op": "remove", "num_edits": 1}]), encoding="utf-8")
    export_all_final_tables(tmp_path, tmp_path / "tables")
    assert (tmp_path / "tables" / "main_comparison.csv").exists()
    assert (tmp_path / "tables" / "stress.tex").exists()

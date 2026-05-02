from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import torch

from shapesplat.evaluation.geometry_metrics import chamfer_l2, compute_geometry_metrics, fscore
from shapesplat.evaluation.paper_metrics import collect_paper_metrics
from shapesplat.experiments.paper_readiness import check_paper_ready
from shapesplat.experiments.paper_runner import load_paper_profile, run_paper_profile
from shapesplat.reporting.paper_tables import export_paper_table

ROOT = Path(__file__).resolve().parents[1]


def _tmp_dir(name: str) -> Path:
    path = ROOT / "outputs" / "test_paper_experiment_pack_tmp" / f"{name}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_load_paper_profile() -> None:
    profile = load_paper_profile("configs/paper/debug.yaml")
    assert profile["profile"] == "debug"
    assert profile["run_main_comparison"]


def test_check_paper_ready_debug() -> None:
    out = _tmp_dir("ready")
    result = check_paper_ready("configs/paper/debug.yaml", out)
    assert "warnings" in result
    assert "errors" in result
    assert result["ready"]


def test_geometry_metrics_optional() -> None:
    out = _tmp_dir("geometry")
    pred = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gt = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    assert float(chamfer_l2(pred, gt)) >= 0.0
    assert 0.0 <= float(fscore(pred, gt, threshold=0.2)) <= 1.0
    result = compute_geometry_metrics(out / "missing_pred.npy", out / "missing_gt.npy")
    assert result["available"] is False


def test_paper_metrics_collect() -> None:
    grouped = collect_paper_metrics({"AttrAcc": 0.8, "Leakage": 0.1, "EditLocality": 0.9})
    assert "object_consistency" in grouped
    assert grouped["object_consistency"]["AttrAcc"] == 0.8


def test_export_paper_table() -> None:
    out = _tmp_dir("table")
    rows = [{"method": "ours", "num_success": 2, "AttrAcc_mean": 0.9, "Leakage_mean": 0.05}]
    export_paper_table(rows, "main_comparison", out / "table.csv", out / "table.tex", "Caption", "tab:test")
    assert (out / "table.csv").exists()
    assert (out / "table.tex").exists()


def test_run_paper_debug_dry_run() -> None:
    out = _tmp_dir("dry")
    profile = load_paper_profile("configs/paper/debug.yaml")
    summary = run_paper_profile(profile, out, dry_run=True)
    assert summary["status"] == "dry_run"
    assert (out / "paper_plan.json").exists()
    assert (out / "command_log.json").exists()


def test_run_paper_experiments_script_dry_run() -> None:
    out = _tmp_dir("script_dry")
    result = subprocess.run(
        [sys.executable, "scripts/run_paper_experiments.py", "--profile", "debug", "--out", str(out), "--dry-run", "--no-run-metadata"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "paper_run_summary.json").exists()

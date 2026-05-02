from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

from shapesplat.experiments.orchestrator import ExperimentPlan, ExperimentStep, load_preset, resolve_placeholders, run_experiment_plan
from shapesplat.experiments.readiness import check_experiment_ready


def _tmp_dir(name: str) -> Path:
    path = Path("outputs") / "test_orchestrator_tmp" / f"{name}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_load_preset() -> None:
    plan = load_preset("configs/presets/comparison.yaml")
    assert plan.name == "comparison"
    assert len(plan.steps) >= 2


def test_resolve_placeholders() -> None:
    cmd = ["python", "x.py", "--out", "{out}", "--preset", "{preset}"]
    out = resolve_placeholders(cmd, {"out": "outputs/x", "preset": "minimal"})
    assert out[3] == "outputs/x"
    assert out[5] == "minimal"


def test_dry_run_experiment_plan() -> None:
    out = _tmp_dir("dry")
    plan = ExperimentPlan(
        name="dry",
        description="dry run",
        default_out=str(out),
        steps=[ExperimentStep(name="hello", enabled=True, command=[sys.executable, "-c", "print('hi')"])],
    )
    summary = run_experiment_plan(plan, out, {"project_root": str(Path.cwd()), "out": str(out)}, dry_run=True)
    assert summary["status"] == "dry_run"
    assert (out / "command_log.json").exists()
    assert summary["steps"][0]["status"] == "dry_run"


def test_run_experiment_plan_success() -> None:
    out = _tmp_dir("success")
    plan = ExperimentPlan(
        name="success",
        description="real command",
        default_out=str(out),
        steps=[ExperimentStep(name="ok", enabled=True, command=[sys.executable, "-c", "print('ok')"])],
    )
    summary = run_experiment_plan(plan, out, {"project_root": str(Path.cwd()), "out": str(out)}, dry_run=False)
    assert summary["status"] == "success"
    assert (out / "logs" / "ok_stdout.txt").exists()
    assert "ok" in (out / "logs" / "ok_stdout.txt").read_text(encoding="utf-8")


def test_check_experiment_ready() -> None:
    out = _tmp_dir("ready")
    result = check_experiment_ready("configs/presets/comparison.yaml", out, {"out": str(out), "preset": "comparison"})
    assert "ready" in result
    assert "checks" in result


def test_list_presets_script_like() -> None:
    assert len(list(Path("configs/presets").glob("*.yaml"))) >= 5


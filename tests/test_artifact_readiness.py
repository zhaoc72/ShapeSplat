from __future__ import annotations

import zipfile
from pathlib import Path
from uuid import uuid4

from shapesplat.validation.artifact import create_artifact_package, make_artifact_manifest
from shapesplat.validation.command_matrix import load_command_matrix, run_command_matrix
from shapesplat.validation.project_health import check_project_health

ROOT = Path(__file__).resolve().parents[1]


def _tmp_dir(name: str) -> Path:
    path = ROOT / "outputs" / "test_artifact_readiness_tmp" / f"{name}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_command_matrix_loads() -> None:
    commands = load_command_matrix("configs/command_matrix.yaml")
    names = {entry["name"] for entry in commands}
    assert "import_check" in names
    assert "minimal" in names


def test_project_health() -> None:
    result = check_project_health()
    assert "healthy" in result
    assert "checks" in result
    assert result["healthy"]


def test_artifact_manifest() -> None:
    manifest = make_artifact_manifest(".")
    paths = [item["path"] for item in manifest["files"]]
    assert manifest["num_files"] > 0
    assert any(path.startswith("src/") for path in paths)
    assert any(path.startswith("configs/") for path in paths)
    assert any(path.startswith("scripts/") for path in paths)


def test_package_artifact_tmp() -> None:
    out = _tmp_dir("package") / "artifact.zip"
    path = create_artifact_package(out, include_docs=True, include_tests=True, include_examples=True)
    assert path.exists()
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
    assert names
    assert not any(name.startswith("outputs/") for name in names)
    assert not any(name.startswith("runs/") for name in names)


def test_validate_artifact_dry_run() -> None:
    out = _tmp_dir("dry")
    rows = run_command_matrix(
        "configs/command_matrix.yaml",
        groups=["quick"],
        dry_run=True,
        context={"validation_out": str(out)},
    )
    assert rows
    assert all(row["status"] == "dry_run" for row in rows)
    assert (out / "command_results.json").exists()

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from shapesplat.reproducibility.environment import collect_environment_info
from shapesplat.reproducibility.finalize import finalize_run_outputs
from shapesplat.reproducibility.hashing import hash_file
from shapesplat.reproducibility.registry import append_run_registry, load_run_registry
from shapesplat.reproducibility.run_info import create_run_info, generate_run_id
from shapesplat.reproducibility.snapshot import index_output_files
from shapesplat.utils.metrics_summary import extract_metrics_summary


def _case_dir(name: str) -> Path:
    root = Path("outputs") / "test_reproducibility_tmp" / f"{name}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_generate_run_id() -> None:
    run_id = generate_run_id("unit")
    assert isinstance(run_id, str)
    assert run_id.startswith("unit_")


def test_collect_environment_info() -> None:
    info = collect_environment_info()
    assert isinstance(info, dict)
    assert "python_version" in info
    assert "platform" in info
    assert "torch_version" in info


def test_hash_file() -> None:
    tmp_path = _case_dir("hash")
    path = tmp_path / "a.txt"
    path.write_text("shape", encoding="utf-8")
    # hash 是结果完整性检查的基础，同一文件内容应得到稳定值。
    assert hash_file(path) == hash_file(path)


def test_output_index() -> None:
    tmp_path = _case_dir("index")
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"fake")
    index = index_output_files(tmp_path)
    assert index["num_files"] >= 2
    assert any(item["path"] == "metrics.json" for item in index["files"])


def test_run_registry() -> None:
    tmp_path = _case_dir("registry")
    registry = tmp_path / "runs" / "run_registry.jsonl"
    info = create_run_info("unit", tmp_path / "out", command="pytest")
    append_run_registry(info, registry_path=registry, metrics_summary={"AttrAcc": 1.0})
    rows = load_run_registry(registry)
    assert len(rows) == 1
    assert rows[0]["run_id"] == info.run_id
    assert rows[0]["metrics_summary"]["AttrAcc"] == 1.0


def test_extract_metrics_summary() -> None:
    tmp_path = _case_dir("metrics")
    metrics = {"AttrAcc": 0.7, "Leakage": 0.1, "InstIoU_mean": 0.5}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    summary = extract_metrics_summary(tmp_path)
    assert summary["AttrAcc"] == 0.7
    assert summary["Leakage"] == 0.1


def test_finalize_run_script_like() -> None:
    tmp_path = _case_dir("finalize")
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("seed: 123\nimage:\n  size: 32\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({"AttrAcc": 0.9}), encoding="utf-8")
    registry = tmp_path / "runs" / "run_registry.jsonl"

    result = finalize_run_outputs(
        out_dir=out_dir,
        config_path=cfg_path,
        run_type="unit",
        registry_path=registry,
        command="pytest finalize",
    )

    assert result["run_id"].startswith("unit_")
    assert (out_dir / "run_info.json").exists()
    assert (out_dir / "environment.json").exists()
    assert (out_dir / "output_index.json").exists()
    assert (out_dir / "metrics_summary.json").exists()
    assert load_run_registry(registry)

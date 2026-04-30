from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from create_example_dataset import create_example_dataset
from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.datasets.image_dataset import build_dataset_from_manifest
from shapesplat.datasets.manifest import load_manifest
from shapesplat.experiments.batch_runner import run_batch_experiment
from shapesplat.experiments.single_image import run_single_image_experiment
from shapesplat.experiments.summary import save_batch_summary


@pytest.fixture
def tmp_path(request):
    """Windows 临时目录偶尔权限受限，这里用项目 outputs 下的测试目录。"""
    root = ROOT / "outputs" / "test_dataset_runner_tmp" / request.node.name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fast_cfg(size: int = 32):
    cfg = merge_config(
        DEFAULT_CONFIG,
        {
            "device": "cpu",
            "image": {"size": size},
            "gaussians": {"visible_min": 8, "visible_max": 16, "hidden_base": 4},
            "training": {
                "visible_warmup_iters": 1,
                "hidden_prior_iters": 1,
                "joint_ownership_iters": 1,
                "edit_finetune_iters": 1,
            },
        },
    )
    cfg["device"] = "cpu"
    return cfg


def test_create_example_dataset(tmp_path):
    out = tmp_path / "dataset"
    create_example_dataset(out, num_images=3, size=64)
    assert (out / "manifest.csv").exists()
    assert len(list((out / "images").glob("*.png"))) == 3


def test_load_manifest(tmp_path):
    out = tmp_path / "dataset"
    create_example_dataset(out, num_images=2, size=64)
    records = load_manifest(out / "manifest.csv")
    assert len(records) == 2
    assert Path(records[0].image_path).exists()
    assert records[0].metadata.get("category") == "toy"


def test_image_dataset(tmp_path):
    out = tmp_path / "dataset"
    create_example_dataset(out, num_images=2, size=64)
    dataset = build_dataset_from_manifest(out / "manifest.csv", image_size=32)
    item = dataset[0]
    assert item["image"].shape == (3, 32, 32)
    assert item["image_id"].startswith("example_")


def test_run_single_image_experiment(tmp_path):
    cfg = _fast_cfg(32)
    out = tmp_path / "single"
    row = run_single_image_experiment(make_synthetic_image(32), cfg, out, image_id="single", save_checkpoint=False)
    assert row["image_id"] == "single"
    assert row["status"] == "success"
    assert "AttrAcc" in row
    assert (out / "metrics.json").exists()


def test_run_batch_experiment_two_images(tmp_path):
    dataset_dir = tmp_path / "dataset"
    create_example_dataset(dataset_dir, num_images=2, size=64)
    dataset = build_dataset_from_manifest(dataset_dir / "manifest.csv", image_size=32)
    cfg = _fast_cfg(32)
    out = tmp_path / "batch"
    rows = run_batch_experiment(dataset, cfg, out, max_images=2, save_checkpoint=False)
    summary = save_batch_summary(rows, out)
    assert len(rows) == 2
    assert summary["num_success"] >= 1
    assert (out / "summary.json").exists()
    assert (out / "per_image_metrics.csv").exists()

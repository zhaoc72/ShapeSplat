from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import pytest

from shapesplat.baselines.dummy_baselines import (
    identity_mask_baseline,
    independent_blob_baseline,
    scene_union_baseline,
)
from shapesplat.baselines.evaluate_baseline import evaluate_baseline_prediction
from shapesplat.baselines.export_inputs import export_baseline_inputs
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.datasets.example_dataset import create_example_dataset
from shapesplat.datasets.manifest import load_manifest


@pytest.fixture
def tmp_path(request) -> Path:
    """Windows 本机临时目录可能被权限策略锁住，测试改用项目 outputs 下的临时目录。"""

    path = ROOT / "outputs" / "test_baseline_protocol_tmp" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _simple_masks(size: int = 32) -> torch.Tensor:
    masks = torch.zeros(2, size, size)
    masks[0, 5:18, 5:18] = 1
    masks[1, 14:27, 14:27] = 1
    return masks


def test_export_baseline_inputs(tmp_path: Path) -> None:
    image = make_synthetic_image(32)
    masks = _simple_masks(32)
    spec = export_baseline_inputs(image, masks, tmp_path / "inputs", "toy_000", crop_padding=4)
    assert Path(spec.image_path).exists()
    assert Path(spec.masks_path).exists()
    assert Path(spec.metadata_path).exists()
    assert (tmp_path / "inputs" / "crops" / "object_000_rgb.png").exists()
    assert (tmp_path / "inputs" / "crops" / "object_000_rgba.png").exists()


def test_dummy_baselines_outputs() -> None:
    image = make_synthetic_image(32)
    masks = _simple_masks(32)
    for fn in [identity_mask_baseline, independent_blob_baseline, scene_union_baseline]:
        pred = fn(image, masks)
        assert pred["alpha"].shape == (32, 32)
        assert pred["ownership"].shape == masks.shape
        metrics = evaluate_baseline_prediction(pred, masks, image=image)
        assert "AttrAcc" in metrics
        assert "InstIoU_mean" in metrics
        assert "Leakage" in metrics


def test_evaluate_baseline_prediction() -> None:
    image = make_synthetic_image(32)
    masks = _simple_masks(32)
    pred = identity_mask_baseline(image, masks)
    metrics = evaluate_baseline_prediction(pred, masks, image=image)
    assert metrics["AttrAcc"] >= 0.0
    assert metrics["InstIoU_mean"] >= 0.0
    assert metrics["Leakage"] >= 0.0


def test_run_dummy_baselines_script(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    manifest = create_example_dataset(dataset_dir, num_images=1, size=64)
    record = load_manifest(manifest)[0]
    out_dir = tmp_path / "dummy"
    cmd = [
        sys.executable,
        "scripts/run_dummy_baselines.py",
        "--config",
        "configs/same_mask.yaml",
        "--input",
        record.image_path,
        "--mask",
        record.metadata["mask_path"],
        "--out",
        str(out_dir),
        "--image-id",
        record.image_id,
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "comparison.json").exists()
    assert (out_dir / "identity_mask" / "ownership.npy").exists()


def test_baseline_dataset_runner(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    manifest = create_example_dataset(dataset_dir, num_images=2, size=64)
    out_dir = tmp_path / "baseline_dataset"
    cmd = [
        sys.executable,
        "scripts/run_baseline_dataset.py",
        "--config",
        "configs/baseline_protocol.yaml",
        "--manifest",
        str(manifest),
        "--out",
        str(out_dir),
        "--max-images",
        "2",
        "--run-dummy",
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "baseline_summary.json").exists()
    assert (out_dir / "baseline_summary.csv").exists()

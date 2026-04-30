from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.datasets.example_dataset import create_example_dataset
from shapesplat.datasets.image_dataset import build_dataset_from_manifest
from shapesplat.experiments.comparison_runner import run_comparison_dataset, run_comparison_for_image
from shapesplat.experiments.comparison_summary import save_comparison_summary, summarize_comparison_rows
from shapesplat.utils.comparison_visualization import make_comparison_grid


@pytest.fixture
def tmp_path(request) -> Path:
    root = ROOT / "outputs" / "test_comparison_runner_tmp" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fast_cfg(size: int = 32) -> dict:
    return merge_config(
        DEFAULT_CONFIG,
        {
            "device": "cpu",
            "image": {"size": size},
            "frontend": {"mask_source": "file", "sam_backend": "stub", "dino_backend": "stub", "depth_backend": "stub"},
            "gaussians": {"visible_min": 8, "visible_max": 16, "hidden_base": 4},
            "training": {
                "visible_warmup_iters": 1,
                "hidden_prior_iters": 1,
                "joint_ownership_iters": 1,
                "edit_finetune_iters": 1,
            },
        },
    )


def _masks(size: int = 32) -> torch.Tensor:
    masks = torch.zeros(2, size, size)
    masks[0, 5:18, 4:17] = 1
    masks[1, 14:28, 15:28] = 1
    return masks


def test_run_comparison_for_image(tmp_path: Path) -> None:
    image = make_synthetic_image(32)
    masks = _masks(32)
    rows = run_comparison_for_image(image, masks, _fast_cfg(32), tmp_path / "cmp", "toy", save_checkpoint=False)
    methods = {r["method"] for r in rows}
    assert "ours" in methods
    assert "identity_mask" in methods
    assert (tmp_path / "cmp" / "comparison.json").exists()
    assert (tmp_path / "cmp" / "qualitative_grid.png").exists()


def test_comparison_dataset_two_images(tmp_path: Path) -> None:
    manifest = create_example_dataset(tmp_path / "dataset", num_images=2, size=64)
    dataset = build_dataset_from_manifest(manifest, image_size=32)
    out = tmp_path / "comparison_dataset"
    rows = run_comparison_dataset(dataset, _fast_cfg(32), out, max_images=2, save_checkpoint=False)
    summary = save_comparison_summary(rows, out)
    assert "ours" in summary
    assert (out / "per_method_summary.json").exists()
    assert (out / "per_image_comparison.csv").exists()


def test_print_comparison_table_data() -> None:
    rows = [
        {"image_id": "a", "method": "ours", "status": "success", "AttrAcc": 0.8, "InstIoU_mean": 0.4},
        {"image_id": "b", "method": "ours", "status": "success", "AttrAcc": 1.0, "InstIoU_mean": 0.6},
        {"image_id": "a", "method": "dummy", "status": "failed", "error": "x"},
    ]
    summary = summarize_comparison_rows(rows)
    assert summary["ours"]["AttrAcc_mean"] == pytest.approx(0.9)
    assert summary["ours"]["InstIoU_mean_mean"] == pytest.approx(0.5)
    assert summary["dummy"]["num_failed"] == 1


def test_qualitative_grid(tmp_path: Path) -> None:
    image = make_synthetic_image(32)
    masks = _masks(32)
    outputs = {
        "identity": {
            "rgb": image,
            "alpha": masks.amax(dim=0),
            "ownership": masks / masks.sum(dim=0, keepdim=True).clamp_min(1e-6),
        }
    }
    path = tmp_path / "grid.png"
    make_comparison_grid(image, masks, outputs, path, title="toy")
    assert path.exists()


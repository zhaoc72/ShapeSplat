from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from create_example_dataset import create_example_dataset
from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.datasets.image_dataset import build_dataset_from_manifest
from shapesplat.datasets.manifest import load_manifest
from shapesplat.experiments.batch_runner import run_batch_experiment
from shapesplat.experiments.summary import save_batch_summary
from shapesplat.frontend.file_mask_loader import load_mask_file
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.optimization.losses import compute_losses
from shapesplat.renderer.backend import build_renderer


@pytest.fixture
def tmp_path(request):
    root = ROOT / "outputs" / "test_file_masks_tmp" / request.node.name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cfg(size: int = 32, overrides=None):
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
    if overrides:
        cfg = merge_config(cfg, overrides)
    cfg["device"] = "cpu"
    return cfg


def test_load_npy_stack_masks(tmp_path):
    masks = np.zeros((2, 32, 32), dtype=np.uint8)
    masks[0, 4:16, 4:16] = 1
    masks[1, 18:28, 18:30] = 1
    path = tmp_path / "masks.npy"
    np.save(path, masks)
    ms = load_mask_file(path, (32, 32), _cfg(32))
    assert ms.masks.shape == (2, 32, 32)
    assert ms.boxes.shape == (2, 4)


def test_load_label_png_masks(tmp_path):
    label = np.zeros((32, 32), dtype=np.uint8)
    label[3:12, 3:12] = 1
    label[18:28, 20:30] = 2
    path = tmp_path / "labels.png"
    Image.fromarray(label, mode="L").save(path)
    ms = load_mask_file(path, (32, 32), _cfg(32))
    assert ms.masks.shape[0] == 2


def test_run_minimal_with_file_mask(tmp_path):
    ds = tmp_path / "dataset"
    create_example_dataset(ds, num_images=1, size=64)
    img = build_dataset_from_manifest(ds / "manifest.csv", image_size=32)[0]["image"]
    mask_path = ds / "masks" / "example_000.npy"
    cfg = _cfg(32, {"frontend": {"mask_source": "file", "mask_path": str(mask_path)}})
    front = build_frontend(img, cfg)
    scene = initialize_scene(front, cfg)
    renderer = build_renderer(front.camera, cfg)
    render = renderer(scene)
    loss, _ = compute_losses(scene, renderer, render, front, cfg, stage="visible")
    loss.backward()
    assert torch.isfinite(loss)


def test_dataset_manifest_with_mask_path(tmp_path):
    ds = tmp_path / "dataset"
    create_example_dataset(ds, num_images=2, size=64)
    records = load_manifest(ds / "manifest.csv")
    assert "mask_path" in records[0].metadata
    assert Path(records[0].metadata["mask_path"]).exists()


def test_batch_runner_with_file_masks(tmp_path):
    ds = tmp_path / "dataset"
    create_example_dataset(ds, num_images=2, size=64)
    dataset = build_dataset_from_manifest(ds / "manifest.csv", image_size=32)
    cfg = _cfg(32, {"frontend": {"mask_source": "file"}})
    out = tmp_path / "batch"
    rows = run_batch_experiment(dataset, cfg, out, max_images=2, save_checkpoint=False)
    summary = save_batch_summary(rows, out)
    assert len(rows) == 2
    assert summary["num_success"] >= 1
    assert (out / "summary.json").exists()

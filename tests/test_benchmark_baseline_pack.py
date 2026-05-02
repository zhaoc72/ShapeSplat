from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch

from shapesplat.baselines.independent_gaussian import run_independent_gaussian_baseline
from shapesplat.baselines.templates.object_centric_templates import make_object_centric_command_template
from shapesplat.baselines.templates.scene_level_templates import make_scene_level_command_template
from shapesplat.benchmarks.standard.builder import build_same_mask_benchmark
from shapesplat.benchmarks.standard.validator import validate_benchmark_manifest
from shapesplat.config import load_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.datasets.example_dataset import create_example_dataset
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.renderer.backend import build_renderer
from shapesplat.renderer.contract import validate_render_output

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_path(request):
    root = ROOT / "outputs" / "test_benchmark_baseline_pack_tmp" / f"{request.node.name}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_validate_benchmark_manifest(tmp_path: Path):
    manifest = create_example_dataset(tmp_path / "dataset", num_images=2, size=64)
    report = validate_benchmark_manifest(manifest, load_config("configs/same_mask.yaml"))
    assert report["num_rows"] == 2
    assert report["num_valid"] > 0


def test_build_same_mask_benchmark(tmp_path: Path):
    manifest = create_example_dataset(tmp_path / "dataset", num_images=2, size=64)
    new_manifest = build_same_mask_benchmark(manifest, tmp_path / "standard", copy_files=True, overwrite=True)
    assert new_manifest.exists()
    report = validate_benchmark_manifest(new_manifest, load_config("configs/same_mask.yaml"))
    assert report["valid"]


def test_render_output_contract():
    cfg = load_config("configs/minimal.yaml")
    cfg["device"] = "cpu"
    cfg["image"]["size"] = 32
    image = make_synthetic_image(32)
    front = build_frontend(image, cfg)
    scene = initialize_scene(front, cfg)
    render = build_renderer(front.camera, cfg)(scene)
    result = validate_render_output(render, len(scene.objects), front.camera.height, front.camera.width)
    assert result["valid"]


def test_independent_gaussian_baseline(tmp_path: Path):
    cfg = load_config("configs/benchmark_baseline.yaml")
    image = make_synthetic_image(32)
    masks = torch.zeros(2, 32, 32)
    masks[0, 6:20, 6:18] = 1
    masks[1, 12:26, 16:28] = 1
    row = run_independent_gaussian_baseline(image, masks, cfg, tmp_path / "independent", image_id="toy")
    assert (tmp_path / "independent" / "render.png").exists()
    assert (tmp_path / "independent" / "alpha.png").exists()
    assert (tmp_path / "independent" / "ownership.npy").exists()
    assert (tmp_path / "independent" / "metrics.json").exists()
    assert "AttrAcc" in row
    assert "Leakage" in row


def test_baseline_templates():
    assert "spar3d" in make_object_centric_command_template("spar3d")
    assert "vggt" in make_scene_level_command_template("vggt")

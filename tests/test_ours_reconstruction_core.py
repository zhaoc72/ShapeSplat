from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import DEFAULT_CONFIG, load_config, merge_config
from shapesplat.data.image_io import load_image
from shapesplat.datasets.benchmark.builder_v2 import build_benchmark_from_existing_manifest
from shapesplat.datasets.example_dataset import create_example_dataset
from shapesplat.datasets.manifest import load_manifest
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.optimization.trainer import Trainer
from shapesplat.reconstruction.ours_runner import run_ours_benchmark, run_ours_single, run_ours_variants_benchmark
from shapesplat.reconstruction.output_protocol import save_ours_output
from shapesplat.reconstruction.readiness import check_ours_core_ready


@pytest.fixture
def tmp_path(request):
    root = ROOT / "outputs" / "test_ours_reconstruction_core_tmp" / request.node.name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fast_cfg(size: int = 32) -> dict:
    cfg = merge_config(
        DEFAULT_CONFIG,
        {
            "device": "cpu",
            "image": {"size": size},
            "frontend": {"mask_source": "file", "sam_backend": "stub", "dino_backend": "stub", "depth_backend": "stub"},
            "frontend_cache": {"use_cache": False},
            "renderer": {"backend": "soft"},
            "shape_bank": {"backend": "toy", "fallback_to_toy": True},
            "gaussians": {"visible_min": 8, "visible_max": 12, "hidden_base": 2},
            "training": {
                "visible_warmup_iters": 1,
                "hidden_prior_iters": 1,
                "joint_ownership_iters": 1,
                "edit_finetune_iters": 1,
                "log_every": 10,
            },
            "ours": {"variant": "full", "save_diagnostics": True, "save_visuals": True},
        },
    )
    cfg["device"] = "cpu"
    return cfg


def test_check_ours_core_ready(tmp_path):
    dataset_dir = tmp_path / "dataset"
    manifest = create_example_dataset(dataset_dir, num_images=1, size=32)
    cfg = load_config("configs/final_ours.yaml")
    report = check_ours_core_ready(cfg, manifest, strict=False)
    assert "warnings" in report
    assert "errors" in report
    assert "checks" in report


def test_run_ours_single(tmp_path):
    dataset_dir = tmp_path / "dataset"
    manifest = create_example_dataset(dataset_dir, num_images=1, size=32)
    record = load_manifest(manifest)[0]
    image = load_image(record.image_path, size=32)
    out = tmp_path / "ours_single"
    row = run_ours_single(image, _fast_cfg(32), out, image_id=record.image_id, record=record, save_checkpoint=False)
    assert row["status"] == "success"
    assert (out / "metrics.json").exists()
    assert (out / "ownership.npy").exists()
    assert (out / "diagnostics.json").exists()


def test_run_ours_benchmark_small(tmp_path):
    dataset_dir = tmp_path / "dataset"
    source_manifest = create_example_dataset(dataset_dir, num_images=2, size=32)
    bench_manifest = build_benchmark_from_existing_manifest(source_manifest, tmp_path / "benchmark", copy_files=True, overwrite=True)
    out = tmp_path / "ours_benchmark"
    rows = run_ours_benchmark(bench_manifest, _fast_cfg(32), out, max_images=2, use_frontend_cache=False)
    assert len(rows) == 2
    assert (out / "ours_summary.json").exists()


def test_run_ours_variants_small(tmp_path):
    dataset_dir = tmp_path / "dataset"
    source_manifest = create_example_dataset(dataset_dir, num_images=1, size=32)
    bench_manifest = build_benchmark_from_existing_manifest(source_manifest, tmp_path / "benchmark", copy_files=True, overwrite=True)
    out = tmp_path / "ours_variants"
    run_ours_variants_benchmark(
        bench_manifest,
        _fast_cfg(32),
        "configs/ours_variants.yaml",
        out,
        variant_names=["full", "visible_only"],
        max_images=1,
        use_frontend_cache=False,
    )
    assert (out / "variant_summary.json").exists()


def test_save_ours_output_protocol(tmp_path):
    dataset_dir = tmp_path / "dataset"
    manifest = create_example_dataset(dataset_dir, num_images=1, size=32)
    record = load_manifest(manifest)[0]
    image = load_image(record.image_path, size=32)
    cfg = _fast_cfg(32)
    front = build_frontend(image, cfg, record=record)
    trainer = Trainer(front, cfg)
    render = trainer.render()
    out = tmp_path / "protocol"
    info = save_ours_output(out, front.image, front.masks, render, {"AttrAcc": 1.0, "Leakage": 0.0}, trainer.scene, cfg, {}, image_id=record.image_id)
    assert (out / "output_spec.json").exists()
    assert (out / "render_final.png").exists()
    assert (out / "alpha_final.png").exists()
    assert (out / "ownership.npy").exists()
    assert info["variant"] == "full"

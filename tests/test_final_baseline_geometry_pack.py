from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.baselines.dummy_baselines import identity_mask_baseline, save_baseline_prediction
from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.image_io import load_image
from shapesplat.datasets.benchmark.builder_v2 import build_benchmark_from_existing_manifest
from shapesplat.datasets.example_dataset import create_example_dataset
from shapesplat.datasets.manifest import load_manifest
from shapesplat.evaluation.alignment import apply_alignment
from shapesplat.evaluation.geometry_metrics import chamfer_l2, compute_geometry_metrics, fscore, save_pointcloud
from shapesplat.evaluation.method_output_evaluator import evaluate_method_output
from shapesplat.experiments.final_comparison import run_final_comparison
from shapesplat.frontend.file_mask_loader import load_mask_file
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.optimization.trainer import Trainer
from shapesplat.reconstruction.output_protocol import export_scene_pointcloud
from shapesplat.reporting.final_tables import export_final_tables


@pytest.fixture
def tmp_path(request):
    root = ROOT / "outputs" / "test_final_baseline_geometry_pack_tmp" / request.node.name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cfg(size: int = 32) -> dict:
    cfg = merge_config(
        DEFAULT_CONFIG,
        {
            "device": "cpu",
            "image": {"size": size},
            "frontend": {"mask_source": "file"},
            "training": {"visible_warmup_iters": 1, "hidden_prior_iters": 1, "joint_ownership_iters": 1, "edit_finetune_iters": 1},
            "gaussians": {"visible_min": 8, "visible_max": 12, "hidden_base": 2},
        },
    )
    cfg["device"] = "cpu"
    return cfg


def _dataset_with_output(tmp_path):
    dataset_dir = tmp_path / "dataset"
    manifest = create_example_dataset(dataset_dir, num_images=1, size=32)
    record = load_manifest(manifest)[0]
    image = load_image(record.image_path, size=32)
    masks = load_mask_file(record.metadata["mask_path"], image_hw=image.shape[-2:], cfg=_cfg()).masks
    out = tmp_path / "method" / "per_image" / record.image_id
    pred = identity_mask_baseline(image, masks)
    save_baseline_prediction(pred, out, "identity_mask", record.image_id)
    return manifest, record, image, masks, out


def test_geometry_metrics_small_pointcloud():
    pred = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gt = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    assert float(chamfer_l2(pred, gt)) >= 0.0
    assert 0.0 <= float(fscore(pred, gt, threshold=0.2)) <= 1.0
    metrics = compute_geometry_metrics(pred, gt, threshold=0.2)
    assert metrics["available"] is True
    assert "ChamferL2" in metrics


def test_alignment_modes():
    pred = torch.rand(5, 3)
    gt = torch.rand(6, 3)
    for mode in ["none", "center", "unit_bbox", "similarity_scale"]:
        a, b = apply_alignment(pred, gt, mode)
        assert a.shape == pred.shape
        assert b.shape == gt.shape


def test_export_scene_pointcloud(tmp_path):
    manifest = create_example_dataset(tmp_path / "dataset", num_images=1, size=32)
    record = load_manifest(manifest)[0]
    cfg = _cfg(32)
    image = load_image(record.image_path, size=32)
    front = build_frontend(image, cfg, record=record)
    trainer = Trainer(front, cfg)
    path = tmp_path / "pred_pointcloud.npy"
    points = export_scene_pointcloud(trainer.scene, path)
    assert path.exists()
    assert points.shape[1] == 3


def test_evaluate_method_output_without_geometry(tmp_path):
    _manifest, record, image, masks, out = _dataset_with_output(tmp_path)
    metrics = evaluate_method_output("identity_mask", out, image, masks, record=record, cfg=_cfg())
    assert "AttrAcc" in metrics
    assert metrics["GeometryAvailable"] is False


def test_evaluate_method_output_with_geometry(tmp_path):
    manifest, record, image, masks, out = _dataset_with_output(tmp_path)
    gt_path = tmp_path / "gt.npy"
    pred_path = out / "pred_pointcloud.npy"
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")
    save_pointcloud(pts, gt_path)
    save_pointcloud(pts, pred_path)
    record.metadata["gt_pointcloud_path"] = str(gt_path)
    metrics = evaluate_method_output("identity_mask", out, image, masks, record=record, cfg={"geometry": {"alignment": "none"}})
    assert metrics["GeometryAvailable"] is True
    assert "ChamferL2" in metrics


def test_run_final_comparison_small(tmp_path):
    manifest, record, image, masks, out = _dataset_with_output(tmp_path)
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "methods:\n"
        "  - name: identity_mask\n"
        "    family: dummy\n"
        "    output_type: alpha_render\n"
        "    native_object_buffers: false\n"
        "    editable: false\n"
        "    source: dummy\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    result = run_final_comparison(manifest, catalog, {"identity_mask": str(tmp_path / "method")}, _cfg(32), tmp_path / "final", max_images=1)
    assert result["method_summary"]
    assert (tmp_path / "final" / "final_method_summary.json").exists()


def test_export_final_tables(tmp_path):
    summary = tmp_path / "summary.json"
    rows = [{"method": "ours_full", "family": "ours", "num_success": 1, "AttrAcc_mean": 1.0, "ChamferL2_mean": 0.0, "FScore_mean": 1.0}]
    summary.write_text(json.dumps(rows), encoding="utf-8")
    written = export_final_tables(summary, tmp_path / "tables")
    assert Path(written["main_comparison"]["csv"]).exists()
    assert Path(written["geometry"]["tex"]).exists()

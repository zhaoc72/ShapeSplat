from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.evaluation.edit_metrics import compute_edit_metrics
from shapesplat.evaluation.metrics import compute_basic_metrics
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.renderer.backend import build_renderer


def _cfg(size: int = 32):
    cfg = merge_config(DEFAULT_CONFIG, {"device": "cpu", "image": {"size": size}, "gaussians": {"visible_min": 8, "visible_max": 16, "hidden_base": 4}})
    cfg["device"] = "cpu"
    return cfg


def _mini_pipeline():
    cfg = _cfg()
    front = build_frontend(make_synthetic_image(int(cfg["image"]["size"])), cfg)
    scene = initialize_scene(front, cfg)
    renderer = build_renderer(front.camera, cfg)
    render = renderer(scene)
    return cfg, front, scene, renderer, render


def test_basic_metrics_shapes():
    _, front, _, _, render = _mini_pipeline()
    metrics = compute_basic_metrics(render, front.masks)
    for key in ("InstIoU_mean", "AttrAcc", "AttrPurity_mean", "Leakage"):
        assert key in metrics
    assert len(metrics["InstIoU_per_object"]) == front.masks.shape[0]
    assert len(metrics["AttrPurity_per_object"]) == front.masks.shape[0]


def test_metric_value_ranges():
    _, front, _, _, render = _mini_pipeline()
    metrics = compute_basic_metrics(render, front.masks)
    assert 0 <= metrics["AttrAcc"] <= 1
    assert 0 <= metrics["Leakage"] <= 1
    assert math.isfinite(metrics["InstIoU_mean"])
    assert math.isfinite(metrics["AttrPurity_mean"])


def test_edit_metrics():
    cfg, front, scene, renderer, render = _mini_pipeline()
    metrics = compute_edit_metrics(scene, renderer, front, render, cfg)
    for key in ("CollateralL1", "EditLocality", "DeletionResidual"):
        assert key in metrics
        assert math.isfinite(metrics[key])


def test_evaluate_script_smoke():
    """轻量测试 report 保存路径；完整 CLI 在手动/集成命令中验证。"""
    from shapesplat.evaluation.report import save_metrics_json

    out_dir = ROOT / "outputs" / "test_evaluation_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    save_metrics_json({"AttrAcc": 1.0, "InstIoU_per_object": [0.5]}, path)
    assert path.exists()

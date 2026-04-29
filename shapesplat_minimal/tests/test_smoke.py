from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

import shapesplat
from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.optimization.losses import compute_losses
from shapesplat.renderer.soft_renderer import SoftGaussianRenderer


def _cfg():
    cfg = merge_config(DEFAULT_CONFIG, {"device": "cpu", "image": {"size": 32}, "training": {"visible_warmup_iters": 1}})
    cfg["device"] = "cpu"
    return cfg


def test_import_shapesplat():
    assert shapesplat.__version__


def test_synthetic_image():
    img = make_synthetic_image(32)
    assert img.shape == (3, 32, 32)
    assert 0 <= float(img.min()) <= float(img.max()) <= 1


def test_frontend():
    cfg = _cfg()
    front = build_frontend(make_synthetic_image(32), cfg)
    assert front.masks.shape[0] >= 1
    assert front.descriptors.shape[0] == front.masks.shape[0]
    assert front.depth.shape == (32, 32)


def test_initialize_scene():
    cfg = _cfg()
    front = build_frontend(make_synthetic_image(32), cfg)
    scene = initialize_scene(front, cfg)
    assert len(scene.objects) == front.masks.shape[0]
    assert scene.all_parameters()["means"].shape[1] == 3


def test_renderer():
    cfg = _cfg()
    front = build_frontend(make_synthetic_image(32), cfg)
    scene = initialize_scene(front, cfg)
    renderer = SoftGaussianRenderer(front.camera)
    out = renderer(scene)
    assert out.rgb.shape == (3, 32, 32)
    assert out.alpha.shape == (32, 32)
    assert out.depth.shape == (32, 32)
    assert out.contributions.shape == (front.masks.shape[0], 32, 32)
    assert out.ownership.shape == (front.masks.shape[0], 32, 32)
    assert out.bg_ownership.shape == (32, 32)


def test_one_train_step():
    cfg = _cfg()
    front = build_frontend(make_synthetic_image(32), cfg)
    scene = initialize_scene(front, cfg)
    renderer = SoftGaussianRenderer(front.camera)
    render = renderer(scene)
    loss, terms = compute_losses(scene, renderer, render, front, cfg, stage="visible")
    loss.backward()
    assert torch.isfinite(loss)
    assert "total" in terms
    assert any(p.grad is not None for p in scene.parameters())

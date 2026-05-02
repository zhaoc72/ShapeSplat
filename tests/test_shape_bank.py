from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from create_toy_shape_bank import create_toy_shape_bank
from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.optimization.losses import compute_losses
from shapesplat.renderer.backend import build_renderer
from shapesplat.shape_prior.file_shape_bank import FileShapeBank
from shapesplat.shape_prior.retrieval import retrieve_shapes
from shapesplat.shape_prior.shape_bank_backend import build_shape_bank
from shapesplat.shape_prior.toy_shape_bank import ToyShapeBank


def _cfg(overrides=None):
    cfg = merge_config(DEFAULT_CONFIG, {"device": "cpu", "image": {"size": 32}})
    if overrides:
        cfg = merge_config(cfg, overrides)
    cfg["device"] = "cpu"
    return cfg


def test_toy_shape_bank():
    cfg = _cfg()
    bank = build_shape_bank(cfg, descriptor_dim=9, device=torch.device("cpu"))
    assert len(bank.assets) >= 3
    for asset in bank.assets:
        assert asset.points.ndim == 2 and asset.points.shape[1] == 3
        assert asset.descriptor is not None
        assert torch.isfinite(asset.descriptor).all()


def test_create_and_load_file_shape_bank():
    # 这里不用 pytest tmp_path：某些 Windows 环境的临时目录权限较严格。
    root = ROOT / "outputs" / "test_shape_bank" / "load_file"
    create_toy_shape_bank(root, num_points=64, descriptor_dim=9)
    cfg = _cfg({"shape_bank": {"backend": "file", "root": str(root), "descriptor_dim": 9, "num_points": 32}})
    bank = FileShapeBank(cfg, descriptor_dim=9, device=torch.device("cpu"))
    assert len(bank.assets) == 3
    assert bank.assets[0].points.shape == (32, 3)
    assert bank.assets[0].descriptor.shape == (9,)


def test_retrieve_shapes():
    bank = ToyShapeBank(descriptor_dim=9, device=torch.device("cpu"), points_per_shape=32)
    descriptors = torch.randn(2, 9)
    descriptors = torch.nn.functional.normalize(descriptors, dim=1)
    retrieved, weights, confidence = retrieve_shapes(descriptors, bank, top_k=2)
    assert len(retrieved) == 2
    assert weights.shape == (2, 2)
    assert confidence.shape == (2,)
    assert torch.isfinite(confidence).all()


def test_auto_shape_bank_fallback():
    cfg = _cfg(
        {
            "shape_bank": {
                "backend": "auto",
                "root": "this/path/does/not/exist",
                "fallback_to_toy": True,
            }
        }
    )
    bank = build_shape_bank(cfg, descriptor_dim=9, device=torch.device("cpu"))
    assert isinstance(bank, ToyShapeBank)


def test_run_with_file_shape_bank():
    root = ROOT / "outputs" / "test_shape_bank" / "run_file"
    create_toy_shape_bank(root, num_points=64, descriptor_dim=16)
    cfg = _cfg(
        {
            "frontend": {"dino_feature_dim": 16},
            "shape_bank": {
                "backend": "file",
                "root": str(root),
                "descriptor_dim": 16,
                "num_points": 64,
                "fallback_to_toy": False,
            },
            "training": {
                "visible_warmup_iters": 1,
                "hidden_prior_iters": 0,
                "joint_ownership_iters": 0,
                "edit_finetune_iters": 0,
            },
        }
    )
    image = make_synthetic_image(32)
    front = build_frontend(image, cfg)
    scene = initialize_scene(front, cfg)
    renderer = build_renderer(front.camera, cfg)
    render = renderer(scene)
    loss, terms = compute_losses(scene, renderer, render, front, cfg, stage="visible")
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in scene.parameters())

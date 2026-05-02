from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.dino_backend import build_dino_backend
from shapesplat.frontend.dino_pooling import pool_mask_descriptors
from shapesplat.frontend.dinov3_stub import DinoV3Stub
from shapesplat.frontend.sam3_stub import Sam3Stub


def _cfg(overrides=None):
    cfg = merge_config(DEFAULT_CONFIG, {"device": "cpu", "image": {"size": 32}})
    if overrides:
        cfg = merge_config(cfg, overrides)
    cfg["device"] = "cpu"
    return cfg


def test_build_stub_dino_backend():
    cfg = _cfg({"frontend": {"dino_backend": "stub"}})
    backend = build_dino_backend(cfg)
    assert hasattr(backend, "extract_dense_features")
    assert hasattr(backend, "pool_descriptors")


def test_stub_dino_outputs_features_and_descriptors():
    cfg = _cfg({"frontend": {"dino_backend": "stub"}})
    image = make_synthetic_image(32)
    masks = Sam3Stub(max_num_objects=4).predict_masks(image).masks
    dino = build_dino_backend(cfg)
    feats = dino.extract_dense_features(image)
    desc = dino.pool_descriptors(feats, masks)
    assert feats.ndim == 3
    assert feats.shape[-2:] == (32, 32)
    assert desc.shape == (masks.shape[0], feats.shape[0])
    assert torch.isfinite(desc).all()
    assert torch.allclose(desc.norm(dim=1), torch.ones(desc.shape[0]), atol=1e-4)


def test_auto_dino_fallback_without_checkpoint():
    cfg = _cfg({"frontend": {"dino_backend": "auto", "dino_checkpoint": "not_exist_dino_checkpoint.pt"}})
    backend = build_dino_backend(cfg)
    assert isinstance(backend, DinoV3Stub)
    feats = backend.extract_dense_features(make_synthetic_image(32))
    assert feats.ndim == 3


def test_dino_pooling_handles_small_masks():
    cfg = _cfg()
    features = torch.rand(8, 16, 16)
    masks = torch.zeros(1, 16, 16)
    masks[0, 7, 7] = 1.0
    desc = pool_mask_descriptors(features, masks, cfg)
    assert desc.shape == (1, 8)
    assert torch.isfinite(desc).all()
